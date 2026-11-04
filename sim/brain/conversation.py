"""The conversation on top of the behaviour engine (ADR-004, layer "raciocínio e conversa").

    uv run python sim/brain/conversation.py --test                    # the dialogue scenarios, local 3B model
    uv run python sim/brain/conversation.py --test --model qwen3.6:35b-mlx --think
    uv run mjpython sim/brain/conversation.py --memory build/brain_memory   # viewer: talk to it in the terminal
    uv run python sim/brain/conversation.py --test --brain elevenlabs  # the same with the ElevenLabs agent
    uv run mjpython sim/brain/conversation.py --brain elevenlabs --voz # viewer: talk to it by voice

What the person says goes, in this order:
1. **the fixed phrases** (behavior.parse: pare, fica, vem, siga, senta, "vai até a bola", "cadê a bola?", ...), when
   short (up to 4 words besides "Cogu"; a sentence starting with a stop word, any length): run at once, no language
   model. "Pare" never waits for one (ADR-004, item 6), and these work offline;
2. **the language model**, with the tools of behavior.Tools as functions and the world memory's text summary as
   context. It never moves a motor: it calls tools, and only says it did what a tool did. Moving tools called in
   the same turn run one after the other ("vai até a bola e depois volta aqui");
3. if the model doesn't answer: "Agora não consigo pensar; só entendo os comandos simples."

The model is a local one through Ollama (`--model`, default qwen2.5:3b, about the size of the iPhone's on-device
model) or the ElevenLabs agent (`--brain elevenlabs`, eleven.py: a cloud model with these tools as client tools). On
the phone the same loop calls Apple's on-device model (Foundation Models) and, when that falls short, the ElevenLabs
agent (ADR-004, item 4); both take the same tools and the same summary. Only text goes to a model: the summary and
what was said. `--voz` adds ElevenLabs' ears (Scribe) and voice (text to speech) around the same loop.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import behavior as beh  # noqa: E402
import pet as petlib  # noqa: E402
from memory import WorldMemory, plain  # noqa: E402

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"
HISTORY = 8           # messages of past dialogue the model sees (text only)
MAX_ROUNDS = 4        # model calls per turn (tool calls, then the answer)
LOCAL_WORDS = 4       # a fixed phrase this short runs without the model ("Cogu" doesn't count)

PERSONA = """Você é o Cogu, um robô de estimação de 40 cm que anda em duas pernas e conversa, como um cachorrinho \
esperto que fala. Fale português do Brasil, com uma ou duas frases curtas, carinhoso e curioso.
Você é um bicho de estimação, não um assistente: não ofereça ajuda nem pergunte "como posso ajudar".
Você não controla motores: para fazer qualquer coisa, chame uma ferramenta, e só diga que vai fazer algo se chamou a \
ferramenta. Para ir a uma coisa ou lugar, use ir_ate com o nome dele. Andar leva tempo: a ferramenta só começa o \
movimento, então diga que está indo, nunca que já chegou.
Quando a pessoa disser "vem comigo" ou "me acompanha", use a ferramenta seguir; "descansa", a ferramenta sentar. \
Para várias ações seguidas, chame todas as ferramentas, na ordem, na mesma resposta (para "vai até o sofá e depois \
volta aqui", chame ir_ate com alvo sofá e depois vir). Nunca escreva o nome de uma ferramenta na resposta: chame-a.
Chame só o que a pessoa pediu nesta fala, nada a mais: "volta pro corredor" é só ir_ate corredor (vir só quando ela pedir para você vir até ela). Se não conhece o lugar pedido, não chame nenhuma ferramenta. Diga que está indo só se chamou ir_ate, vir ou seguir nesta mesma fala.
Responda sobre onde estão as coisas com o que está no contexto (o que você sabe). Não invente coisas nem lugares que \
não estão nele: se não sabe onde fica um lugar, diga e peça para te mostrarem ("me leva lá e diz 'aqui é ...'").
Você não tem mãos, não sobe em móveis e não pula: se pedirem, explique com carinho."""

TOOLS = {   # name: (description, {argument: description})
    "parar": ("Para de andar na hora e fica parado até a próxima ordem.", {}),
    "ficar": ("Fica onde está, virado para a pessoa.", {}),
    "vir": ("Vem até a pessoa e para na frente dela.", {}),
    "seguir": ("Segue a pessoa por onde ela for.", {}),
    "livre": ("Sem ordem: pode fazer o que quiser (descansar, procurar a pessoa).", {}),
    "procurar": ("Procura a pessoa (onde ela foi vista por último, depois olhando em volta).", {}),
    "ir_ate": ("Anda até um lugar com nome (sala, corredor, ...) ou até uma coisa que você conhece (bola, sofá, "
               "mesa, porta, ...).", {"alvo": "o nome do lugar ou da coisa, sem artigo"}),
    "sentar": ("Senta no chão e fica sentado.", {}),
    "deitar": ("Deita de costas no chão.", {}),
    "levantar": ("Levanta (de sentado ou deitado).", {}),
    "onde_esta": ("Diz onde está uma coisa ou lugar, pela memória.", {"coisa": "o nome, sem artigo"}),
    "o_que_vejo": ("Diz o que a câmera está vendo agora.", {}),
    "lembrar_lugar": ("Dá um nome ao lugar onde você está agora. Só quando a pessoa disser 'aqui é ...' nesta fala; "
                      "nunca por conta própria.",
                      {"nome": "o nome do lugar, sem artigo"}),
}
OFFLINE = "Agora não consigo pensar; só entendo os comandos simples."
MOVING = ("vir", "seguir", "procurar", "ir_ate")
ACTIONS = MOVING + ("ficar", "livre", "sentar", "deitar", "levantar")   # after a move in the same turn: queued


def local_command(text: str) -> tuple[str, str | None] | None:
    """A short fixed phrase (behavior.parse) is run without the model; a longer sentence goes to the model even if
    it has a command word in it ("vai até o sofá e depois volta aqui" is not "vem"), except a stop."""
    words = [w.removeprefix("cogu") if len(w) > 6 else w      # (speech to text glued "Cogu, pare" once: "cogupari")
             for w in plain(text).replace(",", " ").replace("!", " ").replace("?", " ").replace(".", " ").split()
             if w != "cogu"]
    cmd = beh.parse(" ".join(words))
    if cmd is None:
        return None
    if len(words) <= LOCAL_WORDS or (cmd[0] == "parar" and words[0] in ("pare", "para", "stop", "chega")):
        return cmd
    return None


def tool_schemas() -> list[dict]:
    return [{"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": {a: {"type": "string", "description": d}
                                                        for a, d in args.items()}, "required": list(args)}}}
            for name, (desc, args) in TOOLS.items()]


class Ollama:
    """A chat model with tools, through a local Ollama server."""

    def __init__(self, model: str = MODEL, think: bool = False, timeout: float = 120.0):
        self.model, self.think, self.timeout = model, think, timeout

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages, "tools": tools, "stream": False,
                "options": {"temperature": 0.2, "num_ctx": 8192}, "keep_alive": "30m"}   # (qwen3's default
        # context, 256 k tokens, takes ~45 GB and makes it ~50x slower)
        if self.model.startswith("qwen3"):          # (without thinking, qwen3.6 talks about acting and calls nothing)
            body["think"] = self.think
        req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["message"]


class Conversation:
    """`hear(text)` -> what the Cogu says. `turns` keeps, per turn, which path answered, the tools called and the
    model's time."""

    def __init__(self, behavior: beh.Behavior, model=None):
        self.behavior, self.model = behavior, model
        self.history: list[dict] = []
        self.turns: list[dict] = []
        self.lock = contextlib.nullcontext()            # the viewer's lock: the model thinks in another thread
        self.voice = self.ears = None                   # eleven.Voice / eleven.Ears: speak and listen (--voz)
        behavior.pet.conversation = self

    def hear(self, text: str) -> str:
        b = self.behavior
        fixed = local_command(text)
        if fixed is not None:                           # a fixed phrase: no model
            with self.lock:
                reply = b.hear(text)
            self._remember(text, reply)
            self.turns.append({"text": text, "path": "local", "tools": [f"{fixed[0]}({fixed[1] or ''})"],
                               "seconds": 0.0, "reply": reply})
            return reply
        with self.lock:
            b.log.append((b.pet.t, "pessoa", text))
        t0, called = time.perf_counter(), []
        try:
            reply = self._think(text, called)
        except Exception as e:                          # no model (offline, not running, too slow)
            reply = OFFLINE
            called.append(f"(erro: {type(e).__name__})")
        with self.lock:
            b.say(reply)
        self._remember(text, reply)
        self.turns.append({"text": text, "path": "model", "tools": called,
                           "seconds": time.perf_counter() - t0, "reply": reply})
        return reply

    def _context(self) -> str:
        with self.lock:
            return self.behavior.tools.resumo()

    def _think(self, text: str, called: list) -> str:
        if self.model is None:
            raise RuntimeError("no model")
        state = {"moved": False, "done": set(), "said": [], "text": text}

        def call(name: str, args: dict) -> str:
            return self._call(name, args, state, called)

        if hasattr(self.model, "turn"):     # a model that keeps the dialogue itself (the ElevenLabs agent, eleven.py)
            reply = self.model.turn(text, self._context(), call)
            return reply.strip() or " ".join(state["said"]) or "Hum?"
        messages = [{"role": "system", "content": PERSONA},
                    *self.history[-HISTORY:],
                    {"role": "user", "content": f"[o que você sabe agora: {self._context()}]\n{text}"}]
        for _ in range(MAX_ROUNDS):
            msg = self.model.chat(messages, tool_schemas())
            calls = msg.get("tool_calls") or []
            if not calls:
                return (msg.get("content") or "").strip() or " ".join(state["said"]) or "Hum?"
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for c in calls:
                name, args = c["function"]["name"], c["function"].get("arguments") or {}
                messages.append({"role": "tool", "content": call(name, args), "tool_name": name})
        return " ".join(state["said"])

    def _call(self, name: str, args, state: dict, called: list) -> str:
        """One tool call of the model: run it (a move after another one this turn waits in the queue)."""
        if isinstance(args, str):
            args = json.loads(args or "{}")
        key = (name, json.dumps(args, sort_keys=True))
        if key in state["done"]:                        # small models repeat a call after its result: once is enough
            return "(já feito)"
        state["done"].add(key)
        if name == "lembrar_lugar" and "aqui" not in plain(state["text"]).split():   # (a model named the sofa
            return "(não: só quando a pessoa disser 'aqui é ...')"                       # corner "cozinha" alone)
        result = self._run(name, args, queue=state["moved"] and name in ACTIONS)
        state["moved"] = state["moved"] or name in MOVING
        called.append(f"{name}({', '.join(str(v) for v in args.values())})")
        state["said"].append(result)
        return result

    def _run(self, name: str, args: dict, queue: bool) -> str:
        if name not in TOOLS:
            return f"(não existe a ferramenta {name})"
        fn = getattr(self.behavior.tools, name)
        params = list(TOOLS[name][1])
        values = [str(args.get(p, "")) for p in params]
        with self.lock:
            if queue:                                   # after another move this turn: when that one ends
                self.behavior.queue.append((name, values))
                return f"(depois que eu terminar: {name} {' '.join(values)})".strip()
            if name in ACTIONS or name == "parar":      # a new order replaces what was lined up
                self.behavior.queue.clear()
            return fn(*values)

    def _remember(self, text: str, reply: str):
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]


# -- the scenarios ---------------------------------------------------------------------------------------------
# (what is said, the tool calls that are right: any one of these, as "name(argument)" sets; seconds to run after)
DIALOGUE = [
    ("Oi Cogu, tudo bem com você?", [set()], 2.0),
    ("Você consegue ir lá perto da bola pra mim?", [{"ir_ate(bola)"}], 20.0),
    ("Onde fica a mesa?", [set(), {"onde_esta(mesa)"}], 1.0),
    ("Agora vai até o sofá e depois volta aqui pertinho de mim", [{"ir_ate(sofá)", "vir()"}], 45.0),
    ("Sobe no sofá!", [set()], 1.0),
    ("Vai pra cozinha buscar um biscoito", [set()], 1.0),
    ("Pare!", [{"parar()"}], 1.0),
    ("Estou indo pro corredor, vem comigo", [{"seguir()"}], 3.0),
    ("Que bom que você achou a bola. Agora descansa um pouco", [{"sentar()"}, {"deitar()"}, {"livre()"}], 6.0),
]


# harder: actions after a move, places it must not invent, questions to answer from the context
DIALOGUE_HARD = [
    ("Cogu, vai até a porta e depois senta lá", [{"ir_ate(porta)", "sentar()"}], 40.0),
    ("Levanta, vamos brincar mais um pouco", [{"levantar()"}, {"levantar()", "ir_ate(bola)"}], 4.0),
    ("Você lembra onde eu deixei a bola?", [set(), {"onde_esta(bola)"}], 1.0),
    ("Me leva até a planta, quero ver se precisa de água", [{"ir_ate(planta)"}], 30.0),
    ("Aqui é o cantinho da planta", [{"lembrar_lugar(cantinho da planta)"}, {"lembrar_lugar(cantinho)"}], 1.0),
    ("O que você tá vendo na sua frente agora?", [{"o_que_vejo()"}], 1.0),
    ("Fica aí quietinho que eu já volto", [{"ficar()"}, {"parar()"}], 2.0),
    ("Quantas pernas você tem?", [set()], 1.0),
    ("Vai pro quarto e me espera lá", [set()], 1.0),
    ("Volta pro corredor, por favor", [{"ir_ate(corredor)"}], 30.0),
]


def run_dialogue(model, label: str, memory_dir: Path = beh.MEMORY_DIR, dialogue=None, **opts) -> int:
    pet = petlib.Pet(**opts)
    pet.person.place(1.5, -0.6, math.pi)
    memory = WorldMemory.load(memory_dir / "memory.json", t0=60.0) if (memory_dir / "memory.json").exists() else None
    if memory is not None:
        pet.perception.load_map(memory_dir / "map.npz")
    b = beh.Behavior(pet, memory, autonomy=False)
    conv = Conversation(b, model)
    pet.run(1.0)
    ok = 0
    print(f"model {label}, "
          f"{'memory from ' + str(memory_dir) if memory else 'no memory'}")
    dialogue = dialogue or DIALOGUE
    for text, right, seconds in dialogue:
        reply = conv.hear(text)
        turn = conv.turns[-1]
        good = {plain(c) for c in turn["tools"]} in [{plain(c) for c in r} for r in right]
        ok += good
        print(f"  > {text}\n    [{turn['path']}, {turn['seconds']:.1f} s, tools: {', '.join(turn['tools']) or '-'}]"
              f" {'ok' if good else 'NOT AS EXPECTED'}\n    Cogu: {reply}")
        pet.run(seconds)
        if b.queue or pet.brain.mode in ("ir", "vem"):
            beh.until(pet, lambda p: not b.queue and p.brain.mode not in ("ir", "vem"), 40.0)
        if turn["tools"]:
            print(f"    (then: {pet.brain.said}; {pet.distance():.2f} m from the person)")
    print(f"  {ok}/{len(dialogue)} as expected; falls {pet.falls}, bumps {pet.bumps}; model time "
          f"{sum(t['seconds'] for t in conv.turns if t['path'] == 'model'):.0f} s in "
          f"{sum(t['path'] == 'model' for t in conv.turns)} turns"
          + (f"; first words after {sum(model.first) / len(model.first):.1f} s (mean), "
             f"{max(model.first):.1f} s (max)" if getattr(model, "first", None) else ""))
    pet.close()
    if hasattr(model, "close"):
        model.close()
    return 0


def make_model(args):
    if args.brain == "elevenlabs":
        import eleven
        return eleven.Agent(), f"ElevenLabs agent ({eleven.load_agent()['llm']})"
    return Ollama(args.model, args.think), f"{args.model}{' (thinking)' if args.think else ''}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--brain", choices=("ollama", "elevenlabs"), default="ollama",
                    help="the language model: a local one (Ollama) or the ElevenLabs agent (eleven.py --setup first)")
    ap.add_argument("--model", default=MODEL, help="an Ollama model with tool calling")
    ap.add_argument("--voz", action="store_true",
                    help="viewer: Enter (empty line) starts and stops the microphone, the Cogu speaks (ElevenLabs)")
    ap.add_argument("--think", action="store_true", help="let a qwen3 model think first (it needs it to call tools)")
    ap.add_argument("--perception", choices=("phone", "true"), default="phone")
    ap.add_argument("--medula", choices=petlib.MEDULAS, default="py", help=petlib.MEDULA_HELP)
    ap.add_argument("--link", choices=tuple(petlib.PROFILES), default="ideal",
                    help="the radio between the phone and the medula (medula.py)")
    ap.add_argument("--memory", default=None, metavar="DIR", help="load memory.json and map.npz from DIR first")
    ap.add_argument("--hard", action="store_true", help="--test: the harder dialogue (DIALOGUE_HARD)")
    ap.add_argument("--no-autonomy", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args(argv)
    opts = dict(perception=args.perception, link=args.link, medula=args.medula)
    try:
        model, label = make_model(args)
    except RuntimeError as e:                           # (no ElevenLabs agent yet)
        print(f"  {e}")
        return 1
    if args.test:
        return run_dialogue(model, label, Path(args.memory) if args.memory else beh.MEMORY_DIR,
                            DIALOGUE_HARD if args.hard else DIALOGUE, **opts)
    pet = petlib.Pet(**opts)
    memory = None
    if args.memory:
        memory = WorldMemory.load(Path(args.memory) / "memory.json", t0=3600.0)
        pet.perception.load_map(Path(args.memory) / "map.npz")
    conv = Conversation(beh.Behavior(pet, memory, autonomy=not args.no_autonomy), model)
    if args.voz:
        import eleven
        conv.voice, conv.ears = eleven.Voice(), eleven.Ears()
    return petlib.run_viewer(pet, args.speed)


if __name__ == "__main__":
    sys.exit(main())
