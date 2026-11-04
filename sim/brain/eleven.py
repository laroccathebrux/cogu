"""ElevenLabs around the Cogu's conversation (ADR-004, item 4): the cloud brain, the ears and the voice.

    export ELEVENLABS_API_KEY=...       # your key (or in ~/.config/cogu/elevenlabs_key); nothing here writes it
    uv run python sim/brain/eleven.py --voices                     # the voices of your account (pick one)
    uv run python sim/brain/eleven.py --setup [--voice ID] [--llm claude-haiku-5-5]   # create / update the agent
    uv run python sim/brain/eleven.py --say "Oi, eu sou o Cogu!"   # hear the voice
    uv run python sim/brain/eleven.py --listen                     # speak, Enter: what Scribe understood
    uv run python sim/brain/eleven.py --selftest                   # the whole client against a fake local server
    uv run python sim/brain/conversation.py --test --brain elevenlabs        # the 9-turn dialogue with the agent
    uv run mjpython sim/brain/conversation.py --brain elevenlabs --voz       # viewer: talk to it by voice

Three pieces:
- **Agent**: the language model of the ElevenLabs agents platform, as conversation.Conversation's model. `--setup`
  creates an agent with conversation.py's persona and its tools as *client tools*: the model asks for a tool, this
  program runs it (on the phone: the app) and sends back the result; the model never reaches a motor. The agent is
  text only (no audio billed, the voice is the piece below); before each phrase what the Cogu knows now (the world
  memory's summary) goes as a contextual update. The dialogue's history stays in the agent's conversation.
- **Ears**: the microphone (ffmpeg) -> Scribe (speech to text). Push to talk: Enter starts, Enter stops.
- **Voice**: text to speech (eleven_flash_v2_5: ~0.1 s, Portuguese) -> the speaker (afplay), in order, in a thread.

The ears and voice go around any brain (the local model too). The fixed phrases are still matched locally on the
transcript, so "pare" never waits for the cloud model; by voice it waits for the transcription (on the phone the
on-device recogniser hears it, ADR-004, item 6). Only text goes to the agent: the summary and what was said.

`build/elevenlabs/agent.json` keeps the agent's and tools' ids (no key). The key is read from the environment only.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import requests
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

API = os.environ.get("ELEVENLABS_BASE", "https://api.elevenlabs.io")
CONFIG = ROOT / "build" / "elevenlabs" / "agent.json"
LLM = "claude-haiku-5-5"        # the agent's model: fast (it talks); --llm claude-sonnet-5-5 for a smarter one
VOICE = "cjVigY5qzO86Huf0OWal"  # the agents platform's default voice, until --voice picks one
TTS_MODEL = "eleven_flash_v2_5"  # ~0.1 s, 32 languages (eleven_v4_turbo: as fast, more expressive, pricier)
STT_MODEL = "scribe_v2"
LANGUAGE = "pt"
QUIET = 0.8           # s without news after the agent's answer: the turn is over
TURN_TIMEOUT = 30.0   # s for the agent's answer (then: "Agora não consigo pensar")

PERSONA_EXTRA = """
O que você sabe agora (onde está, a pessoa, as coisas e lugares de que lembra) chega antes de cada fala como \
atualização de contexto, começando com "O que você sabe agora:". Use só isso para dizer onde estão as coisas."""


class NoKey(RuntimeError):
    pass


KEY_FILE = Path.home() / ".config" / "cogu" / "elevenlabs_key"   # outside the repo; written by you (README)


def key() -> str:
    """ELEVENLABS_API_KEY, or else the file KEY_FILE (so every shell finds it). Never printed."""
    k = os.environ.get("ELEVENLABS_API_KEY")
    if not k and KEY_FILE.exists():
        k = KEY_FILE.read_text().strip()
    if not k:
        raise NoKey(f"sem chave da ElevenLabs: defina ELEVENLABS_API_KEY ou grave a chave em {KEY_FILE} "
                    "(sim/brain/README.md, seção ElevenLabs)")
    return k


class Api:
    """The REST calls (the key goes in the xi-api-key header, never in a URL or a file)."""

    def __init__(self, base: str | None = None, api_key: str | None = None):
        self.base, self.api_key = (base or API).rstrip("/"), api_key
        self.session = requests.Session()

    def request(self, method: str, path: str, raw: bool = False, **kw):
        headers = {"xi-api-key": self.api_key or key()}
        r = self.session.request(method, self.base + path, headers=headers, timeout=kw.pop("timeout", 30), **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"ElevenLabs {method} {path}: {r.status_code} {r.text[:400]}")
        return r.content if raw else r.json()

    def get(self, path: str, **params):
        return self.request("GET", path, params=params)

    def post(self, path: str, body=None, **kw):
        return self.request("POST", path, json=body, **kw) if body is not None else self.request("POST", path, **kw)

    def patch(self, path: str, body):
        return self.request("PATCH", path, json=body)


# -- the agent ---------------------------------------------------------------------------------------------------
def load_agent(path: Path = CONFIG, missing_ok: bool = False) -> dict | None:
    if not Path(path).exists():
        if missing_ok:
            return None
        raise RuntimeError(f"no agent yet: run  uv run python sim/brain/eleven.py --setup  ({path} is missing)")
    return json.loads(Path(path).read_text())


def tool_config(name: str) -> dict:
    import conversation as conv
    desc, args = conv.TOOLS[name]
    return {"type": "client", "name": name, "description": desc, "expects_response": True,
            "response_timeout_secs": 20,
            "parameters": {"type": "object", "required": list(args),
                           "properties": {a: {"type": "string", "description": d} for a, d in args.items()}}}


def agent_body(llm: str, voice: str, tool_ids: list[str]) -> dict:
    import conversation as conv
    return {"name": "Cogu", "tags": ["cogu"],
            "conversation_config": {
                "agent": {"first_message": "", "language": LANGUAGE,
                          "prompt": {"prompt": conv.PERSONA + PERSONA_EXTRA, "llm": llm, "temperature": 0.2,
                                     "tool_ids": tool_ids}},
                "tts": {"voice_id": voice, "model_id": TTS_MODEL},
                "conversation": {"text_only": True, "max_duration_seconds": 3600}}}


def setup(api: Api, llm: str = LLM, voice: str | None = None, path: Path = CONFIG) -> dict:
    """Create the agent and its client tools, or update them (the ids are kept in `path`)."""
    import conversation as conv
    cfg = load_agent(path, missing_ok=True) or {}
    voice = voice or cfg.get("voice_id") or VOICE
    tools = dict(cfg.get("tool_ids", {}))
    for name in conv.TOOLS:
        body = {"tool_config": tool_config(name)}
        if name in tools:
            try:
                api.patch(f"/v1/convai/tools/{tools[name]}", body)
            except RuntimeError:                         # deleted on the site: make it again
                tools.pop(name)
        if name not in tools:
            tools[name] = api.post("/v1/convai/tools", body)["id"]
        print(f"  tool {name}: {tools[name]}")
    body = agent_body(llm, voice, [tools[n] for n in conv.TOOLS])
    agent_id = cfg.get("agent_id")
    if agent_id:
        try:
            api.patch(f"/v1/convai/agents/{agent_id}", body)
        except RuntimeError:
            agent_id = None
    if not agent_id:
        agent_id = api.post("/v1/convai/agents/create", body)["agent_id"]
    cfg = {"agent_id": agent_id, "tool_ids": tools, "llm": llm, "voice_id": voice}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cfg, indent=1))
    print(f"  agent {agent_id} ({llm}, voice {voice}) -> {path}")
    return cfg


class Agent:
    """conversation.Conversation's model: `turn(text, context, call)` -> what the Cogu says. `call(name, args)`
    runs a tool (Conversation._call) and returns its result, which goes back to the agent."""

    def __init__(self, api: Api | None = None, config: dict | None = None):
        self.api, self.cfg = api or Api(), config or load_agent()
        self.ws = None
        self.inbox: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.conversation_id = None

    # the connection: a reader thread answers the pings at once and queues the rest
    def connect(self):
        url = self.api.get("/v1/convai/conversation/get-signed-url", agent_id=self.cfg["agent_id"])["signed_url"]
        self.ws = connect(url, open_timeout=15, max_size=None)
        self.inbox = queue.Queue()
        threading.Thread(target=self._reader, args=(self.ws, self.inbox), daemon=True).start()
        self.ws.send(json.dumps({"type": "conversation_initiation_client_data"}))
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            msg = self._next(deadline - time.monotonic())
            if msg is not None and msg.get("type") == "conversation_initiation_metadata":
                self.conversation_id = msg["conversation_initiation_metadata_event"].get("conversation_id")
                return
        raise TimeoutError("the agent didn't start the conversation")

    @staticmethod
    def _reader(ws, inbox: queue.Queue):
        try:
            for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "ping":
                    ws.send(json.dumps({"type": "pong", "event_id": msg["ping_event"]["event_id"]}))
                else:
                    inbox.put(msg)
        except ConnectionClosed:
            pass
        inbox.put({"type": "_closed"})

    def _next(self, timeout: float) -> dict | None:
        try:
            msg = self.inbox.get(timeout=max(timeout, 0.0)) if timeout > 0 else self.inbox.get_nowait()
        except queue.Empty:
            return None
        if msg.get("type") == "_closed":
            self.ws = None
            raise ConnectionClosed(None, None)
        return msg

    def close(self):
        if self.ws is not None:
            self.ws.close()
            self.ws = None

    # one turn
    def turn(self, text: str, context: str, call) -> str:
        with self.lock:
            for attempt in (0, 1):                       # the conversation may have ended (idle): start a new one
                try:
                    if self.ws is None:
                        self.connect()
                    return self._turn(text, context, call)
                except ConnectionClosed:
                    self.ws = None
                    if attempt:
                        raise
        return ""

    def _turn(self, text: str, context: str, call) -> str:
        while (msg := self._next(0)) is not None:        # left from the last turn (a late tool call runs now)
            self._handle(msg, call, [])
        self.ws.send(json.dumps({"type": "contextual_update", "text": f"O que você sabe agora: {context}"}))
        self.ws.send(json.dumps({"type": "user_message", "text": text}))
        replies: list[str] = []
        start = last = time.monotonic()
        waiting_tool = False                             # after a tool's result the agent answers again: wait for it
        while True:
            now = time.monotonic()
            if replies and not waiting_tool:
                wait = last + QUIET - now
            else:
                wait = start + TURN_TIMEOUT - now
                if wait <= 0 and replies:
                    break
                if wait <= 0:
                    raise TimeoutError("the agent didn't answer")
            if wait <= 0:
                break
            msg = self._next(wait)
            if msg is None:
                continue
            kind = self._handle(msg, call, replies)
            if kind == "tool":
                waiting_tool, last = True, time.monotonic()
            elif kind == "answer":
                waiting_tool, last = False, time.monotonic()
        return " ".join(r for r in replies if r)

    def _handle(self, msg: dict, call, replies: list) -> str | None:
        kind = msg.get("type")
        if kind == "client_tool_call":
            c = msg["client_tool_call"]
            try:
                result, error = call(c["tool_name"], c.get("parameters") or {}), False
            except Exception as e:                       # (a bad argument: the agent hears what went wrong)
                result, error = f"erro: {e}", True
            self.ws.send(json.dumps({"type": "client_tool_result", "tool_call_id": c["tool_call_id"],
                                     "result": str(result), "is_error": error}))
            return "tool"
        if kind == "agent_response":
            replies.append(msg["agent_response_event"]["agent_response"].strip())
            return "answer"
        if kind == "agent_response_correction" and replies:
            replies[-1] = msg["agent_response_correction_event"]["corrected_agent_response"].strip()
        return None


# -- the voice and the ears --------------------------------------------------------------------------------------
class Voice:
    """`say(text)`: spoken in order, in a thread (the brain doesn't wait). `seconds`: time to the audio, per phrase."""

    def __init__(self, api: Api | None = None, voice_id: str | None = None, play: bool = True):
        cfg = load_agent(missing_ok=True) or {}
        self.api, self.voice_id, self.play = api or Api(), voice_id or cfg.get("voice_id") or VOICE, play
        self.seconds: list[float] = []
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._speaker, daemon=True).start()

    def audio(self, text: str) -> bytes:
        t0 = time.perf_counter()
        data = self.api.post(f"/v1/text-to-speech/{self.voice_id}", raw=True, params={"output_format": "mp3_44100_64"},
                             json={"text": text, "model_id": TTS_MODEL, "language_code": LANGUAGE})
        self.seconds.append(time.perf_counter() - t0)
        return data

    def say(self, text: str):
        self.q.put(text)

    def say_now(self, text: str):
        self._play(self.audio(text))

    def _speaker(self):
        while True:
            text = self.q.get()
            try:
                self._play(self.audio(text))
            except Exception as e:                       # no key, offline: it only prints
                print(f"  (voz: {e})")

    def _play(self, data: bytes):
        if not self.play:
            return
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            f.write(data)
            f.flush()
            subprocess.run(["afplay", f.name], check=False)


class Ears:
    """Push to talk: `start()`, `stop()` -> the recording; `transcribe(path)` -> the text (Scribe)."""

    def __init__(self, api: Api | None = None, device: str | None = None):
        # ffmpeg's avfoundation input (":0" the built-in microphone; COGU_MIC=":2" picks another, see
        # ffmpeg -f avfoundation -list_devices true -i "")
        self.api, self.device = api or Api(), device or os.environ.get("COGU_MIC", ":0")
        self.proc, self.path = None, None
        self.seconds: list[float] = []

    @property
    def recording(self) -> bool:
        return self.proc is not None

    def start(self):
        self.path = Path(tempfile.mkstemp(suffix=".wav")[1])
        self.proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "avfoundation",
                                      "-i", self.device, "-ac", "1", "-ar", "16000", "-y", str(self.path)],
                                     stdin=subprocess.PIPE)

    def stop(self) -> Path:
        proc, self.proc = self.proc, None
        try:
            proc.communicate(b"q", timeout=5)             # (ffmpeg stops and closes the file on "q")
        except subprocess.TimeoutExpired:
            proc.kill()
        return self.path

    def transcribe(self, path: Path) -> str:
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            r = self.api.post("/v1/speech-to-text", data={"model_id": STT_MODEL, "language_code": LANGUAGE},
                              files={"file": ("fala.wav", f, "audio/wav")}, timeout=60)
        self.seconds.append(time.perf_counter() - t0)
        return (r.get("text") or "").strip()

    def heard(self, then):
        """Stop and, in a thread, transcribe and pass the text on (`then(text)`)."""
        path = self.stop()

        def run():
            try:
                text = self.transcribe(path)
            except Exception as e:
                print(f"  (ouvido: {e})")
                return
            finally:
                path.unlink(missing_ok=True)
            print(f"  você: {text or '(nada)'}")
            if text:
                then(text)
        threading.Thread(target=run, daemon=True).start()


# -- a fake ElevenLabs, to test the client without a key ---------------------------------------------------------
def fake_server() -> tuple[str, list]:
    """REST and WebSocket servers on localhost that answer like ElevenLabs, with a scripted "model" that calls the
    client tools for the dialogue of conversation.py. Returns (base URL, every message the client sent)."""
    import http.server

    from memory import plain
    from websockets.sync.server import serve

    seen: list = []

    def agent(ws):
        ws.send(json.dumps({"type": "conversation_initiation_metadata",
                            "conversation_initiation_metadata_event": {"conversation_id": "conv_fake"}}))
        ws.send(json.dumps({"type": "ping", "ping_event": {"event_id": 1, "ping_ms": 5}}))
        n = [0]

        def say(text):
            ws.send(json.dumps({"type": "agent_response", "agent_response_event": {"agent_response": text}}))

        def tool(name, params):
            n[0] += 1
            ws.send(json.dumps({"type": "client_tool_call", "client_tool_call": {
                "tool_name": name, "tool_call_id": f"call_{n[0]}", "parameters": params, "expects_response": True}}))
            for raw in ws:
                m = json.loads(raw)
                seen.append(m)
                if m.get("type") == "client_tool_result" and m["tool_call_id"] == f"call_{n[0]}":
                    return m["result"]

        for raw in ws:
            m = json.loads(raw)
            seen.append(m)
            if m.get("type") != "user_message":
                continue
            text = plain(m["text"])
            if "sobe" in text:
                say("Não consigo subir no sofá, sou pequenininho!")
            elif "bola" in text and "achou" not in text:
                say("Vou lá!")                           # an answer, then the tool, then (slower than QUIET) another
                tool("ir_ate", {"alvo": "bola"})
                time.sleep(1.5)
                say("Tô indo até a bola.")
            elif "sofa" in text:
                tool("ir_ate", {"alvo": "sofá"})
                tool("vir", {})
                say("Vou até o sofá e já volto!")
            elif "vem comigo" in text:
                tool("seguir", {})
                say("Tô indo com você!")
            elif "descansa" in text:
                tool("sentar", {})
                say("Vou sentar um pouquinho.")
            elif "cozinha" in text:
                say("Não sei onde fica a cozinha: me leva lá e diz 'aqui é a cozinha'.")
            else:
                say("Oi! Tô aqui, tudo ótimo.")

    ws_server = serve(agent, "localhost", 0)
    ws_port = ws_server.socket.getsockname()[1]
    threading.Thread(target=ws_server.serve_forever, daemon=True).start()

    silence = io.BytesIO()
    with wave.open(silence, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000), w.writeframes(b"\0\0" * 3200)

    class Rest(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, kind="application/json"):
            data = json.dumps(body).encode() if kind == "application/json" else body
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if not self.headers.get("xi-api-key"):
                raise RuntimeError("no xi-api-key header")
            return data

        def do_GET(self):
            self._body()
            if self.path.startswith("/v1/convai/conversation/get-signed-url"):
                self._send({"signed_url": f"ws://localhost:{ws_port}/v1/convai/conversation?agent_id=agent_fake"})
            elif self.path.startswith("/v1/voices"):
                self._send({"voices": [{"voice_id": "voice_fake", "name": "Fake", "labels": {"language": "pt"}}]})

        def do_POST(self):
            data = self._body()
            if self.path == "/v1/convai/tools":
                body = json.loads(data)
                seen.append({"type": "_create_tool", **body})
                self._send({"id": "tool_" + body["tool_config"]["name"]})
            elif self.path == "/v1/convai/agents/create":
                seen.append({"type": "_create_agent", **json.loads(data)})
                self._send({"agent_id": "agent_fake"})
            elif self.path.startswith("/v1/text-to-speech/"):
                seen.append({"type": "_tts", **json.loads(data)})
                self._send(silence.getvalue(), "audio/mpeg")
            elif self.path == "/v1/speech-to-text":
                seen.append({"type": "_stt", "bytes": len(data)})
                self._send({"text": "Pare!"})

        do_PATCH = do_POST

    rest = http.server.ThreadingHTTPServer(("localhost", 0), Rest)
    threading.Thread(target=rest.serve_forever, daemon=True).start()
    return f"http://localhost:{rest.server_address[1]}", seen


def selftest() -> int:
    import conversation as conv
    base, seen = fake_server()
    api = Api(base, api_key="fake-key-for-the-fake-server")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = setup(api, path=Path(tmp) / "agent.json")
    made = [m for m in seen if m["type"] == "_create_agent"][0]
    assert made["conversation_config"]["agent"]["prompt"]["tool_ids"] == [f"tool_{n}" for n in conv.TOOLS]
    assert made["conversation_config"]["conversation"]["text_only"]
    voice, ears = Voice(api, play=False), Ears(api)
    voice.say_now("Oi, eu sou o Cogu!")
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        with wave.open(f.name, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000), w.writeframes(b"\0\0" * 1600)
        heard = ears.transcribe(Path(f.name))
    assert heard == "Pare!" and conv.local_command(heard) == ("parar", None), heard
    print(f"  setup, voice ({voice.seconds[0] * 1000:.0f} ms), ears ({ears.seconds[0] * 1000:.0f} ms): ok")
    agent = Agent(api, cfg)
    code = conv.run_dialogue(agent, "fake ElevenLabs agent (scripted)", perception="phone")
    sent = [m["type"] for m in seen if m["type"] in ("contextual_update", "user_message", "pong")]
    asked = sent.count("user_message")
    pairs = sum(a == "contextual_update" and b == "user_message" for a, b in zip(sent, sent[1:]))
    results = sum(m["type"] == "client_tool_result" for m in seen)
    print(f"  agent turns {asked}, each after a contextual update: {pairs == asked}; tool results sent {results}; "
          f"pongs {sent.count('pong')}")
    assert pairs == asked == 7 and results == 5 and sent.count("pong") >= 1
    return code


# -- the command line --------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", action="store_true", help="create or update the agent and its tools")
    ap.add_argument("--llm", default=LLM, help="the agent's language model (an id from the agents platform)")
    ap.add_argument("--voice", default=None, help="a voice id (from --voices)")
    ap.add_argument("--voices", action="store_true", help="list the voices of the account")
    ap.add_argument("--say", metavar="TEXT", help="speak a phrase")
    ap.add_argument("--listen", action="store_true", help="record until Enter and print what Scribe understood")
    ap.add_argument("--selftest", action="store_true", help="test the client against a fake local server (no key)")
    args = ap.parse_args(argv)
    try:
        if args.selftest:
            return selftest()
        api = Api()
        if args.voices:
            for v in api.get("/v1/voices").get("voices", []):
                labels = ", ".join(f"{k} {val}" for k, val in (v.get("labels") or {}).items())
                print(f"  {v['voice_id']}  {v['name']}  ({labels})")
        if args.setup:
            setup(api, args.llm, args.voice)
        if args.say:
            v = Voice(api, args.voice)
            v.say_now(args.say)
            print(f"  {v.seconds[0]:.2f} s to the audio")
        if args.listen:
            ears = Ears(api)
            ears.start()
            input("  falando... (Enter para parar) ")
            path = ears.stop()
            print(f"  você: {ears.transcribe(path)}  ({ears.seconds[0]:.2f} s)")
            path.unlink(missing_ok=True)
    except NoKey as e:
        print(f"  {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
