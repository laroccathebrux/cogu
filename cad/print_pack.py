"""The print pack: the STLs of the robot (already in print orientation), named and grouped by build plate for the
A1 Mini, with a LEIA-ME in Portuguese. Run after cad/export.py (it copies from build/stl).

    uv run python -m cad.export --no-step && uv run python -m cad.print_pack     # -> build/print/cogu_v1/
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

from cad import assembly
from cad.test_prints import build_test_prints

ROOT = Path(__file__).resolve().parents[1]

# (folder, title, [(stl in build/stl, copies, description, how it lies, supports)])
PLATES = [
    ("0_testes", "Testes de encaixe (antes de tudo)", [
        ("test_prints/fit_test.stl", 1, "pinos e furos com 0,1 / 0,2 / 0,3 / 0,4 mm de folga (entalhes = décimos)",
         "plana", "não"),
        ("test_prints/bearing_test.stl", 1, "3 assentos do 688ZZ (16,05 / 16,10 / 16,15 mm) e o pino do back hub",
         "plana", "não"),
        ("test_prints/insert_test.stl", 1, "4 furos de inserto M3, para acertar a temperatura do ferro", "plana",
         "não"),
        ("test_prints/servo_fit_test.stl", 1, "o alojamento do servo das pernas, com a janela dos conectores",
         "em pé, como a coxa", "não"),
    ]),
    ("1_coxas_canelas", "Coxas e canelas", [
        ("thigh_L.stl", 1, "coxa esquerda", "em pé, fundo do alojamento na mesa", "árvore, só sob as bochechas"),
        ("thigh_R.stl", 1, "coxa direita", "idem", "idem"),
        ("shin_L.stl", 1, "canela esquerda", "idem", "idem"),
        ("shin_R.stl", 1, "canela direita", "idem", "idem"),
    ]),
    ("2_quadril", "Elos do quadril", [
        ("hip_roll_link_L.stl", 1, "elo do roll esquerdo (aloja o pitch)", "viga deitada, braço da frente para cima",
         "árvore sob a ponte de fora, o paddle de trás e a raiz do braço"),
        ("hip_roll_link_R.stl", 1, "elo do roll direito", "idem", "idem"),
        ("hip_yaw_link_L.stl", 1, "elo do yaw esquerdo (aloja o roll)", "de cabeça para baixo, disco na mesa",
         "sob o anel em volta do disco"),
        ("hip_yaw_link_R.stl", 1, "elo do yaw direito", "idem", "idem"),
    ]),
    ("3_pes_pelve_hubs", "Pés, pelve e back hubs", [
        ("foot_L.stl", 1, "pé esquerdo", "sola na mesa", "não; brim de 5 mm (os sulcos dos O-rings ficam embaixo)"),
        ("foot_R.stl", 1, "pé direito", "idem", "idem"),
        ("pelvis_R.stl", 1, "pelve (serve para os dois lados)", "face plana na mesa, ressaltos para cima", "não"),
        ("back_hub_R.stl", 6, "back hub: pitch, joelho e tornozelo, 3 por perna", "disco na mesa, pino para cima",
         "não; 100 % de preenchimento"),
        ("back_hub_roll_R.stl", 2, "back hub do roll (placa de 10 mm), 1 por perna", "idem", "idem"),
    ]),
    ("4_tambor_berco", "Tambor e berço do celular", [
        ("hip_cover_R.stl", 1, "tambor: gaiola da bateria, placa da IMU, janelas dos yaw", "tampo na mesa",
         "não (o piso da gaiola faz ponte de 13 mm)"),
        ("phone_cradle_R.stl", 1, "berço do iPhone 15 Pro Max, com os postes da placa Waveshare",
         "base na mesa, trilhos em pé", "não"),
    ]),
    ("5_bolha", "Bolha de proteção do celular (depois)", [
        ("phone_guard_bubble_lower.stl", 1, "bolha, metade de baixo (PCTG)", "de cabeça para baixo, corte na mesa",
         "não"),
        ("phone_guard_bubble_upper.stl", 1, "bolha, metade de cima (PCTG)", "corte na mesa",
         "árvore só sob a coroa e o topo das costelas"),
        ("phone_guard_tpu_hoop_tyre.stl", 1, "capa do aro (TPU 95A)", "eixo em pé", "não"),
        ("phone_guard_tpu_base_sleeve.stl", 1, "capa da base (TPU 95A)", "eixo em pé", "não"),
        ("phone_guard_tpu_crown_cap.stl", 1, "capa da coroa (TPU 95A)", "domo para cima", "árvore sob o topo"),
    ]),
]

def commit() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "cad"], capture_output=True,
                               text=True).stdout.strip()
        return sha + (" + mudanças ainda não commitadas em cad/" if dirty else "")
    except OSError:
        return "?"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stl", default=str(ROOT / "build" / "stl"))
    ap.add_argument("--out", default=str(ROOT / "build" / "print" / "cogu_v1"))
    args = ap.parse_args(argv)
    stl, out = Path(args.stl), Path(args.out)
    parts = assembly.build_parts()
    grams = {p.name: p.printed_mass_g for p in parts.values()}
    grams.update({p.name: p.printed_mass_g for p in build_test_prints().values()})
    pieces = {"phone_guard_bubble": 2, "phone_guard_tpu": 3}

    def mass(file: str) -> float:
        name = Path(file).stem
        for suffix in ("_L", "_R"):
            if name.endswith(suffix):
                name = name[:-2]
        for whole, n in pieces.items():
            if name.startswith(whole):
                return grams[whole] / n
        return grams[name]

    if out.exists():
        shutil.rmtree(out)
    rows, total_legs, missing = [], 0.0, []
    for folder, title, items in PLATES:
        d = out / folder
        d.mkdir(parents=True)
        rows.append(f"\n### {folder}: {title}\n")
        rows.append("| Arquivo | Cópias | O que é | Na mesa | Suporte | g cada |")
        rows.append("|---|---|---|---|---|---|")
        for file, copies, what, lies, support in items:
            src = stl / file
            if not src.exists():
                missing.append(file)
                continue
            name = Path(file).stem + (f"_x{copies}" if copies > 1 else "") + ".stl"
            shutil.copy2(src, d / name)
            g = mass(file)
            if folder[0] in "1234":
                total_legs += g * copies
            rows.append(f"| `{name}` | {copies} | {what} | {lies} | {support} | {g:.0f} |")
    if missing:
        print("faltam (rode cad/export.py antes):", ", ".join(missing))
        return 1
    text = README.format(today=date.today().strftime("%d/%m/%Y"), commit=commit(), tables="\n".join(rows),
                         total=total_legs)
    (out / "LEIA-ME.md").write_text(text)
    digest = hashlib.sha1(b"".join(p.read_bytes() for p in sorted(out.rglob("*.stl")))).hexdigest()[:10]
    print(f"{out}  ({sum(1 for _ in out.rglob('*.stl'))} STLs, ~{total_legs:.0f} g de PETG nas mesas 1–4, sha1 {digest})")
    return 0


README = """# Cogu v1: pacote de impressão

Gerado em {today} a partir do CAD do commit {commit}, por `cad/print_pack.py`. Os STLs já estão na orientação de
impressão. Se o CAD mudar: `uv run python -m cad.export --no-step && uv run python -m cad.print_pack`.

## Antes de tudo: a mesa 0, com um servo e um rolamento na mão

1. `fit_test` (feito em 12/11: justo no de 0,1, no limite no de 0,2, solto no de 0,3). O CAD usa agora 0,1 mm nos
   assentos dos rolamentos e 0,3 mm por lado em volta dos servos.
2. `bearing_test`: três assentos, marcados com 1, 2 e 3 entalhes (Ø16,05 / 16,10 / 16,15). O 688ZZ tem que entrar
   **com o polegar e ficar preso** no de 2 entalhes, que é o do CAD. Se só ficar firme no de 1, ou só entrar no de 3,
   avise: o assento muda nas peças. O pino tem que girar solto dentro do rolamento.
3. `insert_test`: acerte a temperatura do ferro até o inserto M3 ficar rente (PETG ≈ 230 °C, PLA ≈ 200 °C).
4. `servo_fit_test`: o STS3215 desliza para dentro sem forçar, o disco de trás assenta no furo, os 4 furos M2
   batem com os do servo e **os dois plugues do barramento entram pela janela**. Meça também quanto o plugue
   encaixado sobe acima da face do servo (o CAD supõe ~8 mm de plugue e ~11 mm com a curva do fio).

Se algo não bater, não imprima as pernas: as medidas vão para `cad/parameters.py` e o pacote sai de novo.

## Mesas
{tables}

Total das mesas 1–4: cerca de **{total:.0f} g**. A bolha (mesa 5) pode esperar: o robô anda sem ela.

## Configuração (bico 0,4)

- Camada 0,2 mm, **4 paredes**, preenchimento giroide 30 %; back hubs com 100 %.
- **PETG** (o cinza-claro que já está em casa serve para o protótipo; o branco fica para a versão final). O PLA Lite
  laranja serve para a mesa 0 e para testar encaixes, mas amolece perto de 55–60 °C e racha numa queda.
- Pés com brim de 5 mm.

## O que mudou desde o pacote de 30/10 (`parte_de_baixo`, não use mais)

- Coxa, canela e a viga do pitch ganharam a **janela sobre os conectores** do servo; a coxa e a canela, um bolso na
  bochecha de trás, aberto para fora da perna, onde o plugue e a curva do fio cabem. Antes os cabos do pitch, do
  joelho e do tornozelo não tinham por onde entrar.
- O elo do roll recuou o paddle de trás (conectores do roll livres) e usa o `back_hub_roll`, mais alto.
- O pé tem 130 × 67 mm e uma canaleta para os fios das chaves: entre as duas chaves, por baixo do garfo e subindo
  pela face de fora do paddle do rolamento até perto do eixo.
- O tambor e o berço têm uma janela sobre os conectores de cada yaw, e o tambor uma placa para a IMU (fita dupla
  face de espuma).
- O quadril vai a −130° (sentar quase no chão).

## Montagem: pontos de atenção

- **Cabos por fora da perna**: de cada servo, o fio sai pelo bolso para o lado de fora e sobe pela face de fora até
  a junta de cima, cruzando perto do eixo (laço curto). Prenda com abraçadeira fina; meça os comprimentos com
  barbante antes de cortar (`hardware/wiring/README.md`).
- **Pitch**: com a perna toda à frente (+10 a +15°), o soquete de baixo fica com só 8,6 mm livres. Use o soquete
  de cima, com um cabo em Y (entrada e saída no mesmo plugue), ou confira na bancada que o plugue cabe.
- **Joelho e tornozelo**: a borda do disco da peça de baixo passa a 8,6 mm do soquete, só no milímetro mais perto do
  eixo: o plugue cabe, o fio deve sair para longe do eixo.
- **Chaves dos pés**: fio de silicone 28–30 AWG (o de 22 AWG não cabe na canaleta de 2,5 × 1,5 mm).
- **Chave liga/desliga**: o lugar no tambor ainda não foi decidido; no protótipo, fica no cabo da bateria.
- **Servos**: antes de montar, dê um ID a cada um e leve-os ao centro (2048). Os IDs do firmware são 1–5 na perna
  esquerda e 11–15 na direita, do yaw ao tornozelo (`firmware/medula/esp32/src/board.h`).
"""


if __name__ == "__main__":
    sys.exit(main())
