# ADR-003 — Quadril até −120° para levantar sem impulso

Data: 2026-11-02. Status: **aceita** (CAD atualizado; falta imprimir e testar o roll link novo).

## Problema

Deitado de costas, com o quadril indo só até −85°, as solas não chegam perto do centro de massa, e o
robô só levantava com um impulso, que nunca ficou confiável: o melhor resultado foi 53 de 100 partindo
sobre os pés e 0 de quedas. No simulador (`sim/playground/hip_study.py`, README "Estudo do quadril"),
com o quadril chegando a −120° o robô sobe devagar de costas, sem impulso. Quedas de bruços ou de lado
rolam primeiro até ficar de costas. A política de levantar v8 levanta de 99–100 de 100 quedas, no
modelo de treino e no modelo de contatos completos.

No CAD, a coxa batia a partir de −95° no braço traseiro do U do roll. Esse braço segura o 688ZZ do
roll e descia até a viga do servo do quadril, 12–19 mm atrás do eixo do quadril. Abaixar o eixo
25 mm libera só até −120°, sem folga, e deixa o robô 25 mm mais alto.

## Decisão

O braço traseiro do U (`hip_roll_link`) agora só envolve o rolamento. Ele se liga ao braço da frente
por uma **ponte lateral por fora** (5 × 16 mm, na altura do eixo do roll), em vez de descer até a
viga. A ponte fica:

- 2 mm fora do giro do servo do roll, a mesma folga da viga;
- 7 mm por fora da placa do rolamento da coxa;
- acima de onde a coxa passa.

Mais nada muda:

- **Eixos:** todos no mesmo lugar, então a cinemática e a altura ficam iguais.
- **Peças compradas:** os mesmos servos e os 688ZZ.
- **Massa:** +2,5 g por roll link (robô ≈ 1325 g).
- **Largura:** a ponte fica dentro da largura dos pés.

Faixa do quadril: `hip_pitch` de −120 a +15° (`parameters.JOINT_RANGES_DEG`). Medido com sólidos
exatos (`cad/check_envelopes.py`, que agora varre até −150°):

- **Contra o roll link:** livre até −135°.
- **Perna esticada:** livre até −125°; depois disso o pé encosta no tambor.
- **Joelho dobrado:** livre até −135°; depois disso a coxa encosta no yaw link e no roll link.

A ponte não toca a pelve, o tambor nem a bolha com o roll de −60° a +60°, e o yaw nos limites.

## Consequências

- Imprimir 2 roll links novos (`build/stl/hip_roll_link_L.stl`, `_R.stl`).
- A marcha v10 segue funcionando no modelo novo: anda e aguenta 38 de 40 empurrões.
- O levantar v8 também: 30/30 em cada tipo de partida e 99–100 de 100 quedas.
- **A rever com a peça na mão:**
  - a rigidez da ponte, que segura o rolamento traseiro do roll em balanço;
  - a orientação de impressão;
  - se o conector do barramento do servo do roll ainda cabe no espaço ao lado do braço traseiro.
