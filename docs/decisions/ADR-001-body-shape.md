# ADR-001 — Forma do corpo, colar e chapéu

Data: 2026-10-25. Status: aceita (v0, sim primeiro). Complementa o brief (ADR-000 §4).

## Problema

A pilha do quadril (yaw → roll → pitch) mede **81 mm** da face da pelve até o eixo de pitch, e o
quadril compacto de eixos cruzados foi tentado e rejeitado (ADR-000 §4, nota em `JOINT_RANGES_DEG`).
Com a pelve a 205 mm do chão, tudo isso ficava à vista abaixo dela. O brief pede um corpo "quase
encostando nos pés", uma esfera creme Ø95, colar ~18 mm e chapéu Ø160 × 55.

A solução é de forma do corpo: a pelve sobe para dentro da casca e a casca desce o máximo que a
cinemática permite. O limite vem da **varredura exata** do quadril (`cad/check_envelopes.py` /
`cad/check_body.py`, booleanos build123d sobre as peças impressas), medida por fatia de altura
(referência: origem da pelve, eixo vertical do corpo em x = −5):

| Altura (ref. pelve) | O que está lá | Raio necessário |
|---|---|---|
| 0 … +33 mm | corpos dos 2 servos de yaw (65 mm de largura) | 43,4 mm |
| −12 … −35 mm | yaw links e braços do roll link (yaw −7…+30°, roll ±35°) | 46 mm no topo, 54,4 mm perto do aro |
| abaixo de −39 mm | coxa a −85° de pitch, braços do roll link a ±35° de roll | 73–86 mm, crescendo para 150 mm |

Ou seja: **nenhum corpo com menos de ~Ø100 fecha sobre os servos de yaw**, e **nenhum corpo de
revolução pode descer abaixo de z ≈ −38 mm** sem bater na coxa ou no roll link em algum ponto do
range. A segunda restrição é a que define a altura visível da pilha.

Uma segunda restrição de largura é a tela: a CYD 2,8" tem PCB de **86 × 50 mm**, com a área ativa
8,4 mm fora do centro (a ponta do ESP32 sobra 17 mm de um lado). Num Ø95 ela só cabe atravessando o
centro do corpo (onde estão os servos de yaw). Na frente, qualquer posição que deixe o vidro perto
da superfície exige corda ≥ 97 mm a ≥ 28 mm do eixo (os paddles do roll link varrem até x = 27 abaixo
de z = −12), o que pede Ø ≥ 125 — por isso a tela fica numa **viseira plana** da largura do corpo,
e não dentro da esfera.

## Alternativas avaliadas

Renders em `docs/renders/body_proposal_*.png` (mesmas pernas, colar e chapéu).

| | Forma | Esconde | Altura total | Veredito |
|---|---|---|---|---|
| **A** | Esfera Ø100 (a menor que fecha sobre os servos), fundo aberto em z = −16 | só a pelve; yaw links, roll links e vigas de pitch à vista (≈ 85 mm de pilha) | 325 mm | rejeitada: é o brief "ao pé da letra", mas não esconde nada e a CYD vira uma viseira saltando 12 mm de cada lado |
| **B** | Ovo largo Ø110 (Ø115 no aro), saia até z = −37, ombro a 37°, viseira plana 104 × 71 inclinada 8° para cima | yaw links e metade dos braços do roll link; ficam à vista a metade de baixo dos braços (19 mm) e as vigas de pitch (30 mm) | **325 mm** | **escolhida** |
| **C** | Esfera + duas "coxas" laterais até a viga de pitch (z = −66) | tudo até o eixo de pitch | 325 mm | impossível: para conter os braços do roll link a ±35° a saia teria de ter **166 mm** de largura em z = −65 (render do envelope em `body_proposal_C_pods_envelope.png`) |

A altura total é a mesma nas três porque é fixada por cima, não por baixo: topo dos servos de yaw
(+32,5) → servo de pitch do pescoço acima deles (eixo a +49) → colar 18 mm → aba do chapéu a +65 →
domo 55 → **325 mm com a pelve a 205 mm**. O alvo de 280 mm do brief assumia pé de 15 mm e o eixo
de pitch no fundo do corpo; com esta pilha de quadril ele não é atingível sem agachar mais a pose de
repouso (cada 10 mm de pelve são 10 mm de altura) ou encurtar o domo.

## Decisão (B) — o que foi modelado

- **Corpo** (`cad/parts/body.py`): casca de revolução por spline (`BODY_PROFILE`), parede 2 mm,
  6 nervuras verticais, aberta embaixo (aro Ø99 em z = −37, 2,4 mm acima do ponto mais alto da coxa
  a −85°; a saia **alarga para Ø115 no aro**, porque os braços do roll link chegam a r = 54,4 a −35° de roll com yaw +30° — leve forma de pêra, sem balanço para imprimir com o aro na mesa) e em
  cima (furo Ø73 para o pescoço). 4 orelhas parafusadas nos insertos da pelve (M3 × 8
  por cima), 3 insertos sob o ressalto do pescoço (colar), **viseira**: caixa plana 104 × 71 × 20
  inclinada 8° para cima, saliente da esfera nos lados, com a CYD encaixada pelo vidro (PCB a 42 mm
  do eixo, atrás da varredura dos paddles) e 4 insertos para a moldura acima e abaixo da PCB, trilhos
  verticais para o Pi Zero (em pé à frente dos servos de yaw) e para a placa de servos (entre o Pi e
  a tela), baia da bateria
  2S em pé na traseira (cinta por duas fendas), prateleira da BNO055 (posição canônica
  `IMU_POS`, também o site `imu` do MJCF), furo da chave e fenda USB-C na traseira direita.
- **Pelve**: alargada para 85 mm e cantos aparados a r = 50 em torno do eixo do corpo, porque os
  4 bosses originais (a ±32 mm) **colidiam com os servos de yaw** (222 mm³ cada); agora ficam a ±37,7.
- **Colar** (`cad/parts/collar.py`): anel Ø78 × 18 + flange sobre o ressalto + o `servo_housing()`
  do servo de pitch, deitado (eixo Y, corpo para trás, eixo 8 mm à frente do eixo do corpo), com o
  fundo a +33,9 (1,4 mm acima dos discos traseiros dos servos de yaw). O deck tem duas fendas por
  onde os discos dos paddles e o cubo traseiro do suporte giram. 3 × M3 × 8 por cima, à vista.
- **Suporte do chapéu** (`cad/parts/cap.py`, `cap_bracket`): garfo em U sobre o servo de pitch
  (paddle do horn + paddle do 688ZZ, mesmo padrão da coxa, braços estreitados para 16 mm acima do
  disco para passarem dentro do anel do colar) cuja viga é o `servo_housing()` do servo de roll,
  deitado ao longo de Y, eixo X, centrado sobre o eixo de pitch. A viga fica 25 mm acima do eixo de
  pitch — o mínimo para que as pontas da viga, ao inclinar ±20°, passem por cima do housing de pitch
  (topo a +15). O eixo de roll fica em y = +12,5 mm (o housing é assimétrico em relação ao eixo e
  cabe centrado no domo): o chapéu rola em torno de uma linha 12,5 mm fora do centro, o que desloca
  o ápice 3 mm a ±15°.
- **Chapéu** (`cap`): meio-elipsoide Ø160 × 55, parede 1,6 mm, 8 nervuras radiais só na metade
  externa (o suporte oscila sob o centro), anel de borda com canal para a fita de LEDs, fenda de
  sensor de 360° (1,5 mm de rebaixo) vazada nos 70° frontais para câmera e ToF, 3 manchas em rebaixo
  de 0,8 mm para discos creme encaixados, garfo interno (paddle do horn + paddle do 688ZZ) unido por
  uma sela ao domo. Aba a +65 mm (1 mm acima do colar).
- **Moldura** (`cad/parts/bezel.py`): placa 104 × 71 × 3 com a janela da área ativa (58 × 43, chanfro
  45°) e 4 × M3 × 8 à vista acima e abaixo da janela; prende a CYD pela borda do vidro (os furos
  reais da CYD não foram medidos).
- **Juntas do pescoço**: `neck_pitch` (eixo −Y, + = chapéu inclina para a frente) ±15°, `neck_roll`
  (eixo +X) ±12°. O limite é a aba do chapéu encostando nos cantos superiores da viseira a pitch
  −20° com roll ±15°. Ranges verificados por `cad/check_body.py` (margem 3°).

Resultado: robô completo em pé no MuJoCo (`sim/check_mjcf.py`), **altura 325 mm**, massa estimada
**≈ 1,32 kg** (alvo 1,2 kg; casca do corpo 93 g, chapéu 91 g, colar 35 g, suporte 17 g, moldura 10 g,
eletrônica + bateria ≈ 180 g, 12 servos 660 g). Render: `docs/renders/sim_full_robot_v0.png`.

## Checagens em código

- `cad/check_envelopes.py` (quadril, inalterado) e **`cad/check_body.py`** (novo, no CI):
  corpo + moldura + colar contra a perna em toda a caixa yaw × roll × pitch × joelho; corpo contra
  pelve, servos e eletrônica; **chapéu contra coxa/canela/pé ao agachar e sentar** (pitch e joelho nos
  limites, todo o range do pescoço) — a checagem obrigatória do brief; varreduras de pitch e roll do
  pescoço contra colar, corpo e suporte, comparadas a `NECK_*_RANGE_DEG` com margem de 3°.
- `design_rules.py`: 160 mm agora com 0,05 mm de tolerância (caixa envolvente de spline); o chapéu é
  exatamente Ø160.
- Dois defeitos pré-existentes encontrados por essas checagens e corrigidos de passagem: os 4 bosses
  da pelve invadiam os corpos dos servos de yaw (222 mm³ cada), e o `servo_pocket()` do STS3215 não
  incluía os 1,55 mm dos bosses da tampa traseira — a parede traseira de **todos** os housings
  começava 1,55 mm antes do previsto em `cad/stack.py` (o servo não assentaria). O MJCF e os STL das
  pernas mudam por isso (interior dos housings; envelope externo igual).

## Decisões revisáveis

1. **Ø110 em vez de Ø95** (razão chapéu/corpo 1,45 em vez de 1,6–1,9). Única forma de fechar
   sobre os servos de yaw e de alojar a CYD; para voltar ao Ø95–100 é preciso trocar a tela por uma
   menor (ex.: 2,0" ST7789, PCB ~55 × 38) ou aceitar a viseira saliente.
2. **Viseira plana 104 × 71 × 20 mm** saliente da esfera (inclinada 8° para cima): é a única forma
   de pôr 86 mm de PCB à frente de um corpo de Ø110 sem invadir a varredura do quadril. Visualmente
   é uma "tela de TV" no rosto; as alternativas são a tela menor acima (viseira some) ou um corpo
   Ø125 com a tela embutida.
3. **Altura 325 mm** (brief: ~280). Alavancas: pose de repouso mais agachada (−10 mm por 10 mm de
   pelve; exige retreinar a política), domo de 45 mm (−10), aba do chapéu dentro do domo (−8: o
   colar ficaria parcialmente escondido), pé de 15 mm como no brief (−11, mas a canela varre r ≈ 20
   em torno do tornozelo).
4. **Eixo de roll do chapéu a 12,5 mm do centro**. Centrar exige um servo com o eixo no meio do
   corpo (não é o STS3215) ou subir a viga 13 mm (altura +13).
5. **Empacotamento interno** (Pi em pé atrás da tela, placa de servos entre os dois, bateria em pé
   atrás, IMU na prateleira traseira): posições em `parameters.py`, não validadas com cabos; furos da
   CYD e da placa de servos por medir. O volume útil depois dos 3 servos superiores é pequeno; a
   alternativa é a placa de servos dentro do chapéu (barramento passa pelas duas juntas do pescoço).
6. **Massa 1,32 kg > 1,2 kg**: cortar 50 g no chapéu (parede 1,2 mm + mais nervuras) vale 150 g no
   corpo segundo o brief; a bateria exata (§12 do brief) ainda está aberta.
7. **Colisão só com o chão para as cascas no MuJoCo** (hulls convexos). Para treinar recuperação de
   queda, o chapéu e o corpo colidem com o chão; se um dia for preciso casca–perna na sim, decompor a
   casca em hulls convexos.
