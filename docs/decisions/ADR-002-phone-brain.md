# ADR-002 — Celular como cérebro (substitui a forma de cogumelo)

Data: 2026-10-30. Status: **aceita a direção**; os detalhes abaixo marcados como revisáveis serão
decididos por estudo na simulação. Substitui o ADR-001 (corpo, colar e chapéu) e as seções 4–6 do
brief (ADR-000) no que se refere a corpo, cabeça e eletrônica.

## Problema

O robô completo do ADR-001 ficou feio, e o motivo é estrutural. A pilha do quadril tem 81 mm com
os STS3215, que são grandes para um robô de 28 cm. Para escondê-la, o corpo virou um "balde" de
Ø110, a tela CYD virou uma viseira aparafusada na frente e o chapéu ficou longe das pernas. A forma
de cogumelo pede um corpo redondo e baixo; a mecânica empurra no sentido contrário. O resultado tem
1,32 kg (alvo 1,2), 325 mm (alvo 280) e tornozelos encostando no limite de torque na simulação.

Referência que motivou a mudança: bípedes de mesa recentes que usam um **celular como cérebro**,
preso em pé sobre uma pelve simples, com as pernas à mostra.

## Decisão

1. **O celular é o cérebro.** Ele faz visão, comportamento, conectividade e, no futuro, voz. Já
   traz processador, câmeras, IMU, microfone, alto-falante, Wi-Fi/BLE e a própria bateria.
2. **Montado em pé sobre a pelve, com as câmeras traseiras olhando para a frente.** A tela fica
   virada para trás e **não é rosto**. Visão vale mais que expressão: a câmera traseira é muito
   melhor que a frontal. A tela pode mostrar status para quem está atrás do robô.
3. **Saem do projeto:** corpo, colar, chapéu, os 2 servos do pescoço, a CYD, o Raspberry Pi Zero
   2W, a câmera Pi, o ToF VL53L5CX, o anel de LEDs e a BNO055. Ficam **10 servos**, todos nas pernas.
4. **Locomoção numa "medula" própria, não no celular.** O loop de 50 Hz (observação → política →
   alvos dos servos) roda num microcontrolador na pelve (proposta: **ESP32-S3**) com uma IMU
   própria, ligado à placa de servos. O celular manda só comandos de alto nível (`vx, vy, ω_z`,
   modos) por BLE ou Wi-Fi. Motivos:
   - tempo determinístico: jitter do sistema operacional ou uma queda do BLE não derrubam o robô;
   - funciona igual com iPhone e Android (o iOS não aceita adaptador serial USB comum);
   - a política atual é uma MLP pequena (43 → 128 × 3 → 10, ~40 mil multiplicações por passo), que
     deve caber no ESP32-S3 a 50 Hz — **a confirmar** com um benchmark.
5. **As pernas atuais ficam, e são a identidade do robô** (5 DOF, digitígradas, joelho para trás
   como de ave, coxa e canela de 60 mm). Os exemplos de referência usam joelho humano; a perna
   digitígrada dá ao projeto um ar "alienígena" que é nosso. Só proporções finas (comprimento dos
   links, afastamento do quadril) podem ser ajustadas, e apenas se a simulação mostrar problema de
   torque ou estabilidade com o celular alto, sempre mantendo a forma digitígrada.
6. **Suporte do celular impresso e ajustável**, para um envelope de ~145–165 × 70–78 × 7,5–9,5 mm
   e 165–230 g (sem capinha), com janela livre para o bloco de câmeras. Na simulação o celular é
   uma caixa com massa no topo, e a randomização de domínio cobre massa e altura dentro desse
   envelope.
7. **Proteção em queda é requisito**, não detalhe: o celular é a peça mais cara do robô. O suporte
   precisa de para-choques, e recuperar de queda entra no roteiro de habilidades.

## Consequências

- **Reaproveitado:** pernas, juntas medidas por colisão exata (`cad/check_envelopes.py`), regras de
  design, export do MJCF a partir do CAD, ambiente de treino, PPO, randomização, rollout e
  `compare.py`. As políticas v0/v1 continuam no repositório como referência.
- **Arquivado:** o corpo, o colar, o chapéu e a moldura do ADR-001 saem do modelo, mas ficam no
  histórico do git (último commit com o robô completo: `4432d4c`).
- **Massa estimada ≈ 1,1 kg:** pelve + pernas + 10 servos ≈ 750 g, celular ≈ 200 g, bateria 2S
  ≈ 80 g, ESP32 + IMU + placa de servos ≈ 30 g, suporte ≈ 40 g. Menos peso no alto que o corpo do
  ADR-001 (690 g), o que deve aliviar o torque nos tornozelos.
- **Altura estimada ≈ 390 mm** com as pernas atuais e um celular de 160 mm em pé: mais alta e com o
  centro de massa mais alto que o cogumelo. A primeira coisa a medir na simulação é se a política
  segura esse topo pesado com os torques dos STS3215.
- **BOM:** sai a lista de eletrônica do corpo; entram ESP32-S3, IMU de 6 eixos e suporte do celular.
  Servos, placa de servos, rolamentos, insertos, parafusos e O-rings continuam.
- **Nome:** "Cogu" vem de cogumelo. O nome volta a ser decisão em aberto.

## Decisões revisáveis

1. **ESP32-S3 como medula** em vez de rodar a política no celular. Alternativa: política no celular
   e ESP32 só como ponte serial, aceitando o jitter (a simulação já randomiza 0–40 ms de atraso).
2. **IMU na pelve:** um sensor de 6 eixos comum no mercado brasileiro (ex.: ICM-42688-P, BMI270 ou
   MPU-6050), com filtro de orientação no ESP32. A BNO055 fazia a fusão a bordo, mas não foi
   encontrada no Mercado Livre.
3. **Envelope do celular** (145–165 mm de altura, 165–230 g). Celulares maiores pedem outro suporte.
4. **Tela sem rosto.** Se um dia quiser expressão, a tela pode ser virada para a frente num suporte
   giratório, abrindo mão da câmera traseira.

## Atualização 2026-10-30

**Fase só de simulação.** Nada é impresso nem comprado até o robô andar bem no MuJoCo. Referências
de trabalho (papers, Open Duck Playground, BAM, imitação): `docs/references/mujoco_biped_walking_refs.md`.

**iPhone e Android.** O projeto atende os dois. O que é aberto e documentado é o protocolo entre o
celular e a medula (BLE ou Wi-Fi), não um aplicativo de uma plataforma. O primeiro protótipo usa o
iPhone do autor; o suporte e a simulação cobrem o envelope de celulares acima. A medula ESP32 é o
que torna isso possível: o iPhone não precisa de cabo nem de adaptador serial.

**CAD feito** (`cad/parts/phone_mount.py`), substituindo corpo, colar, chapéu e moldura:

- **Tampa do quadril** (`hip_cover`): caixa impressa aberta embaixo, sobre os servos de yaw, presa
  aos 4 bosses da pelve por tubos de parafuso (M3 × 8). Baia frontal com gaiola para a bateria 2S;
  baia traseira para a placa da medula; furo da chave liga/desliga; 2 insertos M3 para o berço.
  Impressa com o tampo na mesa, sem suporte. 69 g impressos.
- **Berço do celular** (`phone_cradle`): base + dois trilhos em C que seguram as bordas longas do
  celular, que entra por cima; o terço superior fica livre para o bloco de câmeras. Gerado a partir
  de `parameters.PHONE` (padrão: iPhone 15/16 sem capinha, 147,6 × 71,6 × 8,3 mm, 190 g):
  reimprime-se para outro celular. Trilhos são extrusões retas, sem suporte. 19 g impressos.
- **Simulação:** celular, bateria e medula entram como massas no corpo da pelve; o celular colide
  com o chão no modelo completo; câmera `phone_cam` na posição da câmera traseira (dá para ver o
  que o robô "enxerga"). A randomização cobre massa (−40…+55 g) e centro de massa (±4–6 mm) da carga.

Resultado: **1,13 kg, 399 mm**, em pé no MuJoCo (2,6° de inclinação após 3 s), centro de massa
8 mm atrás do eixo do quadril, 173 mm acima do chão.

**Giro de 360° do celular: adiado para a v2 do robô** (decisão do autor em 2026-10-30: celular fixo
é mais simples e mais barato). Um 11º servo sob o berço, com o eixo passando pelo centro do
celular, deixaria o robô virar a tela (rosto, câmera frontal) ou as câmeras traseiras para onde anda,
sem mudar o centro de massa. A ligação sem fio evita cabo torcendo. Custo: +55 g, ~+35 mm de altura.

## Atualização 2026-10-30 (2) — proteção do celular

**Escolhida a bolha de costelas de 150 mm** (`parameters.PHONE_GUARD = "bubble"`, `GUARD_BUBBLE`):
seis costelas, um aro no equador, um anel na base e um topo, formando um ovo de 150 × 170 mm em volta do
celular, com um vão na frente das câmeras traseiras. 56 g em PETG, ~15 peças. O robô fica com **1,20 kg e
415 mm**, centro de massa a 182 mm do chão.

**A tampa do quadril virou um tambor redondo** (`COVER_SHAPE = "round"`) com o mesmo diâmetro da base da
bolha (107 mm), para a bolha assentar rente. Bateria, placa da medula, servos de yaw e tubos dos parafusos
cabem (folga mínima 1,4 mm, na bateria). **O furo da chave liga/desliga saiu**: no tambor não há posição
lateral livre dos tubos dos parafusos; a posição da chave fica para a fase de hardware.

Como foi escolhida (`sim/guard_study.py` e a página interativa de comparação):

| Versão | Direções de queda protegidas | Teste de 8 quedas | Massa extra |
|---|---|---|---|
| Sem proteção (tampa quadrada) | 57 % | celular bateu em 3/8, todas para trás | — |
| Moldura | 100 % | 0/8 | 32 g |
| Bolha de 120 mm | 92 % | 0/8 | 47 g |
| **Bolha de 150 mm** | **99,9 %** (folga mínima 1,8 mm, de cima) | 0/8 | 56 g |

"Direções protegidas": em 800 direções, outra parte do robô chega ao chão pelo menos 2 mm antes do
celular. A bolha de 120 mm deixava as quinas de cima do celular para fora em quedas de cabeça.

Revisável: se a folga de 1,8 mm por cima incomodar, +4 mm de altura na bolha a leva para > 2 mm.

**Exceção às regras de material (só para a bolha), decidida em 2026-10-30.** A regra do projeto é PLA/PETG
(ADR-000 §3). Para a bolha, que leva pancada repetida e serve de apoio para o robô levantar:

- **Costelas, aro, anel e topo em PCTG prata**: copoliéster da mesma família do PETG, bem mais tenaz ao
  impacto, que imprime na A1 Mini com ajustes de PETG. Densidade 1,23 g/cm³.
- **Capas de TPU 95A preto** ("pneus" encaixados por pressão, trocáveis): perfil em C no aro do equador,
  capa com aba no anel da base e um chapéu no topo, 2 mm de espessura. Amortecem e agarram o chão.

Registro em `parameters.MATERIAL_EXCEPTIONS`; as regras de design aceitam esses materiais só nessas peças.
Na simulação completa, PCTG tem atrito 0,4 e TPU 1,0 contra o chão (prioridade de contato 2), para a futura
política de levantar aprender a se apoiar nas capas. Com as capas: 55 g de PCTG + 36 g de TPU, robô com
**1,24 kg e 417 mm**, centro de massa a 186 mm, e **100 %** das direções de queda protegidas (pior folga 3,8 mm).

## Atualização 2026-10-30 (3) — eletrônica comprada

**A medula é a Waveshare Servo Driver with ESP32** (no lugar da ESP32-S3 + interface separada + regulador):
ESP32 com Wi-Fi/Bluetooth/ESP-NOW, interface do barramento ST/SC com 2 conectores (uma cadeia de servos por
perna), telinha OLED, entrada DC 6–12 V por plugue 5,5 × 2,1 mm (a tensão de entrada tem de ser a dos servos:
LiPo 2S), furos Ø2,75 a 23 × 58 mm (placa ≈ 30 × 65 mm). Ela não cabia no tambor, então fica **em pé atrás do
celular, presa em dois postes do berço** com parafusos M2 de sobra dos servos, com a OLED virada para trás e
visível entre as costelas traseiras (folga mínima de 3,5 mm até a bolha). Os cabos dos servos descem por um
furo no berço e no tampo do tambor. **A IMU é uma placa pequena de 6 eixos em I²C** dentro do tambor, atrás.
Robô: 1,24 kg, 417 mm; a política v4 anda igual (0,16 m/s, tremor 0,11, pé a 26 mm, 44/44 % no ar).

A conferir no hardware: se a placa expõe pinos de I²C livres para a IMU e a corrente máxima do plugue DC e
dos conectores de servo. Bateria: em aberto (a Turnigy 2200 mAh 2S de 75 × 33 × 19 mm não cabe no tambor).

## Atualização 2026-10-30 (4) — bateria e alimentação compradas

**Bateria: LiPo Leão 2S 7,4 V 1500 mAh, 70 × 30 × 17 mm, 75 g**, plugue Deans T e balanceamento JST-XH. Cabe em
pé na baia da frente do tambor (canto da frente a 3,9 mm da parede interna). A gaiola passou a ter **0,5 mm de
folga por lado** (`BATTERY_CAGE_CLEARANCE`; antes 0,3 mm, apertado para uma bateria real e uma peça impressa) e a
parede de trás continua no mesmo lugar, a 0,59 mm dos servos de yaw. Robô: 1,25 kg; a política v4 anda igual
(0,16 m/s, 1,71 Hz, tremor 0,11, pé a 26 mm, 44/43 % no ar, sem quedas).

Caminho da energia: bateria → Deans T macho → chave KCD1 (10 A) → XT30 → plugue DC 5,5 × 2,1 mm → placa
Waveshare. O XT30 é o padrão dos cabos internos; o Deans fica só na bateria, sem cortar o conector dela.
Comprados também: carregador ImaxRC B3 Pro (2S/3S, balanceamento), fios de silicone 22 AWG (6 cores), IMU
BMI270 e os 10 servos STS3215. Em aberto: fusível e a posição da chave no CAD.

## Atualização 2026-10-30 (5) — peças de montagem compradas

Comprados também: 10 rolamentos 688ZZ, insertos M3 × 5,7 × 4,6 e parafusos M3 × 8 Allen, O-rings NBR 25 × 3,
chaves táteis 6 × 6 × 5 mm dos pés, porta-fusível mini lâmina com fusíveis de 10 A, fita de velcro de 10 mm para a
gaiola da bateria, bolsa antichamas e alarme de tensão para a LiPo, e 1 kg de PETG cinza-claro (fora da paleta,
que continua branca). Faltam o PETG branco e o coral das peças finais e os 2 servos de reserva. A parte de baixo vai ser impressa primeiro em PLA Lite
laranja, para conferir encaixes; a peça final continua em PETG.

## Atualização 2026-10-30 (6) — bolha pronta para imprimir

A bolha agora é um sólido único, fundido. Antes eram peças sobrepostas, e as sobreposições eram contadas duas
vezes na massa: a bolha caiu de 55 para **50 g**, e o robô ficou com 1,24 kg. O volume é conferido contra um
Monte Carlo da forma analítica, porque as uniões do OCC já perderam material duas vezes. Para imprimir, ela é
cortada **no aro do meio em duas metades**: a de baixo tem 70 mm e sai sem suporte; a de cima tem 100 mm e só
precisa de suporte sob a coroa. Cada uma cabe no limite de 160 mm. A capa de TPU do aro, com abas sobre a borda
de cima e a de baixo, prende as metades. As capas de TPU também saem como STLs separados. A política v4 anda
igual.

## Atualização 2026-10-30 (7) — bolha travada no tambor por baioneta

A bolha sai toda vez que o celular entra ou sai, então a trava é **sem ferramenta**: 3 ressaltos na parede
do tambor, logo abaixo da borda, e uma saia curta na bolha, por fora do tambor, com 3 fendas em L. Desce,
gira 15° no sentido horário (visto de cima), e um calombo de detent na borda de baixo de cada fenda dá o
clique. O calombo fica numa lingueta flexível de ~14 mm, e ele impede que a bolha desgire numa queda. Um
colar sólido liga a saia ao anel da base. Parâmetros em `GUARD_LOCK`. Conferido no CAD: travada, nada se
sobrepõe e ela não sobe nem 1 mm; na posição de entrada, sai livre; o detent interfere 0,3 mm no meio do
giro. A metade de baixo passou a ser impressa de cabeça para baixo (equador na mesa), ainda sem suporte.

As capas de TPU estavam atravessando as costelas (as abas ficam dentro do contorno da bolha): agora têm
entalhes onde cada costela passa, e a capa da base começa acima do colar. Bolha 59 g, TPU 33 g, robô
1,25 kg; a política v4 anda igual.

## Atualização 2026-10-30 (8) — iPhone 15 Pro Max, bolha mais alta e abertura da frente para a câmera e o LiDAR

O primeiro protótipo usa o **iPhone 15 Pro Max** (159,9 × 76,7 × 8,25 mm, 221 g), que tem LiDAR. Antes o CAD
tinha um celular de 147,6 mm: o novo atravessava a casca da bolha. A bolha subiu de 100 para **108 mm** de
semieixo vertical (centro em z 115,6, topo em 223,6), mantendo o raio de 53,6 mm onde encosta no tambor: folga
mínima de 3,8 mm entre o módulo da câmera e a casca. A abertura da frente passou de 60° para **90°** (costelas em
±45°, 105°, 165°, 195° e 255°): com as costelas em ±30°, a da frente à esquerda tampava ~13 % da vista da câmera
traseira, e o LiDAR fica ao lado dela. Agora a câmera principal (74° × 59°, celular em pé) vê 0 % do robô. Os
ressaltos da trava passaram a ter espaçamento desigual (0°, 110°, 240°): a bolha só encaixa na posição certa,
e as outras cinco orientações batem na saia. Robô: 1,28 kg, 431 mm; o celular não tocou o chão em nenhuma de
16 quedas.

Navegação (próximo passo depois da marcha): o ARKit dá a posição do robô no ambiente e o mapa de profundidade
do LiDAR. Plano em duas camadas: a política de marcha continua só com servos e IMU, a 50 Hz; o celular decide
para onde ir, a 5–10 Hz, e manda os comandos de velocidade. No simulador, a câmera `phone_cam` do MJCF renderiza
imagem e profundidade (para fazer o papel do LiDAR) e a posição do ARKit é a verdadeira com erro e atraso.

## Próximos passos

1. CAD da pelve nova com o suporte do celular; MJCF com o celular como caixa randomizada.
2. Retreinar a política no modelo novo e comparar com a v1 (marcha, quedas, torque nos tornozelos).
3. Trabalhar a marcha natural (medição da marcha, recompensa, depois marcha de referência e imitação).
4. Benchmark da política no ESP32-S3 e esboço do firmware da medula.
