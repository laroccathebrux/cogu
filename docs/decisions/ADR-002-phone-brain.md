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

## Próximos passos

1. CAD da pelve nova com o suporte do celular; MJCF com o celular como caixa randomizada.
2. Retreinar a política no modelo novo e comparar com a v1 (marcha, quedas, torque nos tornozelos).
3. Trabalhar a marcha natural (medição da marcha, recompensa, depois marcha de referência e imitação).
4. Benchmark da política no ESP32-S3 e esboço do firmware da medula.
