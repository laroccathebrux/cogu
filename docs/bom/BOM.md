# BOM — robô completo (v0, fase "sim primeiro")

Quantidades para **duas pernas + corpo + colar + chapéu** (ADR-001). Tudo de prateleira; alternativas por região na última coluna.

| Item | Qtd | Spec | Onde |
|---|---|---|---|
| Servo Feetech STS3215 7,4 V (serial bus, 12 bit) | 12 | 3 quadril + joelho + tornozelo por perna, pitch + roll do chapéu. Vem com horn, disco traseiro e parafusos M2 | AliExpress (Feetech oficial), Waveshare (ST3215), RobotShop, Eckstein (EU), Core Electronics (AU) |
| Placa controladora de servos serial (TTL) | 1 | Waveshare "Bus Servo Adapter" ou Feetech URT-1 | idem |
| Rolamento 688ZZ | 12 | 8 × 16 × 5 mm, blindado (10 pernas + 2 pescoço) | qualquer loja de rolamentos / skate / AliExpress |
| Inserto roscado a quente M3 | 11 | 4 pelve + 3 pescoço (corpo) + 4 moldura da tela (corpo). M3 × 5,7 mm, OD 4,6 mm | AliExpress, Amazon ("M3 heat set insert 5.7") |
| Parafuso M3 × 8 allen cilíndrico | 11 | 4 corpo→pelve, 3 colar→corpo, 4 moldura→corpo. Aço inox | ferragem local |
| O-ring NBR 25 × 3 mm | 4 | 2 por pé, canal impresso | loja de vedação / AliExpress |
| Chave tátil 6 × 6 mm | 4 | 2 por pé, altura 4,3–5 mm | qualquer loja de eletrônica |
| Tela ESP32-2432S028R "CYD" 2,8" | 1 | 86 × 50 mm, ILI9341 320×240, touch resistivo | AliExpress / Amazon |
| Raspberry Pi Zero 2W | 1 | fica em pé atrás da tela, em trilhos impressos | revendedores oficiais |
| IMU BNO055 (breakout estilo Adafruit, 20 × 27 mm, 2 furos a 20 mm) | 1 | na prateleira traseira do corpo (`IMU_POS`) | Adafruit / clones |
| Bateria LiPo 2S 1000 mAh soft pack | 1 | ≤ 14 × 31 × 58 mm, em pé na baia traseira, cinta por 2 fendas | lojas de aeromodelismo |
| Regulador 5 V (buck 3 A), chave liga/desliga Ø12, USB-C painel | 1 cada | furos na traseira direita do corpo | — |
| Câmera Pi + ToF VL53L5CX + anel WS2812 | 1 cada | sob a aba do chapéu (janela frontal da fenda, canal do anel na borda) | — |
| Filamento PETG | ~450 g | creme (corpo, colar, pernas) + coral (chapéu) + 3 discos creme para as manchas | — |

Parafusos de servo: só os **M2 originais** que vêm com cada STS3215 (8 para o corpo, 4 para o horn,
4 para o disco traseiro). Até aqui só M3 × 8 é usado; o BOM completo do robô mantém apenas M3 × 8 / 12 / 16.
Pi Zero, BNO055 e placa de servos ficam em trilhos / pinos impressos (sem M2,5 no BOM); furos exatos da CYD e
da placa de servos ainda não foram medidos — a moldura prende a CYD pela borda do vidro.
