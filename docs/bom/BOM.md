# BOM — robô com celular (ADR-002), 2026-10-30

Fase só de simulação: os itens marcados **estável** não dependem de mudanças no CAD; os marcados **em aberto**
ainda podem mudar com o desenho.

| Item | Qtd | Detalhe | Situação |
|---|---|---|---|
| Servo Feetech STS3215 7,4 V | 10 (+2 de reserva) | 5 por perna; vem em kits de 4 → 3 kits | estável |
| Waveshare Servo Driver with ESP32 | 1 | ESP32 + barramento ST/SC (2 portas, uma por perna) + OLED; entrada 6–12 V, plugue 5,5 × 2,1 mm | **comprado** |
| IMU 6 eixos em placa I²C | 1 | ICM-42688-P ou BMI270 (MPU-6050 funciona, mais ruidosa) | estável |
| Bateria LiPo 2S (7,4 V) | 1 | em pé no tambor: largura ≤ 34 mm e, com 2 mm de folga, comprimento ≤ 78 / 77 / 73 / 71 / 69 mm para espessura 15 / 16 / 18 / 19 / 20 mm; 1000–2200 mAh, ≥ 20C | **em aberto** |
| Carregador de balanceamento 2S | 1 | conector JST-XH | estável |
| Cabo bateria → plugue DC 5,5 × 2,1 mm | 1 | do conector da bateria (T-Plug, XT30…) para a entrada da placa | depende da bateria |
| Chave liga/desliga ≥ 5 A + fusível | 1 + 1 | posição no CAD em aberto | em aberto |
| Fios de silicone 20–22 AWG, conectores | — | os cabos dos servos vêm com eles | estável |
| Chave tátil 6 × 6 mm | 4 | 2 por pé (opcional) | estável |
| O-ring NBR 25 × 3 mm | 4 | 2 por pé | estável |
| Rolamento 688ZZ | 10 | um por articulação das pernas | estável |
| Inserto M3 × 5,7 mm / parafuso M3 × 8 | 6 / 6 | 4 tambor→pelve, 2 berço→tambor | estável |
| Filamento PETG branco | ~350 g | pernas, pelve, tambor, berço | estável |
| Filamento PETG coral | ~30 g | pés e aro do tambor (troca de cor) | estável |
| Filamento PCTG prata | ~100 g | bolha (exceção de material, ADR-002) | **comprado** |
| Filamento TPU 95A preto | ~60 g | capas da bolha (exceção de material, ADR-002) | **comprado** |

Não precisa mais: Raspberry Pi Zero, tela CYD, câmera Pi, ToF VL53L5CX, anel de LEDs, BNO055, ESP32-S3
separada, interface de barramento separada e regulador de 5 V (a placa Waveshare regula a própria ESP32).

---

> O BOM abaixo é o do robô-cogumelo (ADR-001), mantido como histórico.

# BOM — robô completo (v0, fase "sim primeiro")

> **Desatualizado desde o [ADR-002](../decisions/ADR-002-phone-brain.md) (2026-10-30).** O robô agora usa um celular como cérebro: saem CYD, Pi Zero, câmera, ToF, anel de LEDs, BNO055 e os 2 servos do pescoço (ficam 10 servos); entram uma placa ESP32-S3 com IMU de 6 eixos e interface do barramento dos servos, e o berço do celular. O projeto está em fase só de simulação: este BOM será refeito quando o robô andar bem no MuJoCo.

Quantidades para **duas pernas + corpo + colar + chapéu** (ADR-001). Tudo de prateleira; alternativas por região na última coluna.

| Item | Qtd | Spec | Onde |
|---|---|---|---|
| Servo Feetech STS3215 7,4 V (serial bus, 12 bit) | 12 | 3 quadril + joelho + tornozelo por perna, pitch + roll do chapéu. Vem com horn, disco traseiro e parafusos M2 | **BR:** Mercado Livre, kit 4× 7,4 V — [MLB7374017854](https://produto.mercadolivre.com.br/MLB-7374017854), R$ 452,31/kit em 25/10/2026 (3 kits = 12). Internacional: AliExpress (Feetech oficial), Waveshare (ST3215), RobotShop, Eckstein (EU), Core Electronics (AU) |
| Placa controladora de servos serial (TTL) | 1 | Waveshare "Bus Servo Adapter" ou Feetech URT-1 | idem |
| Rolamento 688ZZ | 12 | 8 × 16 × 5 mm, blindado (10 pernas + 2 pescoço) | qualquer loja de rolamentos / skate / AliExpress |
| Inserto roscado a quente M3 | 11 | 4 pelve + 3 pescoço (corpo) + 4 moldura da tela (corpo). M3 × 5,7 mm, OD 4,6 mm | AliExpress, Amazon ("M3 heat set insert 5.7") |
| Parafuso M3 × 8 allen cilíndrico | 11 | 4 corpo→pelve, 3 colar→corpo, 4 moldura→corpo. Aço inox | ferragem local |
| O-ring NBR 25 × 3 mm | 4 | 2 por pé, canal impresso | loja de vedação / AliExpress |
| Chave tátil 6 × 6 mm | 4 | 2 por pé, altura 4,3–5 mm | qualquer loja de eletrônica |
| Tela ESP32-2432S028R "CYD" 2,8" | 1 | 86 × 50 mm, ILI9341 320×240, touch resistivo | AliExpress / Amazon |
| Raspberry Pi Zero 2W | 1 | fica em pé atrás da tela, em trilhos impressos | revendedores oficiais |
| IMU BNO055 (breakout estilo Adafruit, 20 × 27 mm, 2 furos a 20 mm) | 1 | na prateleira traseira do corpo (`IMU_POS`) | Adafruit 2472 via Mouser / DigiKey. **Não há no Mercado Livre** (ver nota) |
| Bateria LiPo 2S 1000 mAh soft pack | 1 | ≤ 14 × 31 × 58 mm, em pé na baia traseira, cinta por 2 fendas | HobbyKing (Turnigy 1000 mAh 2S). **Nenhuma no Mercado Livre cabe** (ver nota) |
| Regulador 5 V (buck 3 A) | 1 | Mini360 (MP2307): 4,75–23 V in, 3 A pico / 1,8 A contínuo, 17 × 11 mm | **BR:** ML kit 2× R$ 19,90 — [MLB2076436889](https://www.mercadolivre.com.br/p/MLB2076436889). Alternativa com saída USB-A 5,2 V 3 A (6–26 V in, 28 × 32 × 8 mm): R$ 25,99 — [MLB56492804](https://www.mercadolivre.com.br/p/MLB56492804) |
| Chave liga/desliga Ø12 | 1 | botão metálico 12 mm com trava (retenção), furo Ø12, cabeça 14 mm, 19 mm de comprimento, 3 A | **BR:** ML kit 6× R$ 93,80 — [MLBU3098744716](https://www.mercadolivre.com.br/up/MLBU3098744716); avulso com LED R$ 23 — [MLBU1432091586](https://www.mercadolivre.com.br/up/MLBU1432091586) (LED 12–24 V, não acende em 7,4 V) |
| USB-C painel | 1 | conector USB-C fêmea em placa adaptadora DIP; a face do conector passa pela fenda de 10 × 5 mm (`BODY_USBC_SLOT`) | **BR:** ML avulso R$ 25 — [MLBU5183896373](https://www.mercadolivre.com.br/up/MLBU5183896373); kit 5× R$ 19 — [MLBU5388387513](https://www.mercadolivre.com.br/up/MLBU5388387513) |
| Câmera Pi + ToF VL53L5CX + anel WS2812 | 1 cada | sob a aba do chapéu (janela frontal da fenda, canal do anel na borda) | — |
| Filamento PETG | ~450 g | creme (corpo, colar, pernas) + coral (chapéu) + 3 discos creme para as manchas | — |

Parafusos de servo: só os **M2 originais** que vêm com cada STS3215 (8 para o corpo, 4 para o horn,
4 para o disco traseiro). Até aqui só M3 × 8 é usado; o BOM completo do robô mantém apenas M3 × 8 / 12 / 16.

Anúncio do Mercado Livre (conferido em 25/10/2026): STS3215 7,4 V, 19,5 kg·cm a 7,4 V, encoder magnético
12 bits, redução 1/345, 5–8,4 V, TTL 1 Mbps — a mesma ficha do servo usado no CAD e na sim (±1,9 N·m).
**Não** é a versão 12 V / 30 kg·cm. Vendedor Xinyi Yang Store, envio da China, máx. 5 kits por compra.
A página lista impostos estimados (R$ 119,53 importação + R$ 117,08 ICMS por kit) sem deixar claro se já
estão nos R$ 452,31: confirmar o total no checkout (R$ 113 ou R$ 172 por servo). Placa controladora não
vem no kit; conteúdo (horn, disco, parafusos M2, cabo) conferir na chegada, pois a descrição está cortada.
Para a fase "uma perna real" lembrar que uma perna usa 5 servos: 1 kit de 4 não basta.

Busca no Mercado Livre em 25/10/2026 (IMU, bateria, regulador, chave, USB-C):

- **IMU**: a medida do BOM (20 × 27 mm, 2 furos a 20 mm, `BNO055` em `cad/parameters.py`) é a da Adafruit 2472, que não
  aparece em nenhum anúncio, nem com filtro de marca Adafruit. O que existe são dois outros formatos: **CJMCU-055** (roxa,
  quase quadrada, header de 10 pinos de um lado e 2 furos do outro com espaçamento bem menor que 20 mm; nacional, R$ 238,67,
  [MLBU1738983155](https://www.mercadolivre.com.br/up/MLBU1738983155)) e **GY-BNO055** (azul, 12 × 20 mm, 2 furos em cantos
  opostos; importada, R$ 48–70, p. ex. [MLB2065186027](https://www.mercadolivre.com.br/p/MLB2065186027)). Para usar uma
  delas é preciso trocar o dicionário `BNO055` no CAD pelo footprint real, medido na chegada.
- **Bateria**: a medida do BOM corresponde à Turnigy 1000 mAh 2S (HobbyKing), que não é vendida no ML. Medidas reais das
  2S 1000 mAh nacionais abertas: Futaba MK 20C 54 × 29 × 20 mm (R$ 139,99, [MLB2089585852](https://www.mercadolivre.com.br/p/MLB2089585852));
  30C Futaba 57 × 30 × 20 mm (R$ 169,90, [MLB2081996195](https://www.mercadolivre.com.br/p/MLB2081996195));
  Gens Ace 45C 70 × 33 × 11 mm (R$ 209, [MLB2110421643](https://www.mercadolivre.com.br/p/MLB2110421643));
  "3 vias" 76 × 56 × 6,8 mm (R$ 99,95); Leão 1100 mAh airsoft 2 células de 105 × 20 × 5,5 mm (R$ 149,90).
  As duas de 20 mm de espessura caberiam se `BATTERY_2S` crescer de 14 para 20 mm na baia traseira.
- **USB-C**: o projeto ainda não define se a porta é para carga da 2S ou só dados; carregar 2S por USB-C exige uma placa
  carregadora de 2 células, que não está no BOM.
Pi Zero, BNO055 e placa de servos ficam em trilhos / pinos impressos (sem M2,5 no BOM); furos exatos da CYD e
da placa de servos ainda não foram medidos — a moldura prende a CYD pela borda do vidro.
