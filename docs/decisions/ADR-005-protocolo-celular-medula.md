# ADR-005 — O protocolo entre o celular e a medula

Data: 2026-11-04. Status: **aceita a versão 1**, testada só no simulador; os itens revisáveis no fim dependem da
bancada (ESP32, servos e bateria de verdade). Detalha a ligação dos ADR-002 e ADR-004: o celular pensa e navega, a
medula (a Waveshare Servo Driver with ESP32) anda, senta, deita e levanta.

## Problema

No simulador, o cérebro (`sim/brain/pet.py`) mandava a velocidade direto para a política de marcha, a cada passo de
controle de 20 ms, sem atraso e sem perda. No robô, o celular e a medula estão ligados por rádio (BLE): os pacotes
chegam com atraso, às vezes fora de ordem, às vezes não chegam, e a ligação pode cair por um segundo. Falta um
contrato que o firmware do ESP32 e o app do iPhone sigam, e saber se o Cogu continua sem cair com um rádio ruim.

## Decisão

1. **Dois pacotes binários de no máximo 20 bytes**, para caber numa escrita ou notificação BLE com o MTU padrão
   (23), sem negociar MTU. Os mesmos bytes servem depois para ESP-NOW ou UDP. Campos little-endian, CRC-16/CCITT-FALSE
   nos 2 últimos bytes. A referência é `sim/brain/protocol.py`; os pacotes de exemplo em hex que ele imprime são o
   teste que o firmware e o app têm de passar.

   **COMANDO** (celular → medula, 14 bytes, 10 vezes por segundo, e na hora num "pare" ou num pedido de postura):

   | Bytes | Campo | Unidade |
   |---|---|---|
   | 0 | versão (1) << 4 \| tipo (1) | 0x11 |
   | 1 | seq | u8, +1 a cada pacote |
   | 2–7 | vx, vy, ωz | i16 mm/s, mm/s, mrad/s |
   | 8 | postura pedida | 0 em pé, 1 sentado, 2 deitado |
   | 9 | id do pedido de postura | u8, +1 a cada pedido novo |
   | 10 | flags | bit 0 PARE (zera a velocidade já, sem rampa) |
   | 11 | reservado | 0 (depois: altura e inclinação do corpo, a postura expressiva do ADR-004) |
   | 12–13 | CRC | |

   **TELEMETRIA** (medula → celular, 20 bytes, 10 vezes por segundo):

   | Bytes | Campo | Unidade |
   |---|---|---|
   | 0 | versão << 4 \| tipo (2) | 0x12 |
   | 1 | seq | u8 |
   | 2 | ack | o último seq de comando aceito |
   | 3 | estado | andando, sentando, sentado, deitando, deitado, levantando (da postura), levantando (da queda) |
   | 4 | id do último pedido de postura atendido | u8 |
   | 5 | flags | bit 0 watchdog, bit 1 bateria baixa, bit 2 corte, bit 3 falha de servo |
   | 6–7 | bateria | u16 mV |
   | 8–11 | roll, pitch (IMU da medula) | i16 centésimos de grau |
   | 12–15 | vx, ωz que a marcha está fazendo | i16 mm/s, mrad/s |
   | 16 | quedas desde que ligou | u8 |
   | 17 | reservado | 0 |
   | 18–19 | CRC | |

2. **Cada comando traz o estado inteiro, não um evento.** Perder um pacote só atrasa 100 ms; o próximo traz tudo de
   novo. A postura é o único "evento", e vai como **pedido com id**: a medula age uma vez por id novo, então um pacote
   repetido, atrasado ou perdido não faz o Cogu sentar duas vezes nem esquecer o pedido.
3. **A medula se protege sozinha, diga o celular o que disser** (`sim/brain/medula.py`):
   - limita a velocidade à faixa em que a marcha foi treinada e faz a rampa (0,5 m/s², 3 rad/s²);
   - **watchdog**: sem comando válido por 0,3 s, fica parada no lugar (velocidade 0, em pé) até o próximo;
   - pacote com CRC errado, versão errada ou seq velho é descartado;
   - caiu, levanta sozinha (get-up v8) e continua com o último pedido;
   - bateria abaixo de 7,0 V (3,5 V por célula) avisa; abaixo de 6,8 V senta, ignora velocidade e "em pé".
4. **O "pare" não espera o próximo ciclo:** o celular manda o comando na hora, com a flag PARE, e repete a flag em
   todos os comandos enquanto estiver parado.
5. No simulador, o celular (`pet.py`: percepção, cérebro, comportamento, conversa) só fala com a medula por esses
   pacotes, num rádio falso com atraso, variação, perda, bytes corrompidos e quedas da ligação
   (`medula.PROFILES`). O perfil `ideal` (a cada passo, sem atraso) é o comportamento de antes.

## Resultados no simulador

Perfis do rádio: **ideal** (50 Hz, sem atraso); **ble** (10 Hz, 30 ± 20 ms, 2 % de perda, 0,2 % corrompidos);
**ruim** (10 Hz, 80 ± 60 ms, 10 % de perda, 1 % corrompidos e 1 s sem ligação a cada 20 s).

Os cenários de `pet.py --test` (percepção só pelo celular), em cada perfil:

| Cenário | Antes (direto) | ideal | ble | ruim |
|---|---|---|---|---|
| vem, ao lado do sofá: chega em | 11,3 s | 11,3 s | 11,3 s | 12,5 s |
| vem, no corredor | 20,6 s | 20,7 s | 19,9 s | 21,0 s |
| vem, atrás do robô | 8,7 s | 8,7 s | 8,7 s | 8,7 s |
| siga, a volta: distância média / p90 | 1,11 / 2,04 m | 1,10 / 2,03 m | 1,13 / 2,15 m | 1,13 / 2,10 m |
| siga, "pare" a 0,2 m/s: parado em | 0,22 s | 0,34 s | 0,32 s | 0,30 s |
| fica, a pessoa dá a volta: deriva | 0,21 m | 0,22 m | 0,20 m | 0,42 m |
| quedas / toques em móveis, em todos | 0 / 1 | 0 / 1 | 0 / 0 | 0 / 0 |

No ruim, na volta do siga (90 s): 261 pacotes perdidos nos dois sentidos, 6 recusados pelo CRC, 2 atrasados
descartados, 8 vezes o watchdog parou o Cogu (as quedas de 1 s da ligação e perdas seguidas), maior buraco 1,18 s.
O Cogu não caiu e não bateu em nada; o "pare" continuou abaixo de 0,35 s. A diferença do "pare" entre "antes" e
"ideal" é a fase do passo em que ele chega (a velocidade é a mesma), não o rádio. O que mais piora no ruim é o
"fica": ele escorrega mais (0,42 m em 75 s), provavelmente porque cada queda da ligação o para no meio de uma
virada e ele recomeça.

A medula sozinha (`medula.py --test`):
- o celular para de mandar com o Cogu a 0,25 m/s: parado 0,82 s depois do último comando, 0,12 m adiante;
- "senta" repetido 40 vezes, com 35 pacotes embaralhados: senta uma vez só, 21 pacotes velhos descartados;
- bateria quase no fim: avisa na hora, corta aos 4,5 s, senta sozinho e recusa o "em pé" pedido depois.

O comportamento (`behavior.py --test`), que pede posturas pela medula:

| Cenário | ideal | ruim |
|---|---|---|
| senta / vem / deita / levanta: pronto em | 4,0 / 11,6 / 8,0 / 6,2 s | 4,2 / 13,6 / 8,2 / 6,6 s |
| sozinho: senta perto da pessoa, levanta quando ela some, procura e acha | 44 / 15 / 20 s | 44 / 15 / 16 s |
| vai até a bola / o corredor | 7,3 / 12,2 s | 8,2 / 14,2 s |
| quedas / toques | 0 / 0 | 0 / 0 |

O "vai até o sofá" chegava em 1 de 3 sementes nos dois perfis e também no código de antes do protocolo: era a
navegação perto do sofá, não o rádio. Corrigido em seguida (o ponto de chegada agora tem folga e não pula de lado;
`sim/brain/README.md`): chega em 8 de 8 sementes, em 10–16 s, nos perfis ideal e ruim.

## Alternativas consideradas

- **Texto (JSON) em vez de binário**: mais fácil de depurar, mas um comando passa de 20 bytes e precisaria negociar
  MTU no BLE. O binário cabe em qualquer conexão e o `protocol.py` imprime cada pacote em hex para depurar.
- **Mandar só quando muda**: menos rádio, mas um pacote perdido deixaria a medula com a velocidade velha. Com o
  estado inteiro a 10 Hz, o watchdog também detecta a queda da ligação.
- **Watchdog que desliga o torque** (como estava no ADR-004): o Cogu cairia toda vez que o BLE piscasse. Ficar
  parado em pé é seguro e volta sozinho quando a ligação volta. Desligar o torque fica para queda e falha de servo.
- **Política de marcha no celular**: descartada no ADR-002; o jitter do rádio iria direto para a marcha.

## Consequências

- O firmware da medula e o app do iPhone têm um contrato fechado e testes prontos (os pacotes de exemplo).
- A marcha nunca depende do rádio: com a ligação caída, o Cogu para em menos de um segundo e espera.
- Mudar o formato exige subir a versão no byte 0; a medula descarta versões que não conhece.

## Revisável (depende da bancada)

- As correntes da bateria (2 A andando, 3 A levantando, 0,3 A sentado) são chutes; os limites de 7,0 e 6,8 V também.
- 0,3 s de watchdog e 10 Hz de comando: medir o atraso real do BLE entre o iPhone e o ESP32.
- Se ficar sem ligação por muito tempo (por exemplo 30 s), sentar para economizar bateria.
- A falha de servo (bit 3) ainda não é simulada; o barramento dos STS3215 informa temperatura, tensão e erro.
