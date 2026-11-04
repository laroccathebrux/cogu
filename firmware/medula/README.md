# A medula: o firmware do ESP32

A medula anda, senta, deita, levanta, cuida da bateria e conversa com o celular pelos pacotes do
[ADR-005](../../docs/decisions/ADR-005-protocolo-celular-medula.md). Este é o código que vai rodar na placa Waveshare
Servo Driver with ESP32, e é **o mesmo código** que o simulador roda no lugar de `sim/brain/medula.py`: o MuJoCo faz
o papel do corpo (software in the loop). Se o Cogu anda no simulador com este firmware, o firmware está certo antes
de a placa existir.

| Pasta | O que é |
|---|---|
| `src/` | O núcleo, C++ portátil sem nada de Arduino: protocolo (`protocol`), a rede neural (`mlp`), a marcha v10, o levantar v8 e o sentar/deitar com roteiro (`controllers`), as regras da medula (`medula`), o filtro da IMU (`imu`). `sil.cpp` é a ponte para o Python (só no computador). |
| `gen/` | Gerado por `tools/export.py` a partir dos checkpoints e das configurações do simulador: os pesos (`cogu_policies.h`) e cada constante e tabela (`cogu_params.h`). Não é editado à mão e não vai para o git. |
| `test/` | Os testes nativos: os pacotes de exemplo do `protocol.py` byte a byte, toda corrupção de 1 byte recusada. |
| `esp32/` | O projeto PlatformIO da placa: barramento dos servos STS3215, IMU de 6 eixos (MPU-6050), BLE (NimBLE), o laço de 50 Hz e um console serial para a bancada. Compila; ainda não rodou na placa. |

```bash
make -C firmware/medula                              # gera os headers, a biblioteca do simulador e roda os testes nativos
uv run python sim/brain/medula_c.py --test           # o firmware contra a medula em Python, e sozinho
uv run python sim/brain/pet.py --test --medula c     # o cérebro inteiro em cima do firmware
uv run python sim/brain/pet.py --test --medula robot # ... sentindo só o que o robô sente, com o filtro da IMU
uv run mjpython sim/brain/conversation.py --brain elevenlabs --voz --medula c   # conversar com o firmware
cd firmware/medula/esp32 && pio run                  # compilar para o ESP32 (uv tool install platformio)
```

## Como o firmware é conferido

1. **As redes:** os pesos exportados, numa reconstrução em NumPy e em C++, contra o JAX: diferença máxima de
   6 × 10⁻⁷ na ação, nas observações de uma corrida de verdade.
2. **Na sombra da medula em Python:** as duas recebem a mesma leitura dos sensores e os mesmos pacotes a cada passo,
   e a do Python move o corpo, numa sequência de 60 s que anda, vira, para, senta, deita, levanta, é empurrada,
   cai e levanta de novo. Os alvos dos servos ficam a no máximo 1,5 × 10⁻⁵ rad um do outro, e o estado e a
   telemetria são iguais em todos os 3000 passos.
3. **Sozinho:** os cenários do `medula.py` (celular mudo, "senta" repetido e embaralhado, bateria no fim) e os do
   cérebro (`pet.py --test` e `behavior.py --test`), com o firmware movendo o corpo.
4. **Como o robô:** o robô não mede a altura nem a velocidade do quadril. Com `--medula robot`, "parado" passa a ser
   o giroscópio abaixo de 0,3 rad/s (era a velocidade do quadril), "caído" é só a inclinação, e a gravidade vem do
   filtro do firmware alimentado pelo giroscópio e pelo acelerômetro simulados (era a orientação perfeita).

Um detalhe que só apareceu no C++: somar 0,02 s em `float` 15 vezes dá menos que 0,3, e o firmware disparava um passo
depois do Python (e o relógio da medula derivava). Os relógios e cronômetros ficam em `double`, como no Python;
o resto (a rede, os ângulos) fica em `float`.

## Resultados no simulador

`pet.py --test` (percepção só pelo celular, rádio ideal):

| Cenário | medula.py | firmware | firmware como o robô | robô, rádio ruim |
|---|---|---|---|---|
| vem, ao lado do sofá / no corredor / atrás: chega em | 11,3 / 20,7 / 8,7 s | 11,3 / 20,7 / 8,7 s | 11,1 / 19,1 / 8,5 s | 12,6 / 21,5 / 8,7 s |
| siga, a volta: distância média / p90 | 1,10 / 2,03 m | 1,12 / 2,09 m | 1,13 / 2,10 m | 1,16 / 2,17 m |
| siga, "pare": parado em | 0,34 s | 0,30 s | 0,22 s | 0,30 s |
| fica, a pessoa dá a volta: deriva | 0,22 m | 0,22 m | 0,26 m | 0,36 m |
| quedas / toques em móveis, em todos | 0 / 1 | 0 / 1 | 0 / 0 | 0 / 2 |

`behavior.py --test`:

| Cenário | medula.py | firmware | como o robô |
|---|---|---|---|
| senta / vem / deita / levanta: pronto em | 4,0 / 11,6 / 8,0 / 6,2 s | 4,0 / 11,6 / 8,0 / 6,3 s | 4,0 / 10,6 / 8,0 / 7,5 s |
| vai até o corredor / o sofá | 12,9 / 15,7 s | 13,7 / 15,7 s | 12,7 / 14,8 s |
| sozinho: senta, levanta quando a pessoa some, acha | 44 / 15 / 20 s | 44 / 15 / 20 s | 44 / 15 / 19 s |
| quedas / toques | 0 / 0 | 0 / 0 | 0 / 0 |

O "levanta" como o robô leva 1,3 s a mais: sem a velocidade do quadril, ele espera o giroscópio acalmar para
entregar à marcha.

## No ESP32

`pio run` compila para a ESP32-WROOM-32 da placa (`esp32dev`, partição de 3 MB): **963 kB de flash (31 %) e 41 kB de
RAM estática (13 %)**, com o BLE, as duas redes (320 kB de pesos, lidos direto da flash) e as tabelas. Sobra espaço
para mais redes (a pose v2, movimentos novos).

O tempo do laço na placa ainda não foi medido. No Mac a marcha leva 23 µs por passo; no ESP32 (240 MHz, os pesos
vindo da flash por um cache de 32 kB) a conta dá alguns milissegundos dos 20 do passo. O console serial (`s`)
imprime o tempo de cada parte (servos e IMU, medula, escrita) e o pior laço; se a rede passar de ~5 ms, os pesos da
marcha vão para a RAM (cabem: 160 kB) ou para float16.

O laço, a cada 20 ms: lê os 10 servos (posição e velocidade, um por vez no barramento a 1 Mbps) e a IMU, passa a
gravidade pelo filtro, entrega os pacotes que chegaram pelo BLE, roda a medula e manda os alvos numa SYNC WRITE só;
a telemetria sai por notificação BLE a 10 Hz. A tensão da bateria vem dos próprios servos (um por passo).

## Revisável (bancada)

- Os pinos (servos em 18/19, I²C em 21/22), os IDs dos servos e a calibração de cada junta (`esp32/src/board.h`):
  o console serial (`p`) imprime as posições cruas para medir o zero e o sentido de cada uma.
- A IMU: o MPU-6050 é um chute (o ADR-002 só diz "6 eixos"); a orientação dela no quadril (`IMU_TO_PELVIS`); o
  ganho do filtro (2 /s).
- O limiar de "parado" pelo giroscópio (0,3 rad/s) e "caído" só pela inclinação.
- O torque fica desligado ao ligar a placa, até o primeiro comando do celular (ou `t` no console), e a marcha
  começa de onde as pernas estão.
- A falha de servo só é avisada (bit 3 da telemetria); desligar o torque nela fica para depois de ver as falhas reais.
- O sentar e deitar continuam com o roteiro, sem a política pose v2 (como no simulador).
