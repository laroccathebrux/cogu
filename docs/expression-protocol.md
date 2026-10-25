# Protocolo de expressão (Pi → ESP32 / CYD)

Rascunho v0. A locomoção (Raspberry Pi) e a expressão (ESP32 na tela CYD) são **desacopladas**:
o ESP32 renderiza os olhos sozinho; o Pi só envia o **estado de expressão**.

## Transporte

- UART 115200 8N1 no conector P1 da CYD (TX/RX/GND, 3,3 V). Alternativa: ESP-NOW/Wi-Fi.
- Mensagens de texto ASCII, uma por linha, terminadas em `\n`. Fáceis de testar com um terminal serial.

## Mensagens

```
EXPR <estado> <x> <y>     # estado: neutral | happy | sleepy | surprised | sad | blink
                          # x, y: direção do olhar, -100..100 (0 0 = frente)
LED <r> <g> <b> [modo]    # anel de LEDs do chapéu (se controlado pelo ESP32); modo: solid | breathe | spin
PING                      # ESP32 responde PONG <uptime_ms>
```

Eventos do ESP32 para o Pi:

```
TOUCH <x> <y> <down|up>   # toque na tela ("carinho")
```

Regras: comando desconhecido é ignorado com `ERR <texto>`; sem mensagens por 10 s o ESP32 volta ao estado `neutral`.
