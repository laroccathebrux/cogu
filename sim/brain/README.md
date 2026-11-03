# sim/brain — o cérebro do Cogu no simulador (ADR-004)

O cérebro do bichinho (percepção, memória do mundo, navegação, comportamento, conversa) é prototipado aqui, contra o
MuJoCo, antes do app em Swift. Tudo que o celular vai ver sai da câmera `phone_cam` do modelo.

| Arquivo | O que faz |
|---|---|
| `apartment.py` | Monta um apartamento em volta do robô com MjSpec (o MJCF do robô não é tocado) → `build/mjcf/apartment.xml` |
| `phone_sensors.py` | O LiDAR do iPhone (`Lidar`) e a posição do ARKit (`ArkitPose`), simulados a partir da `phone_cam` |

## Apartamento

Sala de 4 × 3,5 m com sofá, mesa de centro (40 cm: ele vê por baixo, mas não passa), rack com TV preta (o LiDAR a
perde), planta e tapete; uma porta para um corredor de 2,5 m; uma **bola de futebol** tamanho 5 (22 cm, 430 g, casca
oca que rola, `--ball` para os tamanhos 3 e 4) que o robô pode empurrar; uma **pessoa** de 1,7 m (corpo mocap: o
cérebro a move; ela empurra o robô se andar contra ele). O teto existe só para a câmera e o LiDAR (grupo 5), não
para o viewer. O robô continua primeiro no `qpos`, então as políticas leem como antes.

```bash
uv run python sim/brain/apartment.py
uv run mjpython sim/playground/play.py --checkpoint sim/playground/checkpoints/cogu_walk_v10 \
    --getup sim/playground/checkpoints/cogu_getup_v8 --pose script --xml build/mjcf/apartment.xml
```

A marcha v10 anda igual dentro dele (8 s a 0,15 m/s: 1,30 m, como no modelo de treino); sentar, levantar e deitar
também.

## LiDAR e ARKit simulados

O robô usa o LiDAR do iPhone. O simulador não tem LiDAR, então `Lidar` renderiza a profundidade da `phone_cam` e a
estraga como o sensor real:

- mapa de 256 × 192, profundidade em metros, 0 = sem leitura;
- alcance ~5 m; ruído crescendo com a distância (~1 cm a 1 m);
- superfícies escuras (e o que se chamar glass / mirror) quase não voltam;
- bordas com salto de profundidade ("flying pixels") com confiança baixa, como o mapa de confiança do ARKit (0, 1, 2);
- ARKit: a pose da câmera chega 50 ms atrasada, com deriva lenta (posição e rumo) e ruído.

```bash
uv run python sim/brain/phone_sensors.py --at 2.1 0.9 -145    # -> docs/renders/sim_lidar.png
```

![câmera, profundidade, confiança e pontos vistos de cima](../../docs/renders/sim_lidar.png)

## Próximos passos

1. Navegação e comandos com percepção perfeita (posição verdadeira da pessoa e da bola): siga, pare, vem, fica.
2. Percepção só pela câmera e pelo LiDAR simulados: mapa de obstáculos, pessoa, bola.
3. Memória do mundo e motor de comportamento.
4. Conversa (ElevenLabs e o modelo local) por cima, chamando as mesmas ferramentas.
