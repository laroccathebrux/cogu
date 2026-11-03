# sim/brain — o cérebro do Cogu no simulador (ADR-004)

O cérebro do bichinho (percepção, memória do mundo, navegação, comportamento, conversa) é prototipado aqui, contra o
MuJoCo, antes do app em Swift. Tudo que o celular vai ver sai da câmera `phone_cam` do modelo.

| Arquivo | O que faz |
|---|---|
| `apartment.py` | Monta um apartamento em volta do robô com MjSpec (o MJCF do robô não é tocado) → `build/mjcf/apartment.xml` |
| `phone_sensors.py` | O LiDAR do iPhone (`Lidar`) e a posição do ARKit (`ArkitPose`), simulados a partir da `phone_cam` |
| `navigation.py` | Mapa do chão, planejador e seguidor de caminho que viram o comando da marcha (vx, vy, wz) |
| `pet.py` | Os comandos simples (pare, fica, vem, siga) no apartamento, a pessoa que anda, os cenários de teste e o viewer |

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

## Comandos com percepção perfeita: pare, fica, vem, siga

O primeiro passo do cérebro usa as posições verdadeiras (robô, pessoa, bola) em vez da câmera; a navegação e os
comandos são os mesmos que o app vai usar quando a percepção vier do LiDAR e da câmera.

- **pare**: zera o comando na hora (sem rampa) e fica parado até o próximo comando. Para em ~0,4 s, andando a
  0,25 m/s, e não sai do lugar (2 cm).
- **fica**: fica onde está e vira para a pessoa quando ela sai de ±25° da frente.
- **vem**: vai até 0,5–0,6 m da pessoa, por onde for mais curto e longe dos móveis, para de frente para ela e passa
  para "fica".
- **siga**: anda atrás da pessoa a ~0,75 m, inclusive pela porta e pelo corredor; espera virado para ela enquanto ela
  estiver a menos de 1 m e à vista, e recua devagar se ela vier para cima dele.

Sinônimos: para / stop / chega; fique / espera; vem aqui / aqui; segue / me segue / vem comigo. O reconhecimento de
fala do celular vai entregar essas palavras direto, sem LLM (ADR-004, itens 4 e 6).

Como funciona (`navigation.py`): um mapa de 4 cm do chão com tudo que é sólido e mais baixo que o robô (a mesa de
centro conta: ele não passa por baixo), mais a pessoa (raio 25 cm) e a bola como círculos. A cada 0,5 s o cérebro
roda um Dijkstra a partir do robô (scipy, ~5 ms) e escolhe o destino como a célula mais barata no anel certo em volta
da pessoa e que a enxerga. Passagens a menos de 10 cm de um obstáculo são proibidas, e a menos de 30 cm custam mais
(ele anda pelo meio). Entre um plano e outro, um pure pursuit transforma o caminho em vx e wz. O comando muda em
rampa (0,5 m/s², 3 rad/s²), e a velocidade máxima é 0,25 m/s (a marcha v10 treinou até 0,3).

Dois reflexos que não dependem do plano:
- não anda para a frente se o ponto 15 cm à frente encosta em algo, a pessoa inclusive;
- antes de girar no lugar perto de algo (menos de 14 cm), dá um passo de lado ou para trás para ter espaço. Girar
  varre ~11 cm em volta da pelve, e o robô começa a 10 cm da mesa de centro.

Uma queda passa para o levantar v8, e o comando continua depois.

A pessoa é o corpo mocap do apartamento. Nos testes ela faz uma volta (sala, corredor, sala) a 0,25 m/s e desvia do
robô a 40 cm sem atravessar parede, como alguém que contorna o bicho no caminho. No viewer, ela é arrastada com o
mouse.

```bash
uv run python sim/brain/pet.py --test                      # cenários sem janela (~20 s)
uv run python sim/brain/pet.py --plot --video              # -> docs/renders/sim_pet_commands.png, sim_pet_siga.mp4
uv run mjpython sim/brain/pet.py                           # viewer: digite os comandos no terminal e Enter
```

No viewer, digite `siga`, `pare`, `vem` ou `fica` no terminal (é o lugar da fala). `andar` faz a pessoa dar a volta e
`parar` a segura. End é "pare". Para mover a pessoa, clique duas vezes nela e arraste com Ctrl + botão direito.

Resultados (2026-11-03, `--test`, sem atraso nos servos):

| Cenário | Resultado |
|---|---|
| vem, pessoa ao lado do sofá (2,0; 1,2) | chega em 10,2 s, para a 0,64 m, de frente (1°) |
| vem, pessoa no corredor (4,2; −1,25) | passa pela porta, chega em 19,1 s, para a 0,65 m (17° de frente) |
| vem, pessoa atrás do robô (−1,0; 0) | sai de perto da mesa, vira, chega em 8,2 s, 0,66 m |
| siga, a volta inteira (90 s) | distância média 0,97 m, 90% do tempo abaixo de 1,34 m, máximo 1,43 m, mínimo 0,39 m; entra no corredor (x 3,4 m) |
| siga, pare aos 15 s, siga aos 30 s | parado em 0,42 s, desloca 2 cm até o "siga"; depois segue como antes |
| fica, a pessoa dá a volta | sai 28 cm do lugar (girar no lugar desliza), erro médio de 20° para a pessoa |

Nenhuma queda e nenhum toque em móvel, parede ou pessoa em nenhum cenário. Com 40 ms de atraso nos servos
(`--delay 2`), os números ficam parecidos. A exceção é o "vem" ao lado do sofá: o pé encosta na mesa em 2 passos
de controle, logo na saída. Com a pessoa a 0,4 m/s (`--person-speed 0.4`), o robô fica até 2,5 m para trás e alcança
quando ela para. Nesse caso, uma vez ela passa a 24 cm dele (o reflexo segura o avanço) e encosta por 2 passos de
controle. Gente de verdade anda a ~1,2 m/s, então no "siga" ele vai ficar para trás e alcançar nas paradas.

![trajetos no siga e no vem](../../docs/renders/sim_pet_commands.png)

Revisável:
- velocidade máxima 0,25 m/s, abaixo dos 0,3 do treino da marcha;
- no "fica" ele só gira (sem voltar ao ponto) e desliza ~30 cm numa volta inteira da pessoa;
- distâncias: 0,5–0,6 m no "vem" e 0,7–0,8 m no "siga".

## Próximos passos

1. ~~Navegação e comandos com percepção perfeita (posição verdadeira da pessoa e da bola): siga, pare, vem, fica.~~
   Feito (acima).
2. Percepção só pela câmera e pelo LiDAR simulados: mapa de obstáculos, pessoa, bola.
3. Memória do mundo e motor de comportamento.
4. Conversa (ElevenLabs e o modelo local) por cima, chamando as mesmas ferramentas.
