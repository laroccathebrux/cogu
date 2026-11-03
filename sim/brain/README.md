# sim/brain — o cérebro do Cogu no simulador (ADR-004)

O cérebro do bichinho (percepção, memória do mundo, navegação, comportamento, conversa) é prototipado aqui, contra o
MuJoCo, antes do app em Swift. Tudo que o celular vai ver sai da câmera `phone_cam` do modelo.

| Arquivo | O que faz |
|---|---|
| `apartment.py` | Monta um apartamento em volta do robô com MjSpec (o MJCF do robô não é tocado) → `build/mjcf/apartment.xml` |
| `phone_sensors.py` | O LiDAR do iPhone (`Lidar`) e a posição do ARKit (`ArkitPose`), simulados a partir da `phone_cam` |
| `navigation.py` | Mapa do chão, planejador e seguidor de caminho que viram o comando da marcha (vx, vy, wz) |
| `perception.py` | O que o cérebro sabe: o estado verdadeiro, ou só o celular (mapa do LiDAR, pose do ARKit, pessoa e bola pela câmera) |
| `pet.py` | Os comandos simples (pare, fica, vem, siga) no apartamento, a pessoa que anda, os cenários de teste e o viewer |
| `memory.py` | A memória do mundo: o que está onde, quando foi visto, para que serve; lugares com nome; o resumo em texto |
| `behavior.py` | O motor de comportamento: as ferramentas que a conversa vai chamar, os comandos falados e o que ele faz sozinho |

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
uv run python sim/brain/pet.py --test --perception true    # cenários sem janela (~20 s)
uv run python sim/brain/pet.py --plot --video --perception true   # -> sim_pet_commands.png, sim_pet_siga.mp4
uv run mjpython sim/brain/pet.py --perception true        # viewer: digite os comandos no terminal e Enter
```

No viewer, digite `siga`, `pare`, `vem` ou `fica` no terminal (é o lugar da fala). `andar` faz a pessoa dar a volta e
`parar` a segura. End é "pare". Para mover a pessoa, clique duas vezes nela e arraste com Ctrl + botão direito.

Resultados (2026-11-03, `--test --perception true`, sem atraso nos servos):

| Cenário | Resultado |
|---|---|
| vem, pessoa ao lado do sofá (2,0; 1,2) | chega em 10,2 s, para a 0,64 m, de frente (1°) |
| vem, pessoa no corredor (4,2; −1,25) | passa pela porta, chega em 19,1 s, para a 0,65 m (17° de frente) |
| vem, pessoa atrás do robô (−1,0; 0) | sai de perto da mesa, vira, chega em 8,2 s, 0,66 m |
| siga, a volta inteira (90 s) | distância média 0,97 m, 90% do tempo abaixo de 1,34 m, máximo 1,43 m, mínimo 0,39 m; entra no corredor (x 3,4 m) |
| siga, pare aos 15 s, siga aos 30 s | parado em 0,42 s, desloca 2 cm até o "siga"; depois segue como antes |
| fica, a pessoa dá a volta | sai 25 cm do lugar (girar no lugar desliza), erro médio de 20° para a pessoa |

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

## Percepção só pelo celular

Os mesmos comandos, agora sabendo só o que o iPhone vai saber (`perception.py`, `PhonePerception`); é o padrão do
`pet.py` (`--perception true` volta à posição verdadeira). Nada do estado verdadeiro chega ao cérebro:

- **posição**: a do ARKit (50 ms atrasada, com deriva e ruído), levada da câmera à pelve pela montagem do celular;
- **mapa**: o LiDAR 10 vezes por segundo, cada quadro com a sua pose do ARKit, num grid de 4 cm (log-odds). Dois
  pontos numa célula entre 5 e 45 cm de altura, a menos de 3,5 m, são obstáculo; um ponto do chão (abaixo de 3 cm)
  libera a célula. O que ele nunca viu é passável, mas custa 3 vezes mais. Tudo fica no referencial do ARKit e deriva
  com ele, como no celular;
- **pessoa**: a segmentação de pessoas do ARKit (`personSegmentationWithDepth`). A posição é a mediana dos pontos do
  LiDAR na máscara, 10 cm para trás (o meio das pernas). Um quadro em 10 falha, e menos de 60 pixels não é detecção.
  Os pixels dela saem do mapa, e os do próprio robô também;
- **bola**: um detector de objetos (Core ML), centrada atrás dos pontos que ele vê;
- no simulador as duas detecções saem da renderização de segmentação da `phone_cam`, a mesma do LiDAR; as falhas
  são sorteadas.

O que muda no comportamento:
- com a pessoa fora da vista, o cérebro vai até onde a viu por último (ou para onde ela ia, até 1,5 s à frente) e,
  se ela não estiver lá, gira no lugar procurando;
- sem nunca a ter visto, gira até achá-la;
- no "fica", ele vira para onde a viu por último.

Um erro do simulador apareceu no caminho. O LiDAR usava a pose atrasada do ARKit, e o balanço da marcha virava
obstáculo falso no chão longe. No ARKit, cada quadro de profundidade vem com a pose do próprio quadro
(`ArkitPose.now`).

Resultados (2026-11-03, `--test`, sementes 0 e 2, e 1 com 40 ms de atraso nos servos):

| Cenário | Posição verdadeira | Só o celular |
|---|---|---|
| vem, pessoa ao lado do sofá | 10,2 s, para a 0,64 m | 11,1–11,8 s, 0,67–0,69 m |
| vem, pessoa no corredor | 19,1 s, 0,65 m | 19,3–21,4 s, 0,56–0,60 m |
| vem, pessoa atrás do robô (fora da vista) | 8,2 s, 0,66 m | 8,6–9,2 s, 0,58–0,59 m (gira até achar) |
| siga, a volta inteira | média 0,97 m, máximo 1,43 m | média 1,11–1,16 m, máximo 2,2–2,5 m |
| siga, pare aos 15 s | parado em 0,42 s, 2 cm | parado em 0,2–0,65 s, 2 cm |
| fica, a pessoa dá a volta | 20° de erro médio, 90% < 27° | 20–21° de erro médio, 90% < 60° (fora da vista) |

Nenhuma queda. No "siga" com o celular, ele demora no começo porque a pessoa está atrás dele, fora da vista, e
ele gira para o lado errado antes de achá-la; depois segue como antes. O reflexo segura o avanço por 0,7 a 1,3 s na
porta, quando os dois se cruzam. Num "fica" (semente 0), ele encosta o domo na mesa de centro por 3 passos de
controle e o pé num pé da mesa por 1. A mesa está a 90° à esquerda dele no começo, fora dos 90° de campo da câmera,
então o passo de lado antes de girar não a conhece.

![trajetos com a percepção do celular e o mapa que ele montou](../../docs/renders/sim_pet_phone.png)

O mapa que ele montou na volta (à direita) tem as paredes, o sofá, o rack, a planta, a bola e a frente e os lados da
mesa de centro. O tampo é visto só de lado, porque a câmera fica a 38,6 cm, na altura dele.

```bash
uv run python sim/brain/pet.py --test                      # só o celular (padrão)
uv run python sim/brain/pet.py --test --perception true    # a posição verdadeira
uv run python sim/brain/pet.py --plot --video              # -> docs/renders/sim_pet_phone.png, sim_pet_siga_phone.mp4
```

Revisável:
- procurar girando no lugar. Uma alternativa é ir explorar o que ele nunca viu ou perguntar "cadê você?"; isso entra
  com a memória do mundo e a conversa;
- células nunca vistas são passáveis (otimista): mais rápido, mas ele descobre os obstáculos ao chegar perto;
- a detecção da pessoa é a segmentação do simulador com falhas sorteadas, não uma rede rodando sobre a imagem
  renderizada.

## Memória do mundo e motor de comportamento

Em cima da navegação e da percepção do celular, `behavior.py` junta três coisas.

**Memória do mundo** (`memory.py`):
- Guarda cada coisa que os detectores veem: sofá, mesa, porta, rack, TV, planta, bola e pessoa. Cada uma tem posição,
  quando foi vista pela primeira e pela última vez e para que serve ("a mesa: baixa demais para passar por baixo").
- A bola e a pessoa ficam na última posição vista; os móveis, na média das vezes que foram vistos.
- Os lugares ganham nome por voz ("aqui é a sala"). O lugar de uma coisa é o lugar com nome mais perto que a enxerga.
- `summary()` é o texto que o modelo de linguagem vai ler, e é só isso que pode ir para a nuvem: "Estou na sala, em
  pé, seguindo você. A pessoa está a 0,8 m, à minha frente. Lembro de: a bola está perto da mesa, na sala (vi há
  37 s): dá para empurrar e brincar; ...".
- A memória e o mapa do LiDAR são salvos em disco (`build/brain_memory/`, fora do git) e carregados na sessão
  seguinte. No celular, o referencial volta com o mapa salvo do ARKit; no simulador, cada sessão começa no mesmo
  referencial.

**Ferramentas** (`Tools`): o que o modelo de linguagem vai chamar no passo 4, cada uma devolvendo o que o Cogu diz.
São elas: `parar`, `ficar`, `vir`, `seguir`, `livre`, `procurar`, `ir_ate(alvo)`, `sentar`, `deitar`, `levantar`,
`onde_esta(coisa)`, `o_que_vejo`, `lembrar_lugar(nome)` e `resumo`. Até lá, frases fixas chegam a elas sem LLM: "vai
até a bola", "vai para o corredor", "cadê a bola?", "o que você vê?", "aqui é a cozinha", "o que você sabe?",
"senta", "deita", "levanta", "livre", "me procura".

**Sozinho**, só quando está livre e nunca sob uma ordem como "fica":
- com a pessoa à vista, vira para ela e, depois de 40 s assim, senta para descansar;
- sentado por conta própria, levanta quando a perde de vista por 3 s;
- com a pessoa fora de vista por 15 s, vai procurá-la: primeiro onde a viu por último, depois girando. Se não a
  achar em 15 s, diz "Não te achei".

Mudanças por baixo:
- "deita" passa antes pelo sentar, como no `pose_eval`. Direto de pé ele tombava para trás em 1,7 s.
- "ir" para no ponto livre mais perto da coisa quando nada a menos de 0,5 m está livre.
- Uma mesa vista pelo celular bloqueia o contorno inteiro no mapa. A câmera fica na altura do tampo: vê a borda,
  os pés e o chão por baixo, e o mapa achava que dava para passar por baixo dela.

Resultados (2026-11-03, `uv run python sim/brain/behavior.py --test`, só o celular):

| Cenário | Resultado |
|---|---|
| memória depois da volta (siga) | erro das coisas: TV 0,13 m, bola 0,13, mesa 0,14, planta 0,16, porta 0,27, rack 0,22, sofá 0,37 (o lado que ele vê, não o meio) |
| "cadê a bola?" | "A bola está perto da mesa, na sala (vi há 37 s)." |
| "vai até a bola" | chega em 7,6 s, a 0,59 m da bola |
| "vai para o corredor" (nomeado na volta) | chega em 13,3 s (x 2,71 m, na porta do corredor) |
| "vai até o sofá" | chega em 17,7 s, a ~0,2 m da frente do sofá |
| sessão seguinte, memória e mapa do disco | "vai para o corredor" sem vê-lo do início: 15,5 s |
| livre, a pessoa parada à vista | senta sozinho aos 44 s |
| a pessoa passa por trás dele, fora da vista | levanta 14,5 s depois e sai procurando 7,6 s depois; acha a pessoa em 18,8 s, a 1,0 m |
| senta, vem (sentado), deita, levanta | sentado em 4 s; de pé e chega em 10,6 s; deitado e esticado em 8 s; de pé com o levantar v8 em 6,2 s |

Nenhuma queda nem toque. Com `--perception true` a memória é exata, mas a autonomia de procurar não aparece: ele
vê através das paredes e nunca perde a pessoa.

```bash
uv run python sim/brain/behavior.py --test                              # os cenários (~3 min)
uv run mjpython sim/brain/behavior.py                                   # viewer: converse no terminal
uv run mjpython sim/brain/behavior.py --memory build/brain_memory       # com a memória da última sessão
```

Revisável:
- tempos da autonomia: senta depois de 40 s, procura depois de 15 s sem ver a pessoa e desiste depois de 15 s;
- a posição de um móvel é a média dos lados que ele viu (o sofá fica ~0,4 m para a frente do meio). Para ir até
  ele e dizer onde está, isso basta;
- lugares só por nome dado pela voz. Salas automáticas, separando o mapa por portas, ficam para quando houver mais
  de um cômodo para testar.

## Próximos passos

1. ~~Navegação e comandos com percepção perfeita (posição verdadeira da pessoa e da bola): siga, pare, vem, fica.~~
   Feito (acima).
2. ~~Percepção só pela câmera e pelo LiDAR simulados: mapa de obstáculos, pessoa, bola.~~ Feito (acima).
3. ~~Memória do mundo e motor de comportamento.~~ Feito (acima).
4. Conversa (ElevenLabs e o modelo local) por cima, chamando as mesmas ferramentas.
