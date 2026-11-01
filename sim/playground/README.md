# sim/playground — política de caminhada (MuJoCo Playground / MJX)

Primeira política de locomoção do Cogu: **velocity tracking** (seguir um comando `vx, vy, ω_z`)
treinada com PPO (Brax) sobre MJX, no estilo dos envs de bípedes do MuJoCo Playground
(Berkeley Humanoid joystick) e do Open Duck Playground.

O modelo é o MJCF **gerado do CAD** — `build/mjcf/cogu_lower_body_mjx.xml` — nunca editado à mão.
Se o robô precisar mudar (kp, damping, alcance, massa), mude `cad/parameters.py` ou `cad/export.py`
e regenere com `uv run python -m cad.export`.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `cogu_env.py` | Env `CoguJoystick(mjx_env.MjxEnv)`: observações, recompensa, terminação, latência do atuador, empurrões |
| `randomize.py` | Domain randomization por env: atrito do chão, massas, carga extra na pelve, modelo do servo (atrito, inércia, kp/kv, torque de travamento com a tensão da bateria, amortecimento do motor), offset de pose |
| `train.py` | PPO (Brax) com seed fixa; salva checkpoints em `build/checkpoints/cogu_walk/<passo>/` e `progress.jsonl` |
| `rollout.py` | Roda a política a partir do keyframe `rest` com um comando fixo e grava `build/videos/walk.mp4` |
| `compare.py` | Compara checkpoints no modelo atual: distância em 8 s e taxa de queda sob empurrões |
| `colab_train.ipynb` | Treino em GPU NVIDIA no Google Colab: gera o modelo do CAD, treina, testa sim2sim, grava vídeo e devolve um zip com o checkpoint |
| `play.py` | Roda a política no **viewer do MuJoCo**, comandada pelo teclado, no MuJoCo normal (sim2sim). `--follow` acompanha um treino em andamento |
| `command_tracking.py` | Quanto a política segue cada comando (frente, trás, lado, giro, parado): velocidade atingida e, parado, tempo com pé no ar |
| `check_env.py` | Smoke test do env (CI): fica em pé 2 s, observações finitas, randomizer em lote |
| `jax_compat.py` | Shim: Brax 0.14.2 ainda chama `jax.device_put_replicated`, removido do namespace público no jax 0.10 |

## Modelo do servo (2026-10-30)

Até a v4, o servo simulado tinha torque máximo (±1,9 N·m) mas **velocidade ilimitada**. O STS3215 a 7,4 V é um
motor DC com redução 1:345 e malha de posição própria: o torque disponível cai linearmente até zero na
velocidade sem carga (4,72 rad/s). O MJCF agora monta isso com peças nativas do MuJoCo/MJX: atuador de posição
saturado em ±1,91 N·m (o PD da malha do servo) mais amortecimento de junta igual a torque de travamento ÷
velocidade sem carga, 0,40 N·m·s/rad (a força contraeletromotriz). Atrito de 0,05 N·m e inércia refletida de
0,01 kg·m² são estimativas, para medir num servo de verdade. Valores em `ServoSTS3215` (`cad/parameters.py`).

A v4 usava o joelho a 4 rad/s, o que um servo com carga não alcança. Neste modelo ela cai a 0,25 e 0,30 m/s e
segue os outros comandos com menos força. Esse é o motivo da v5.

**v5.** Mudanças no env: com comando zero a fase da marcha fica parada na posição inicial (os dois
pés no chão) e um custo novo cobra pé no ar (`feet_air_still`); o rastreamento ficou mais exigente
(`tracking_sigma` 0,05 → 0,02, giro 0,5 → 0,2), porque a v4 seguia só metade dos comandos de lado e de ré. Teste
local (`command_tracking.py`, modelo com o iPhone 15 Pro Max):

- do zero, 3 M passos: ficou parada com qualquer comando, sem nunca dar um passo. O servo realista deixa cada
  passo caro e ficar parado é um ótimo local;
- a partir da v4 (`--restore`), 2 M passos: 0,14 m/s a 0,15 pedido, 0,18 e 0,19 m/s a 0,25 e 0,30 sem cair
  (a v4 caía), pé a 22,5 mm, 0 % de pé no ar com comando zero (a v4: 40 %). Giro, ré e lado ainda fracos.

## Recuperar de empurrões: v6 e v7 (2026-10-31)

`push_recovery.py`: empurrão súbito (mudança de velocidade do corpo, como no treino) para frente / trás / lados,
de 0,2 a 0,6 m/s, parado e andando a 0,15 m/s; conta quantos de 40 o robô aguenta e quanto os pés se afastam.

| | v5 | v6 | v7 |
|---|---|---|---|
| Empurrões aguentados (de 40) | 20 | 20 | **21** |
| Parado e empurrado: distância entre os pés | 55 mm (não pisa) | 55 mm | **62–111 mm (pisa)** |
| Parado, para frente | cai a 0,4 | cai a 0,3 | **aguenta 0,4** |

- **v6** (30 M no Colab, a partir da v5): penalidades de "ficar parado" só com o corpo devagar, empurrões até
  0,6 m/s. Não aprendeu o passo de recuperação: com comando zero o relógio da marcha ficava parado e as
  recompensas de passo desligadas, então pisar nunca valia nada.
- **v7** (30 M no Colab, a partir da v5): um empurrão que a IMU sente (giro > 0,6 rad/s ou inclinação > 9°) liga o
  relógio da marcha e as recompensas de passo por 1 s, mesmo com comando zero. Agora ela pisa para se recuperar;
  ainda cai para trás a 0,3 m/s e com 0,5 m/s ou mais em qualquer direção, perto do limite do servo realista.
  Checkpoint em `checkpoints/cogu_walk_v7` (o download do Colab tinha chegado em `build/`; refeito o teste de
  empurrão aqui: 21/40, igual ao Colab). Vídeo: `docs/renders/sim_walk_v7.mp4`.

## Levantar (2026-10-31)

`cogu_getup.py` + `train.py --env getup` + modelo `cogu_lower_body_mjx_getup.xml` (bolha, tambor e pernas tocam
o chão) + `getup_eval.py` + `play.py --getup` (troca sozinho entre levantar e andar). Ainda não levanta:

- 1ª versão (ação na faixa inteira de cada junta): nem os episódios que começam em pé ficavam em pé;
- ação como desvio da pose de repouso, saída inicial ~0, ruído 0,3 (na verdade ignorado: ver v2 abaixo), currículo de inícios (em pé / inclinado /
  deitado), avaliação sem ruído: no início do treino 32 % do tempo em pé (quem começa em pé fica), mas o treino
  (11 M no Colab) desaprende — 0,9 %. Com o ruído do treino, quem está em pé cai do mesmo jeito, deitado é um
  estado estável que paga razoável, e o horizonte do desconto (0,97: ~0,7 s) é curto para uma manobra de
  segundos.

### Estudo de viabilidade (2026-10-31): com as pernas atuais, não

`getup_search.py` (CPU, ~60 ms por simulação, 9 núcleos) deita o robô de frente, de costas ou de lado e procura
com CMA-ES uma sequência de 3–9 poses (alvos em toda a faixa das juntas + duração de cada uma), avaliada em três
variações de atrito e força do servo. Resultados:

- **Busca aberta: zero levantadas.** 10 800 sequências aleatórias e todas as rodadas de CMA-ES (4 posições,
  3–9 poses, com e sem partida de sequências boas) terminam deitadas. O impulso mais forte que existe deixa a
  pelve, por um instante, a ~53–58° do vertical (up_z 0,52–0,60) e o robô cai de volta.
- **Análise quase estática** (para cada inclinação do corpo, existe pose das pernas em que o centro de massa
  fica sobre os apoios, sem passar do stall?): em pé só nos pés, ele se equilibra até ~50° inclinado para a frente
  ou ~60° para trás; apoiado em bolha + pés, só a partir de ~80°. **Entre ~60° e ~75° nenhuma combinação de apoios
  (pés, bolha, tambor, coxas ou canelas) segura o robô** — levantar exigiria arremessar o corpo por essa faixa e
  pará-lo em equilíbrio, e a busca não achou como. Torque não é o limite: as poses de apoio pedem 0,1–0,5 N·m.
- **A ideia de encolher as pernas e escorar na bolha** funciona até a metade: de costas, puxando as coxas para trás,
  ele ergue o quadril acima da bolha (até ~110°, bolha + pés, ~0,2 N·m) — vídeo `build/getup_search/back_propped_heel30.mp4`.
  De frente não dá: o quadril só vai 15° para a frente (a carcaça da coxa encosta na viga do quadril). A volta
  para o vertical passa pela faixa sem apoio.
- **O que fecha a faixa:** um **calcanhar** mais longo. Com a sola 30 mm mais comprida para trás, há pose só nos
  pés até 75° e bolha + pés a partir de 78° (de costas); 45 mm dá folga. Joelho até −25° ou tornozelo até ±80°
  (faixas que o CAD mede como livres) sozinhos não fecham. Mesmo com o calcanhar, a sequência aberta ainda não
  levanta: falta a troca de apoio (a bolha sai do chão quando o peso passa para os pés), que provavelmente pede
  realimentação (RL), não sequência fixa.

```bash
uv run python sim/playground/getup_search.py --pose back --keyframes 5 --generations 300
uv run python sim/playground/getup_search.py --objective peak            # o mais em pé que cada posição alcança
uv run mjpython sim/playground/getup_search.py --view build/getup_search/*_peak.json
```

### Calcanhar e impulso + equilíbrio (2026-10-31, segunda rodada)

`getup_static.py` mapeia, numa grade (inclinação × quadril × joelho × tornozelo, as duas pernas iguais), todas
as poses em que o robô fica parado sobre seus apoios, e procura caminhos entre poses vizinhas. A "janela de
pouso" é a maior inclinação em que só os pés seguram o robô e de onde um caminho lento leva a `rest`:

| Calcanhar (sola mais longa para trás) | 0 mm (atual) | 30 mm | 45 mm | 60 mm |
|---|---|---|---|---|
| janela de pouso (estática) | 55° | 65° | 75° | 80° |

No simulador dinâmico a janela é ~10° menor: com 60 mm, posto sobre os pés a 70° (ou 60°), o caminho lento em
rampa (0,5–2 s por 10°) o deixa em pé, pelve a 203 mm; a 80° ele cai de volta na bolha. **Nenhum calcanhar (até
90 mm) dá caminho lento a partir de deitado**: passar o peso da bolha para os pés pediria os pés atrás e acima do
quadril (~−120° de quadril; a junta vai a −85°). Então levantar = um impulso que pousa nos pés + um controle que
segura.

- O impulso existe: buscas com objetivo "em pé só nos pés" (`--objective land`) chegam a up_z 0,62 (52°) no
  robô atual; com o calcanhar de 60 mm o corpo passa por up_z 0,89 (27°) por um instante. Força sobra.
- Sequência fixa não segura (o momento certo varia com atrito e servo). Passar para a v7 no pouso
  (`getup_handoff.py`) também não: a v7 foi treinada até 60° de inclinação e nunca viu um pouso desses.

Próximo passo: uma política de levantar treinada com esses pousos como ponto de partida (currículo: começar sobre
os pés a 30–60°, depois do impulso, depois deitado), comparando o pé atual com o calcanhar de 45 mm antes de mudar
o pé no CAD.

```bash
uv run python sim/playground/getup_static.py --xml build/mjcf/variants/heel_45.xml --out build/getup_search/path_heel45.json
uv run python sim/playground/getup_search.py --pose back --objective land --keyframes 3
uv run python sim/playground/getup_handoff.py build/getup_search/back_land.json
```

(As variantes `build/mjcf/variants/heel_*.xml` são o MJCF de get-up com a caixa da sola alongada para trás,
geradas com `MjSpec`; o CAD do pé não mudou.)

### Política de levantar v2: pousos como ponto de partida (2026-10-31)

- **O bug que derrubava todos os treinos de levantar:** a política padrão do brax (`tanh_normal`) ignora
  `init_noise_std` e a inicialização da média — o ruído de exploração era ~0,7 rad, não 0,3. Com a ação zero
  (pose `rest`) mais ruído, o robô em pé ainda está de pé após 6 s em 100 % das vezes com 0,05, 84 % com 0,1, 3 %
  com 0,15 e 0 % com 0,3. Todo treino, mesmo só com partidas em pé, deitava em 0,2 M passos. Agora a política é
  `normal` com log-desvio aprendido a partir de 0,05 (a ação é cortada em ±1 no ambiente), entropia 1e-4, desconto
  0,99; em 0,4 M passos locais quem começa em pé continua 100 % em pé.
- **Partidas** (`start_mix`): em pé 10 %, inclinado em `rest` 10 %, **sobre os pés inclinado 35 %**, **no meio do
  impulso 20 %**, deitado após queda 25 %. As duas novas vêm de `getup_starts.py`: poses do mapa quase estático em
  que só os pés seguram e há caminho lento até `rest` (15°–55° de costas / 40° de frente no pé atual; até 75° com
  o calcanhar de 45 mm), e instantâneos das sequências de impulso de `getup_sequences/` quando a pelve passa de
  up_z 0,3. O treino mostra o % do tempo em pé por tipo de partida.
- Recompensa de altura agora `((h − 0,07)/(0,20 − 0,07))²` (peso 2): a linear pagava 65 % a um agachamento.
- O auto-reset do Playground não reinicia o `info`: o filtro e a fila de atraso agora recomeçam da pose inicial no
  primeiro passo de cada episódio.
- Calcanhar sem mexer no CAD: `variants.py --heel 45` (no notebook, `HEEL = 45`).

```bash
uv run python sim/playground/variants.py --heel 45
uv run python sim/playground/getup_starts.py --xml build/mjcf/variants/heel_45.xml
uv run python sim/playground/train.py --env getup --xml build/mjcf/variants/heel_45.xml --steps 2000000
uv run python sim/playground/getup_eval.py CKPT --xml build/mjcf/variants/heel_45.xml --starts feet throw drop
```

**Resultado (2026-11-01): não aprendeu a levantar.** 50 M passos no Colab para cada pé (T4, 4096 robôs, 74 min,
11,3 mil passos/s), do zero, `cogu_getup_v2` e `cogu_getup_v2_heel45` (só em `build/checkpoints/`, não
versionados). % do tempo em pé por partida, no treino:

| Partida | Pé atual, 0 → 50 M | Calcanhar 45 mm, 0 → 50 M |
|---|---|---|
| Em pé | 100 → 90 | 100 → 100 |
| Inclinado em `rest` | 9 → 1 | 9 → 1 |
| Sobre os pés inclinado | 33 → 4 | 25 → 30 (máx. 33) |
| Meio do impulso | 0 → 0 | 0 → 0 |
| Deitado após queda | 0 → 0 | 0 → 0 |

`getup_eval.py`, 20 tentativas no MuJoCo normal: sobre os pés 1 de 20 (pé atual) e 6 de 20 (calcanhar); do impulso
e deitado, 0 de 20 nos dois. O mesmo que os treinos locais de 2 M passos (5 e 3 de 20 sobre os pés).

- O conserto do ruído funcionou para ficar em pé (o pé atual ainda escorrega para 65–90 % no fim), mas nenhuma
  política ficou melhor que a ação zero: as partidas sobre os pés já davam 25–33 % no passo 0, ficando parado.
- Com ruído 0,05 a exploração não descobre o impulso + pegada; com ruído maior o robô em pé cai (tabela acima).
  Mais passos do mesmo PPO não resolvem: levantar precisa de uma referência de movimento.
- O calcanhar de 45 mm segura melhor as partidas sobre os pés (30 % vs 4 % no fim, 6 vs 1 de 20), coerente com a
  janela estática maior (até 75°). Não muda nada no impulso nem deitado.

### Força, pescoço, joelhos e pegada roteirizada (2026-11-01)

Antes de treinar por imitação, testes no MuJoCo normal do que limita o levantar. Variantes por `variants.py`
(`--servo`, `--neck`, `--knee-max`, além de `--heel`; o CAD não muda). Ver os treinos no viewer:
`uv run mjpython sim/playground/getup_eval.py CKPT --starts feet throw drop --view`.

- **Força não é o limite.** Busca de impulso de costas (`getup_search.py --objective land`, 3 poses, 120 gerações),
  o quanto o robô pousa em pé só sobre os pés (up_z): atual 0,56–0,58; servo ×1,5 0,56; ×2 0,61–0,64; pescoço
  ±60° com a massa do celular (250 g) 0,52–0,55; calcanhar 45 mm 0,57–0,58; calcanhar 45 + pescoço 0,65–0,69. Os
  impulsos salvos saturam algum servo em 30 % do tempo, mas servos mais fortes não pousam mais em pé.
- **Joelho até 130°** (o CAD permite 95°: as abas da canela batem na coxa) não muda o mapa estático: janela de
  pouso de 55° (pé atual) e 75° (calcanhar 45), sem apoio entre 60° e 75° no pé atual.
- **De joelhos não é atalho.** Separando o mapa estático em ilhas de poses ligadas: existem poses de joelhos estáveis
  (de bruços, canelas + pés, 5–35°, pelve até 190 mm), mas isoladas: de bruços o mais perto de deitado que segura é
  75–80° (tambor + pés), um salto de ~40° até a ilha de joelhos e de ~35° até em pé sobre os pés.
- **Levantar de costas pede pernas assimétricas.** Os melhores impulsos usam uma perna a −82° de quadril e a outra
  a −6°, com o hip roll a ±35°: o robô sai meio de lado, rolando na borda da bolha. Com as duas pernas iguais
  (`getup_reference.py --search`, impulso + pegada, 600 candidatos × 3 variações) o pico é up_z 0,24 e nada levanta.
  O mapa estático, que só considera pernas iguais, não vê esse caminho.
- **A pegada é o gargalo.** `getup_reference.py`: um controle roteirizado que, partindo das poses sobre os pés de
  `getup_starts`, lê a inclinação e anda devagar pelo caminho estático até `rest` (de costas e de bruços). Melhor
  ajuste (1 rad/s, 10° à frente): **17 %** em pé no pé atual, **40 %** com o calcanhar de 45 mm (ficar parado em
  `rest`: 15 % e 23 %). As poses "estáveis" do mapa têm 5 mm de margem: o pé tem 65 × 52 mm e o centro de massa está
  a 191 mm do chão. A base de apoio é pequena demais para pegar o pouso com folga.

### Estudo do pé (2026-11-01)

`foot_study.py`: para cada sola (calcanhar para trás, ponta para a frente, largura para fora, em mm; `variants.py`),
a janela de pouso estática, a pegada roteirizada a partir das poses sobre os pés daquela sola (80 pousos, nominal e
com piso escorregadio + servo fraco) e a marcha v7 **sem retreinar** (8 s, e os 40 empurrões de `push_recovery.py`).

| Pé (calcanhar, ponta, largura) | Sola (mm) | Janela costas / bruços | Pegada (costas / bruços) | Escorregadio + fraco | Marcha 0,15 / 0,25 / giro 8 s | Empurrões |
|---|---|---|---|---|---|---|
| atual | 65 × 52 | 55° / 35° | 25 % (31 / 18) | 30 % | 1,37 m / 1,78 m / 330° | 21/40 |
| 45, 0, 0 | 110 × 52 | 75° / 35° | 42 % (54 / 33) | 44 % | cai 2,1 s / cai 1,5 s / 302° | 14/40 |
| 30, 20, 0 | 115 × 52 | 65° / 45° | 39 % (52 / 24) | 41 % | 0,13 m / 0,24 m / −82° | 30/40 |
| 45, 20, 0 | 130 × 52 | 75° / 45° | 36 % (57 / 19) | 36 % | 0,08 m / cai 6,6 s / 229° | 30/40 |
| 60, 20, 0 | 145 × 52 | 80° / 45° | 39 % (52 / 24) | 40 % | 0,41 m / cai 4,1 s / 145° | 29/40 |
| 45, 40, 0 | 150 × 52 | 75° / 55° | 35 % (57 / 16) | 35 % | 0,85 m / 0,74 m / −18° | 30/40 |
| **45, 20, 15** | **130 × 67** | 75° / 45° | 36 % (57 / 19) | 36 % | 0,95 m / 0,33 m / 278° | **38/40** |
| 60, 40, 15 | 165 × 67 | 80° / 55° | 35 % (55 / 13) | 36 % | 0,59 m / 0,73 m / 225° | 37/40 |

- **A janela cresce com o pé:** de costas, de 55° para 75–80° com o calcanhar; de bruços, de 35° para 45–55° com
  a ponta.
- **A pegada roteirizada sobe de 25 % para 35–42 %** (de costas chega a 52–57 %). Cada sola é testada com os pousos
  dela, e um pé maior inclui pousos mais inclinados; por isso não sobe mais. Esse roteiro é cru (anda pelo caminho
  estático sem olhar velocidade): a pegada boa tem que vir do RL, e o pé maior dá a ela mais margem.
- **Empurrões:** a sola mais larga segura 37–38 de 40 com a mesma política v7 (atual: 21).
- **A marcha precisa ser retreinada para qualquer pé novo:** a v7 tropeça com a sola mais longa (só calcanhar: cai
  em 2 s). Isso mede a política, não o pé.

**Marcha v8 no pé 130 × 67 mm (45, 20, 15).** `checkpoints/cogu_walk_v8_foot/`: 30 M passos no Colab (T4, 27 min) a partir da
v7, no modelo de marcha com essa sola (`walk_heel45_toe20_width15.xml`). Sim2sim, 8 s a partir do repouso:

| Medida | v7 no pé atual | v7 no pé novo | **v8 no pé novo** |
|---|---|---|---|
| 0,15 m/s: distância / velocidade | 1,37 m / 0,18 | 0,95 m / 0,12 | 1,23 m / **0,16** |
| 0,25 / 0,30 m/s | 0,23 / 0,24 | 0,14 / 0,15 | 0,22 / 0,23 |
| ré 0,10 m/s | 0,09 | 0,02 | 0,09 |
| de lado ±0,10 m/s | 0,03 / 0,04 | 0,02 / 0,04 | 0,03 / 0,03 |
| giro ±0,6 / 1,0 rad/s | 0,56 / −0,57 / 0,86 | 0,47 / −0,34 / 0,78 | **0,58 / −0,60 / 0,87** |
| giro 8 s a 0,8 rad/s | 330° | 278° | **348°** |
| empurrões | 21/40 | 38/40 | **37/40** |
| parado: pé no ar | 0 % | 0 % | **46 %** |
| marcha a 0,15: pé / no ar / escorregão | 27 mm / 0,28 s / 0,010 | 13 mm / 0,06 s / 0,125 | 23 mm / 0,27 s / 0,013 |

- **O pé novo anda tão bem quanto o atual** (velocidades e giro iguais ou melhores, a deriva lateral da v7 sumiu) e
  **segura quase o dobro de empurrões** (37 contra 21).
- **Defeito novo: parado, ela marcha no lugar** (pé no ar 46 % do tempo com comando zero; a v7 fica de pé). Também
  a frequência medida pelo `gait_metrics.py` subiu para 3,6 Hz (mesmos 23 passos em 8 s; tremor 0,02).

**Por que a v8 marchava no lugar.** Desde a v7, um empurrão que o IMU sente (giro > 0,6 rad/s ou 9° de inclinação)
liga o relógio da marcha e as recompensas de passo por 1 s, mesmo com comando zero. A v8 aprendeu a se empurrar: com
comando zero ela balança o corpo nos primeiros 40 ms, o giro passa de 0,6 rad/s e o modo de recuperação fica ligado
para sempre (100 % do tempo no sim2sim, pé no ar 88 %; a v7 não passa de 0,34 rad/s parada). Correção no env
(v9): o relógio continua seguindo o IMU, como no robô, mas as recompensas de passo e a isenção de `stand_still` /
`feet_air_still` só valem até 2 s depois de um empurrão de verdade (`recover.push_credit`). Também `stand_still`
−1 → −2, `feet_air_still` −1 → −3 e comando zero em 20 % dos sorteios (era 10 %). Retreinar a v8 com isso não
desfez o vício em 1 M passos (local); partindo da v7 ele não aparece.

**Marcha v9 no pé novo.** `checkpoints/cogu_walk_v9_foot/`: 30 M passos no Colab (T4) a partir da v7, com o env
corrigido. Vídeo: `docs/renders/sim_walk_v9.mp4`. Mesmas medidas da tabela acima:

| Medida | v7 no pé atual | v8 no pé novo | **v9 no pé novo** |
|---|---|---|---|
| 0,15 m/s: distância / velocidade | 1,37 m / 0,18 | 1,23 m / 0,16 | **1,27 m / 0,16** |
| 0,25 / 0,30 m/s | 0,23 / 0,24 | 0,22 / 0,23 | 0,22 / 0,24 |
| ré 0,10 m/s | 0,09 | 0,09 | **0,11** |
| de lado ±0,10 m/s | 0,03 / 0,04 | 0,03 / 0,03 | 0,03 / 0,04 |
| giro ±0,6 / 1,0 rad/s | 0,56 / −0,57 / 0,86 | 0,58 / −0,60 / 0,87 | **0,64 / −0,61 / 0,97** |
| giro 8 s a 0,8 rad/s | 330° | 348° | **374°** |
| empurrões | 21/40 | 37/40 | **37/40** |
| parado: pé no ar | 0 % | 46 % | **0 %** |
| marcha a 0,15: pé / no ar / escorregão | 27 mm / 0,28 s / 0,010 | 23 mm / 0,27 s / 0,013 | 23 mm / 0,27 s / 0,016 |

A v9 fica parada como a v7, anda e gira tão bem ou melhor que a v8 e segura os mesmos 37 de 40 empurrões. O
`gait_metrics.py` ainda mede 3,6 Hz no pé novo (23 passos em 8 s, como a v7 no pé atual a 1,7 Hz): provavelmente a medida, não o
passo. Andar de lado continua fraco (um terço do pedido).

**O pé novo foi para o CAD** (`cad/parameters.py`: `FOOT_LENGTH` 130, `FOOT_WIDTH` 67, `FOOT_OUTSET` 7,5,
`ANKLE_X_FROM_HEEL` 67). A sola exportada é a mesma caixa da variante `walk_heel45_toe20_width15.xml`; o pé passou
de 15 para 34 g impressos e o robô de 1282 para 1320 g. A v9 no modelo do CAD: 1,27 m / 1,75 m / 376°, 38 de 40
empurrões. `variants.py` e `foot_study.py` agora somam ao pé novo (a coluna "atual" da tabela acima é o pé antigo).

**Marchando depois do empurrão.** Parada, um empurrão de 0,4 m/s ou mais deixava a v9 (e a v7) marchando no lugar para
sempre: os passos de recuperação mantêm o giro acima de 0,6 rad/s, cada leitura renovava a janela de 1 s e o relógio
da marcha nunca parava. Detector novo (`recover.latch`, v10): cada empurrão abre **uma** janela de 1,5 s que os passos
não renovam, e depois o detector descansa 1 s. A v9 com ele, sem retreino (`play.py --latch`): para de dar passos
1,6–1,7 s depois de qualquer empurrão parado (0,2 a 0,6 m/s, quatro direções) e continua segurando 38 de 40. Os
checkpoints até a v9 seguem com o detector antigo, a não ser com `--latch`; o env já treina com o novo.

**Marcha v10: pé do CAD e detector novo.** `checkpoints/cogu_walk_v10/`: 30 M passos no Colab (T4) a partir da v9, no
modelo exportado do CAD (pé de 34 g, robô de 1320 g), com `recover.latch`. Vídeo: `docs/renders/sim_walk_v10.mp4`.
Sim2sim no modelo do CAD, 8 s a partir do repouso:

| Medida | v9 (+ `--latch`) | **v10** |
|---|---|---|
| 0,15 m/s: distância / velocidade | 1,27 m / 0,16 | **1,30 m / 0,17** |
| 0,25 m/s em 8 s / 0,30 m/s | 1,75 m / 0,24 | **1,80 m / 0,25** |
| ré 0,10 m/s | 0,11 | 0,10 |
| de lado ±0,10 m/s | 0,03 / 0,04 | 0,03 / 0,02 |
| giro ±0,6 / 1,0 rad/s | 0,64 / −0,61 / 0,97 | 0,63 / −0,60 / 0,93 |
| giro 8 s a 0,8 rad/s | 376° | 369° |
| empurrões | 38/40 | **38/40** |
| parado: pé no ar | 0 % | 0 % |
| parado, empurrão 0,4–0,6 m/s: para de dar passos | 1,6–1,7 s | 1,6 s (cai só a 0,6 para a frente) |
| marcha a 0,15: pé / no ar / escorregão | 23 mm / 0,27 s / 0,016 | 22 mm / 0,27 s / **0,007** |

A v10 é a marcha atual: igual à v9 com `--latch`, um pouco mais rápida, escorrega metade, e o detector com que ela
treinou é o que ela usa. Andar de lado continua fraco.

## Resultado v5 — servo realista, ficar parado, giro simétrico (2026-10-30)

`checkpoints/cogu_walk_v5/`: 30 M passos no Colab (T4, 4096 robôs, 26,5 min, 18,9 mil passos/s), partindo da v4,
no robô com o iPhone 15 Pro Max (1,28 kg) e o modelo de servo realista. Vídeo: `docs/renders/sim_walk_v5.mp4`.

`command_tracking.py` no MuJoCo normal (sim2sim), 8 s a partir da pose de repouso, sem quedas em nenhum comando:

| Comando | v4 (modelo novo) | **v5** |
|---|---|---|
| 0,15 m/s | 0,13 | **0,16** |
| 0,25 m/s | cai | **0,21** |
| 0,30 m/s | cai | **0,22** |
| ré 0,10 m/s | 0,02 | **0,07** |
| de lado 0,10 m/s | 0,02 / 0,01 | 0,03 / 0,03 |
| giro +0,6 / −0,6 rad/s | 0,42 / −0,30 | **0,56 / −0,59** |
| giro 1,0 rad/s | 0,54 | **0,75** |
| parado: pé no ar | 40 % | **0 %** |

`gait_metrics.py` a 0,15 m/s: 1,71 Hz, tremor 0,03, pé a 25 mm (26 / 24 mm), 46 / 49 % no ar, escorregão 0,010 m/s.
O giro ficou simétrico (o desvio para a esquerda sumiu) e a ré dobrou; andar de lado continua fraco (um terço do
pedido) — é o próximo ajuste da marcha.

## Resultado v4 — primeira marcha de verdade (2026-10-30)

`checkpoints/cogu_walk_v4/`: 50 M passos no Colab (GPU T4, 4096 robôs, 37 min, 22,7 mil passos/s), seed 0, no robô
com celular (ADR-002). Recompensas novas: filtro passa-baixa na ação (`action_filter_alpha` 0,4), cobranças de
suavidade, de pé escorregando, de pé baixo e de pé plantado, e mais peso para fase e tempo de voo. As versões
v0/v1 não davam passos: vibravam as pernas a 12,6 Hz e patinavam.

`gait_metrics.py` a 0,15 m/s, 8 s a partir da pose de repouso, no MuJoCo normal (sim2sim):

| Medida | v1 | **v4** | Uma marcha |
|---|---|---|---|
| Velocidade | 0,16 m/s | 0,16 m/s | 0,15 pedido |
| Frequência do passo | 12,6 Hz | **1,71 Hz** | 1,5–2,5 Hz |
| Tremor (potência > 5 Hz) | 0,99 | **0,10** | < 0,2 |
| Movimento das juntas | 6° | **20°** | 15–40° |
| Altura do pé | 4 mm | **26 mm** | ≥ 20 mm |
| Tempo no ar, esq / dir | — | **43 / 45 %** | 30–50 % |
| Pé escorregando | 0,085 m/s | **0,014 m/s** | < 0,03 |

Sim2sim do notebook: 1,22 m em 8 s a 0,15 m/s, 1,60 m a 0,25 m/s, 1,08 m com ruído e 20 ms de atraso, parado com
comando zero; não cai em nenhum. Treinada sem a bolha, a mesma política anda igual no robô atual com a bolha de
PCTG e as capas de TPU (1,24 kg): a randomização de massa da pelve cobriu o peso novo. Girando (ω = 0,8 rad/s)
ela ainda recua um pouco; giro e marcha de lado são os próximos ajustes. Vídeo: `docs/renders/sim_walk_v4.mp4`.

## Resultado v1 — robô completo (2026-10-26)

`checkpoints/cogu_walk_v1/`: mesmos hiperparâmetros do v0, 10 M passos, seed 0, sobre o MJCF do commit
16fd71b (corpo + colar + chapéu: 1,31 kg, CoM alto, IMU na posição da BNO055). O processo morreu sem erro
a 4,5 M passos (pressão de memória na máquina) e foi retomado com `--restore` do checkpoint de 4,5 M;
`progress.jsonl` concatena as duas metades. Rollout determinístico a partir de `rest`, 8 s:

| Comando | v1 (robô completo) | v0 (só pernas) |
|---|---|---|
| `vx = 0,15 m/s` | 0,90 m (0,11 m/s), não cai — [vídeo](../../docs/renders/sim_walk_v1.mp4) | 0,94 m |
| `vx = 0,25 m/s` | 1,32 m (0,17 m/s), não cai | 1,53 m |
| `vx = 0` | parado (2 cm de deriva) | parado |
| `vx = 0,1, ω_z = 0,8 rad/s` | gira sem cair | gira |

`compare.py` no modelo completo (32 episódios de 10 s com empurrões e comandos aleatórios):

| Checkpoint | 8 s a 0,15 m/s | quedas | duração média |
|---|---|---|---|
| v0 no robô completo | cai em 0,8 s | 100 % | 1,4 s |
| **v1** | 0,97 m, não cai | **9 %** | 9,8 s |

Os servos (kp 8, ±1,9 N·m) seguram o tronco de 690 g, mas com pouca folga: andando a 0,15 m/s os tornozelos
encostam no limite de 1,9 N·m em 1,3 % dos passos e os joelhos chegam a 1,7 N·m (RMS geral 0,6 N·m). O STS3215
real dá ~1,9 N·m só em stall; vale confirmar na identificação do atuador (BAM) antes de confiar nisso. A marcha
continua curta e arrastada; a recompensa final (15,6) e a duração (8,9 s de 10 s) são um pouco melhores que
as do v0 por causa do IMU mais alto, que dá um sinal de inclinação mais limpo.

```bash
uv run python sim/playground/rollout.py --checkpoint sim/playground/checkpoints/cogu_walk_v1
uv run python sim/playground/compare.py sim/playground/checkpoints/cogu_walk_v0 sim/playground/checkpoints/cogu_walk_v1
```

## Resultado v0 — só pernas (2026-10-25)

`checkpoints/cogu_walk_v0/`: 10 M passos de PPO, seed 0, 54 min no M1 Max (CPU, 1024 envs, 3,1 k passos/s),
sobre o MJCF do commit b3df239. Rollout determinístico a partir do keyframe `rest`, 8 s:

| Comando | Resultado |
|---|---|
| `vx = 0,15 m/s` | anda 0,94 m (0,12 m/s médio), não cai — [vídeo](../../docs/renders/sim_walk_v0.mp4) |
| `vx = 0,25 m/s` | 1,53 m (0,19 m/s), não cai |
| `vx = 0` | fica parado (1 cm de deriva) |
| `vx = 0,1, ω_z = 0,8 rad/s` | gira e avança, não cai |

Na avaliação com comandos aleatórios + randomização + empurrões a duração média do episódio é 8,6 s de 10 s
(recompensa 14,1): a marcha é um arrastar de pés curto, ainda cai às vezes. É a prova do pipeline, não a
política final — ver "Como escalar".

```bash
uv run python sim/playground/rollout.py --checkpoint sim/playground/checkpoints/cogu_walk_v0
```

## Treinar em GPU no Google Colab

A GPU do Mac não serve: testado em 2026-10-30, o plugin Metal do JAX enxerga a GPU mas não implementa
uma operação que o MJX usa em todo passo (`mhlo.cholesky`), e está parado desde 2024 numa versão antiga
do JAX. O caminho rápido é uma GPU NVIDIA na nuvem.

[Abrir o notebook no Colab](https://colab.research.google.com/github/laroccathebrux/cogu/blob/main/sim/playground/colab_train.ipynb)
→ *Ambiente de execução → Alterar o tipo → GPU (T4)* → *Executar tudo*. O notebook clona este
repositório, instala o JAX com CUDA 12 da mesma versão do projeto, gera o MJCF a partir do CAD, treina
(padrão: 50 M passos, 4096 robôs), roda o teste sim2sim, grava um vídeo e baixa um zip. No Mac:

```bash
unzip ~/Downloads/cogu_walk_gpu.zip -d build/checkpoints/
uv run mjpython sim/playground/play.py --checkpoint build/checkpoints/cogu_walk_gpu
```

## Ver a política no viewer (sim2sim) e acompanhar o treino

`play.py` roda a política no MuJoCo "normal" (o motor em C, CPU), não no MJX onde ela foi treinada. Se
o robô anda aqui, a política não depende de nada específico do MJX — é o mesmo teste sim2sim que o Open
Duck faz antes do hardware. A observação, a escala da ação, a fase da marcha e a ordem exata em que o
ambiente monta tudo isso são refeitas sem JAX.

```bash
uv run mjpython sim/playground/play.py --checkpoint sim/playground/checkpoints/cogu_walk_v1
uv run mjpython sim/playground/play.py --follow build/checkpoints/cogu_walk_v2   # acompanha um treino
uv run python sim/playground/play.py --headless --vx 0.15 --noise 1 --delay 1 \
    --checkpoint sim/playground/checkpoints/cogu_walk_v1                           # sem janela
```

Teclas: ↑/↓ velocidade para a frente, ←/→ girar, PgUp/PgDn de lado, End para, Home volta à pose de
repouso; Ctrl + arrastar com o botão direito empurra o robô. `--follow DIR` troca de política sozinho
sempre que o `train.py` salva um checkpoint novo em DIR (uma vez por avaliação), então dá para ver o
robô aprendendo enquanto o treino roda. O treino em si não tem janela: são 1024 robôs em paralelo no JAX.

**macOS:** o viewer controlado por script precisa do `mjpython`. Com o Python gerenciado pelo uv, o
`mjpython` não acha a `libpython`; crie o link uma vez:

```bash
ln -sf "$(uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')/libpython3.12.dylib" .venv/libpython3.12.dylib
```

Primeiro teste sim2sim (2026-10-30): a política **v1**, treinada no robô-cogumelo, roda no MuJoCo normal
sobre o **modelo com celular** e anda 1,21 m em 8 s a 0,15 m/s; com ruído de observação e 20 ms de
atraso, 1,04 m. Parada com comando zero, 1 cm de deriva. O modelo novo é mais leve (1,13 kg) e a v1 já
tinha sido treinada com carga variável na pelve.

## Uso

```bash
uv run python -m cad.export --out build          # gera build/mjcf/cogu_lower_body_mjx.xml
uv run python sim/playground/check_env.py        # ~30 s: env OK?
uv run python sim/playground/train.py --steps 200000 --num-evals 2    # smoke test (~4 min no M1 Max)
uv run python sim/playground/train.py --steps 10000000 --num-evals 10  # ~1 h no M1 Max (CPU)
uv run python sim/playground/rollout.py --vx 0.15 --seconds 8          # build/videos/walk.mp4
uv run python sim/playground/rollout.py --vx 0.1 --wz 0.8 --out build/videos/turn.mp4
```

`train.py` imprime, a cada avaliação, recompensa média, duração média do episódio (500 = 10 s sem cair)
e velocidade média para frente. Throughput no M1 Max (CPU, 1024 envs): ~4 k passos de env/s no
rollout puro, ~3,1 k passos/s no treino completo. 128 envs de avaliação × 500 passos por avaliação.

## O env em uma página

- **Controle**: 50 Hz (`ctrl_dt` 0,02 s) sobre física a 500 Hz (10 substeps). Ação = offset em radianos
  (× `action_scale` 0,4) sobre a pose `rest`, enviado aos 10 atuadores `position` (kp 8, kv 0,3, ±1,9 N·m,
  do MJCF). O alvo passa por um buffer com **atraso de 0–2 passos (0–40 ms)** sorteado por episódio.
- **Observação `state` (43)** — o que o robô real terá: gyro (3), direção da gravidade no frame da pelve (3),
  comando (3), posição das juntas − rest (10), velocidade das juntas (10), última ação (10), fase da
  marcha cos/sin (4). Ruído uniforme em tudo que vem de sensor.
- **Observação `privileged_state` (100)** — só para o crítico: `state` + leituras sem ruído, velocidade
  linear, altura da pelve, torques, contato dos pés, velocidade dos pés, tempo no ar, atraso sorteado.
- **Recompensa** (escalas em `default_config().reward_config.scales`): tracking de velocidade linear
  (exp(−erro²/0,05)) e angular; penalidades de velocidade vertical, rotação xy, inclinação, altura da
  pelve (alvo 0,20 m), torque, variação de ação, escorregamento dos pés, limites de junta, pose,
  "ficar parado" quando o comando é zero, queda (−1); recompensas de tempo de voo dos pés (0,1–0,35 s) e
  de **fase da marcha** (altura do pé seguindo uma curva alvo com ápice de 2,5 cm, a 1,5–2 Hz) — é o que
  induz o passo alternado, como no Berkeley Humanoid.
- **Terminação**: pelve inclinada > 60° ou a menos de 10 cm do chão, ou NaN.
- **Comandos**: `vx ∈ [−0,10; 0,30] m/s`, `vy ∈ [−0,10; 0,10]`, `ω_z ∈ [−1; 1] rad/s`, 10 % dos episódios com
  comando zero, re-sorteio a cada 5 s. **Empurrões** de 0,05–0,3 m/s a cada 3–6 s.
- **Domain randomization** (`randomize.py`): atrito do chão U(0,4; 1,0), massas ×U(0,9; 1,1), **+(−20 g … +100 g)
  na pelve** (corpo, bateria e chapéu ainda não existem no CAD), atrito seco ×U(0,8; 1,2), armature ×U(1; 1,05),
  kp e kv dos servos ×U(0,8; 1,2) independentes, offset da pose de repouso ±0,03 rad.

### MJCF: variante `_mjx`

`cad/export.py` gera dois arquivos com a mesma cinemática, massas, inércias e atuadores:

- `cogu_lower_body.xml` — todas as peças são meshes de colisão (contatos fiéis; `check_mjcf.py`, viewer);
- `cogu_lower_body_mjx.xml` — meshes só visuais; a única colisão é **uma caixa por sola** (sola–chão e
  sola–sola; nenhuma outra autocolisão), `iterations=3`, `ls_iterations=5`, `eulerdamp` desligado. É o padrão dos envs do
  Playground ("feet only"): MJX fica ~10× mais rápido e o treino não precisa dos contatos entre peças.

Os dois têm o site `imu` na origem da pelve, sites `L_foot_site`/`R_foot_site` no centro da sola, sensores
(`gyro`, `accelerometer`, `local_linvel`, `upvector`, `global_linvel`, `global_angvel`, posição/velocidade
dos pés, contato pé–chão) e a câmera `track`.

## Como escalar (GPU / Hugging Face Jobs)

No CPU o treino de 10 M passos leva ~1 h; uma política boa precisa de 50–200 M passos. Em GPU o mesmo
código roda com `impl="warp"` ou `impl="jax"` e milhares de envs:

```bash
# instalar com CUDA (Linux): jax[cuda12] no lugar do jax de CPU
uv pip install "jax[cuda12]==0.11.2"
uv run python sim/playground/train.py --steps 100000000 --num-envs 4096 --batch-size 1024 \
    --num-minibatches 4 --num-evals 20 --out build/checkpoints/cogu_walk_gpu
```

Hugging Face Jobs (`pip install -U huggingface_hub`; precisa de token com permissão de jobs):

```bash
hf jobs run --flavor a10g-small --timeout 4h \
    -e HF_TOKEN=$HF_TOKEN \
    ghcr.io/astral-sh/uv:python3.12-bookworm \
    bash -c "git clone https://github.com/laroccathebrux/cogu.git && cd cogu && \
             uv sync --frozen && uv pip install 'jax[cuda12]==0.11.2' && \
             uv run python -m cad.export --no-step && \
             uv run python sim/playground/train.py --steps 100000000 --num-envs 4096 --batch-size 1024 && \
             uv run python sim/playground/rollout.py && \
             uv run python -c \"from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='build/checkpoints/cogu_walk', repo_id='laroccathebrux/cogu-walk', repo_type='model')\""
```

(O comando acima publica o checkpoint em um repo de modelo no Hub; `rollout.py` aceita `--checkpoint`
apontando para a pasta baixada.) Para deploy no Pi o próximo passo é exportar a rede `state → ação` para
ONNX, como no `Open_Duck_Mini_Runtime`.

## Decisões revisáveis

- Rede: política (128, 128, 128), crítico (256, 256, 256), `tanh_normal`, observações normalizadas.
- `action_scale` 0,4 rad; atraso máximo 2 passos; `tracking_sigma` 0,05; ápice do pé 2,5 cm; altura alvo 0,20 m.
- Terminar em 60° de inclinação (e não só "de cabeça para baixo" como o Berkeley): o Cogu é baixo e cai rápido.
- No MJX só as solas colidem (entre si e com o chão); coxas/canelas não — o CAD garante 10 mm de folga
  entre as pernas em repouso (`LEG_GAP`), mas poses extremas de hip roll/yaw podem cruzar as pernas sem
  penalidade na sim.
- A pelve recebe +(−20…+100) g de carga aleatória como proxy do corpo/chapéu até eles existirem no CAD.
