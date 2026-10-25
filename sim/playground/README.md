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
| `randomize.py` | Domain randomization por env: atrito do chão, massas, carga extra na pelve, atrito/armature das juntas, kp/kv dos servos, offset de pose |
| `train.py` | PPO (Brax) com seed fixa; salva checkpoints em `build/checkpoints/cogu_walk/<passo>/` e `progress.jsonl` |
| `rollout.py` | Roda a política a partir do keyframe `rest` com um comando fixo e grava `build/videos/walk.mp4` |
| `check_env.py` | Smoke test do env (CI): fica em pé 2 s, observações finitas, randomizer em lote |
| `jax_compat.py` | Shim: Brax 0.14.2 ainda chama `jax.device_put_replicated`, removido do namespace público no jax 0.10 |

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
rollout puro, ~2–3 k passos/s no treino completo. 128 envs de avaliação × 500 passos por avaliação.

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
- `cogu_lower_body_mjx.xml` — meshes só visuais; a única colisão é **uma caixa por sola contra o chão**
  (sem autocolisão), `iterations=3`, `ls_iterations=5`, `eulerdamp` desligado. É o padrão dos envs do
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
- Sem colisão pé–pé no MJX. **Atenção**: na pose de repouso as solas se sobrepõem 12 mm em Y
  (`FOOT_WIDTH` 52 mm, centros das solas a ±20 mm) — isso é um problema de CAD a resolver antes de imprimir.
- A pelve recebe +(−20…+100) g de carga aleatória como proxy do corpo/chapéu até eles existirem no CAD.
