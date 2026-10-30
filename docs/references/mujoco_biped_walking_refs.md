# Treinar o bípede (com celular acoplado) para andar naturalmente no MuJoCo — referências

Contexto para a sessão do Claude Code: bípede 3D-printed estilo Open Duck Mini / BDX, servos de hobby (Feetech STS3215 ou similar), estrutura 100% impressa, **com um smartphone fixado ao corpo** (payload pesado e alto; potencialmente o "cérebro" + IMU + câmera). Objetivo: política de caminhada natural/expressiva treinada em MuJoCo (MJX ou MuJoCo Warp) e transferida ao robô real.

---

## 1. Comece por aqui (pipeline completo, mesmo tipo de robô)

| Link | Por que importa |
|---|---|
| https://github.com/apirrone/Open_Duck_Playground | Framework de RL em **MuJoCo Playground/MJX** para o Open Duck Mini v2. Tem `joystick.py` (rewards), `randomize.py` (domain randomization), export para ONNX, `mujoco_infer.py`, e um guia "Adding a new robot" — copiar a pasta `open_duck_mini_v2` como ponto de partida é o caminho mais curto. Modela **backlash** dos servos no MJCF. |
| https://github.com/apirrone/Open_Duck_reference_motion_generator | Gera movimentos de referência (gait paramétrico via Placo/IK) → `polynomial_coefficients.pkl` usado como **imitation reward** no Playground. É isso que dá o "andar de patinho" natural em vez do gait robótico que o PPO puro acha. |
| https://github.com/apirrone/Open_Duck_Mini | Repo-hub (branch `v2`): CAD, BOM, guias, `BEST_WALK_ONNX_2.onnx`, links para runtime. |
| https://github.com/apirrone/Open_Duck_Mini_Runtime | Runtime no robô real (Raspberry Pi Zero 2W): lê IMU, roda ONNX, manda posições aos servos a 50 Hz. Referência de como fechar o loop sim→real. |
| https://deepwiki.com/apirrone/Open_Duck_Playground | Explicação navegável da arquitetura do Playground (rewards, randomization, export). |
| https://dev.to/frankfu/understanding-reinforcement-learning-through-openduck-1if0 | Walkthrough completo de alguém que replicou o Open Duck: fluxo modelo → sim → RL → hardware → deploy, com FAQ. |
| https://la.disneyresearch.com/wp-content/uploads/BD_X_paper.pdf | Paper original do BD-X (Disney) — é a referência que o Open Duck segue para o imitation reward. |

---

## 2. Papers-chave sobre caminhada *natural* (não só estável)

### Imitação de referência / estilo
- **Design and Control of a Bipedal Robotic Character** (Disney Research, RSS 2024) — https://arxiv.org/abs/2501.05204 — Como a Disney treina os droids BDX: reward = imitação + regularização + sobrevivência; políticas separadas para movimentos perpétuos/periódicos/episódicos; referência de animação gerada e amostrada densamente. É *o* paper para "andar com personalidade".
- **DeepMimic** (Peng et al., 2018) — https://arxiv.org/abs/1804.02717 — Formulação clássica de reward de imitação (pose, velocidade, end-effector, root).
- **Learning Agile Robotic Locomotion Skills by Imitating Animals** (Peng et al., 2020) — https://arxiv.org/abs/2004.00784 — Pesos de reward concretos (w_p=0.5, w_v=0.05, w_e=0.2, w_rp=0.15, w_rv=0.1) e sim-to-real com imitação de mocap.
- **AMP: Adversarial Motion Priors** (Peng et al., 2021) — https://arxiv.org/abs/2104.02180 — Discriminador aprende um "style reward" a partir de clipes de referência; elimina reward engineering pesado.
- **AMP Make Good Substitutes for Complex Reward Functions** (Escontrela et al., IROS 2022) — https://arxiv.org/abs/2203.15103 — AMP em robô real; gaits mais naturais e energeticamente eficientes. Código: https://github.com/Alescontrela/AMP_for_hardware
- **AMP in the Wild** (Wang et al., 2023) — https://arxiv.org/abs/2304.10888 — AMP + teacher-student para robustez sim-to-real.
- **Learning Bipedal Walking on a Quadruped via AMP** (2024) — https://arxiv.org/abs/2407.02282 — AMP aplicado a gait bípede; boa revisão de literatura na introdução.
- **AMP + sim-to-real em plataforma bípede não convencional** (2025) — https://arxiv.org/abs/2509.05581 — Retargeting de mocap humano + domain randomization + rewards especializadas.
- **Feature-based vs GAN-based Learning from Demonstrations** (survey, 2025) — https://arxiv.org/abs/2507.05906 — Quando usar imitação por tracking explícito (DeepMimic) vs. adversarial (AMP).

### Sem referência de movimento (reward periódico)
- **Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition** (Siekmann et al., ICRA 2021) — https://arxiv.org/abs/2011.01387 — Descreve walking/running/hopping/skipping com rewards periódicos sobre forças de contato e velocidades dos pés, usando uma variável de fase cíclica. Alternativa ao imitation reward quando não há referência boa. Versão HTML: https://ar5iv.labs.arxiv.org/html/2011.01387
- **Adaptive Mimic** — https://arxiv.org/abs/2112.03735 — Adapta automaticamente o peso imitação vs. desempenho durante o treino (útil quando a referência é imperfeita, caso típico de gait gerado por IK).

### Plataformas open-source comparáveis (hardware 3D-printed + RL em sim)
- **ToddlerBot** (Stanford, CoRL 2025) — https://arxiv.org/abs/2502.00893 / https://toddlerbot.github.io — Humanoide totalmente impresso; pipeline de **system identification de motor** transferível e calibração de zero que dá digital twin fiel e sim-to-real zero-shot. Vale copiar o sysID.
- **Berkeley Humanoid Lite** (RSS 2025) — https://arxiv.org/abs/2504.17249 / https://lite.berkeley-humanoid.org — Open-source, atuadores com redução impressa, controlador de locomoção RL com transfer zero-shot.
- **PhoneBot** (UCLA RoMeLa, 2026) — https://arxiv.org/abs/2610.08737 — **Bípede humanoide de 13 motores que usa o smartphone como torso, IMU e computador** (política roda no Android a 50 Hz; SBC só faz ponte com o barramento de servos). Trata o telefone como corpo inercial principal, não como payload passivo; randomiza ganhos, limites de torque, atrito; usa asymmetric actor-critic e data augmentation por espelhamento. **É o paper mais próximo do nosso caso.**
- **MOBIUS** (2025) — https://arxiv.org/abs/2511.01774 — Apêndice com tabela completa de hiperparâmetros MuJoCo (timestep, iterações de solver, rede) e ranges de domain randomization para bípede.

---

## 3. Ferramentas de simulação / treino

- **MuJoCo Playground** — https://github.com/google-deepmind/mujoco_playground · site https://playground.mujoco.org · paper https://arxiv.org/abs/2502.08844 — Ambientes de locomoção em MJX (e agora MuJoCo Warp), PPO via Brax, domain randomization, curriculum. Treina humanoide pequeno em <15 min em 2×RTX 4090. Colab de locomoção: https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb
- **mjlab** (Zakka et al., 2026) — https://github.com/mujocolab/mjlab · docs https://mujocolab.github.io/mjlab · paper https://arxiv.org/abs/2601.22074 — API manager-based do Isaac Lab sobre **MuJoCo Warp**; vem com velocity tracking e **motion imitation** prontos; `uv run play --agent zero/random` para sanity check do MDP antes de treinar. Requer GPU NVIDIA (no Mac só avaliação).
- **MJX docs** — https://mujoco.readthedocs.io/en/stable/mjx.html — Limitações do MJX (contatos, solver) que afetam modelagem.
- **MuJoCo modeling guide** — https://mujoco.readthedocs.io/en/stable/modeling.html — Inércias, `<actuator position>` com `kp`, `frictionloss`, `armature`, `damping`, limites.
- **MuJoCo tutorial Colab** — https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb — Fundamentos: contatos, atuadores, sensores.
- **MuJoCo Menagerie** — https://github.com/google-deepmind/mujoco_menagerie — Modelos MJCF de referência (Berkeley Humanoid, Unitree G1, Booster T1, etc.) para copiar convenções de atuador/sensor/colisão.
- **LocoMuJoCo** — https://github.com/robfiras/loco-mujoco · https://arxiv.org/abs/2311.02496 — Benchmark de imitation learning para locomoção com datasets de mocap, baselines (incl. GAIL/AMP-like) e domain randomization embutida.
- **legged_gym / rsl_rl** (ETH) — https://github.com/leggedrobotics/legged_gym · https://github.com/leggedrobotics/rsl_rl · paper https://arxiv.org/abs/2109.11978 — Mesmo sendo Isaac Gym, é a referência de estrutura de rewards de locomoção (tracking lin/ang vel, feet air time, action rate, torques, orientation) que o Playground e mjlab copiam.
- **ProtoMotions — Domain Randomization & Sim2Sim** — https://nvlabs.github.io/ProtoMotions/tutorials/workflows/domain_randomization.html — Dicas práticas, incluindo como o MuJoCo combina atrito (max dos dois geoms) e por que testar sim2sim no MuJoCo antes do real.

---

## 4. Do CAD para o MJCF

- **onshape-to-robot** (Rhoban) — https://onshape-to-robot.readthedocs.io · https://github.com/Rhoban/onshape-to-robot — Exporta assembly do Onshape para URDF/SDF/**MJCF** direto; usado pelo Open Duck. Seção "Design-time considerations" explica como nomear mates/links para o export sair certo.
- **URDF → MJCF (PickNik guide)** — https://docs.picknik.ai/how_to/configuration_tutorials/create_robot_sim_config/migrate_to_mujoco_config/ — Passos de pós-processamento: `<mujoco><compiler discardvisual="false"/>`, meshes de colisão simplificadas, atuadores de posição.
- **mujoco_ros2_control URDF→MJCF tool** — https://control.ros.org/kilted/doc/mujoco_ros2_control/mujoco_ros2_control/docs/tools.html — Alternativa de conversão com tags MuJoCo embutidas no URDF.

---

## 5. Atuadores e sim-to-real (onde projetos de hobby quebram)

- **BAM — Better Actuator Models** (Rhoban) — https://github.com/Rhoban/bam · https://bam.readthedocs.io — Modelos de atrito estendidos (Stribeck, dependência de carga) identificados em bancada de pêndulo; **já tem parâmetros para Feetech STS3215 (7,4 V)**, Dynamixel XL330/XL320/MX; API para plugar no MuJoCo CPU e no mjlab. O Open Duck usou isso. Paper: "Extended Friction Models for the Physics Simulation of Servo Actuators".
- **Backlash em servos UART (Feetech/Waveshare)** — https://zenodo.org/records/19261714 — Bancada low-cost para medir folga; útil para alimentar o modelo de backlash do Playground.
- **SysID diferenciável com MJX** — https://arxiv.org/abs/2508.04696 — Identificação de parâmetros sem medir torque, integrada ao loop de RL.
- **Sim-to-real de locomoção bípede compliant (revisão)** — https://www.emergentmind.com/topics/sim-to-real-transfer-of-compliant-bipedal-locomotion — Fontes do gap (compliance, atraso 5–20 ms, atrito, contato), ranges típicos de randomização (massa ±10%, atrito 0,4–1,0, etc.).
- **PGTT (MJX, terreno)** — https://arxiv.org/abs/2510.18348 — Tabela de rewards e randomização para locomoção perceptiva em MJX; mostra que MJX exige bem menos compute que Isaac Gym.
- Blog citado pelo Open Duck como referência de sim2real: https://www.haonanyu.blog/post/sim2real/

---

## 6. O celular acoplado — o que muda

- **PhoneBot** (acima, https://arxiv.org/abs/2610.08737) — modelo mental: o telefone **é** o torso/IMU. Montagem rígida faz a IMU do telefone virar a IMU de referência da política, sem calibração extra.
- **OpenBot** (Intel, ICRA 2021) — https://github.com/isl-org/OpenBot · https://arxiv.org/abs/2008.10631 — Stack Android (TFLite, sensores, comunicação serial/Wi-Fi) que a PhoneBot estende; referência de arquitetura "phone como cérebro".
- **ONNX Runtime Mobile** — https://onnxruntime.ai/docs/tutorials/mobile/ — Rodar a política exportada do Playground/mjlab no Android/iOS.
- **Android Sensor API (IMU)** — https://developer.android.com/develop/sensors-and-location/sensors/sensors_motion — `TYPE_ROTATION_VECTOR`, `TYPE_GYROSCOPE`, taxas de amostragem; necessário se a IMU do telefone for a observação.

---

## 7. Dicas práticas para quem está montando o MJCF e o ambiente

1. **Modele o celular como um `<body>` próprio** com massa/inércia reais (um smartphone médio tem 170–230 g — num robô de ~1,5 kg isso é 10–15% da massa, acima do CoM). Coloque-o na posição exata do suporte.
2. **Randomize massa, posição e orientação do celular** (±30–50 g, ±1–2 cm) durante o treino — o usuário vai trocar de modelo de telefone e o suporte flexiona.
3. **Decida cedo qual IMU a política observa** (telefone vs. IMU dedicada no corpo). Se for o telefone, o site/sensor no MJCF deve ficar no frame do telefone, incluindo sua inclinação de montagem.
4. **Use atuadores de posição (`<position kp=...>`) com `forcerange` = torque real do servo** e aplique o modelo de atrito do BAM. Servos de hobby não são torque-controlados; a política deve emitir alvos de posição (ou resíduos sobre uma pose padrão, como PhoneBot/Open Duck).
5. **Comece pelo imitation reward** (reference motion generator do Open Duck) ou por reward periódico (Siekmann) — PPO só com "ande pra frente" gera gaits feios. Depois ajuste regularizações: action rate, torques, orientação do tronco, feet air time, altura do pé em swing.
6. **Frequência de controle 50 Hz, `timestep` 0,002–0,005 s, decimation 4–10** é o padrão desses projetos; mantenha igual no runtime real.
7. **Adicione atraso de observação/ação (1–3 steps) e ruído** na randomização — é o principal gap em servos UART via Wi-Fi/serial.
8. **Pipeline de validação:** treino no MJX/Warp → inferência no MuJoCo CPU (`mujoco_infer.py`, sim2sim) → robô real. Sanity check com agente zero/random (mjlab) antes de gastar GPU.
9. **Colisões:** use primitivas (caixas/cápsulas) para pés e pernas, não as meshes do CAD; o MJX sofre com meshes de colisão complexas.
10. **Treino sem DR primeiro** para confirmar que referência e rewards funcionam; depois liga a randomização (dica do ProtoMotions).
