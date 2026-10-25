# sim/mjcf

O MJCF do Cogu **não vive aqui**: ele é gerado a partir do CAD por `uv run python -m cad.export`
e sai em `build/mjcf/` (com as meshes em `build/mjcf/meshes/`):

- `cogu_lower_body.xml` — robô completo (pernas, corpo, colar, chapéu com 2 DOF), peças impressas como meshes
  de colisão. Para `sim/check_mjcf.py` e o viewer. O nome do arquivo é histórico (`sim/playground` o referencia).
  As cascas (corpo e chapéu) colidem só com o chão (bits de contato 2/2): o MuJoCo colide **hulls convexos**, e o
  hull de uma casca oca "colidiria" com o quadril que ela esconde. A folga real casca–perna é medida por
  `cad/check_body.py` (booleanos exatos). Colar, suporte do chapéu e moldura da tela são só visuais.
  Servos, CYD, Pi, bateria e placas aparecem como meshes de referência (`ref_*`, grupo 2, sem colisão nem massa:
  a massa deles entra como massa pontual no `inertial` de cada corpo).
- `cogu_lower_body_mjx.xml` — variante de treino (MuJoCo Playground / MJX): meshes só visuais, uma
  caixa de colisão por sola (sola–chão e sola–sola), nenhuma outra autocolisão. Mesmas juntas de perna, massas,
  inércias, atuadores e keyframe; o **pescoço fica fixo** (sem juntas `neck_*`), porque a política controla só os
  10 servos das pernas — corpo, colar e chapéu entram como massa e inércia.

Os dois trazem o site `imu` na posição da BNO055 dentro do corpo (`parameters.IMU_POS`), sites nas solas, sensores (gyro, acelerômetro, velocímetro,
vetor "para cima", velocidades globais, contato pé–chão) e a câmera `track`.

Nunca edite um MJCF à mão. Mude `cad/parameters.py`, a peça em `cad/parts/` ou `cad/export.py` e gere de novo.

`uv run python sim/check_mjcf.py` carrega o modelo gerado, imprime massas e inércias, e simula
3 s parado na pose de repouso para verificar que o robô fica em pé.
