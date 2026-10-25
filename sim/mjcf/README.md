# sim/mjcf

O MJCF do Cogu **não vive aqui**: ele é gerado a partir do CAD por `uv run python -m cad.export`
e sai em `build/mjcf/` (com as meshes em `build/mjcf/meshes/`):

- `cogu_lower_body.xml` — todas as peças como meshes de colisão. Para `sim/check_mjcf.py` e o viewer.
- `cogu_lower_body_mjx.xml` — variante de treino (MuJoCo Playground / MJX): meshes só visuais, uma
  caixa de colisão por sola (sola–chão e sola–sola), nenhuma outra autocolisão. Mesmas juntas, massas, inércias, atuadores e keyframe.

Os dois trazem o site `imu` na pelve, sites nas solas, sensores (gyro, acelerômetro, velocímetro,
vetor "para cima", velocidades globais, contato pé–chão) e a câmera `track`.

Nunca edite um MJCF à mão. Mude `cad/parameters.py`, a peça em `cad/parts/` ou `cad/export.py` e gere de novo.

`uv run python sim/check_mjcf.py` carrega o modelo gerado, imprime massas e inércias, e simula
3 s parado na pose de repouso para verificar que o robô fica em pé.
