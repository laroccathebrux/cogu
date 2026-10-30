# sim/mjcf

O MJCF do Cogu **não vive aqui**: ele é gerado a partir do CAD por `uv run python -m cad.export`
e sai em `build/mjcf/` (com as meshes em `build/mjcf/meshes/`). Os nomes dos arquivos são históricos
(`sim/playground` os referencia); os dois têm o robô completo do ADR-002: pelve, duas pernas
digitígradas de 5 DOF, tampa do quadril, berço e celular.

- `cogu_lower_body.xml` — peças impressas como meshes de colisão. Para `sim/check_mjcf.py` e o viewer.
  Tampa e berço colidem só com o chão (bits de contato 2/2): o MuJoCo colide **hulls convexos**, e o hull
  da tampa aberta embaixo "colidiria" com o que está sob ela. O celular também colide com o chão (quedas).
  Servos de yaw, celular, bateria e placa da medula são meshes de referência (`ref_*`); a massa deles entra
  como massa pontual no `inertial` do corpo da pelve.
- `cogu_lower_body_mjx.xml` — variante de treino (MuJoCo Playground / MJX): meshes só visuais, uma
  caixa de colisão por sola (sola–chão e sola–sola), nenhuma outra autocolisão. Mesmas juntas, massas,
  inércias, atuadores e keyframe.

Os dois trazem o site `imu` na placa da medula dentro da tampa (`parameters.IMU_POS`), sites nas solas,
sensores (gyro, acelerômetro, velocímetro, vetor "para cima", velocidades globais, contato pé–chão), a
câmera `track` que segue o robô e a câmera `phone_cam` na câmera traseira do celular.

Nunca edite um MJCF à mão. Mude `cad/parameters.py`, a peça em `cad/parts/` ou `cad/export.py` e gere de novo.

`uv run python sim/check_mjcf.py` carrega o modelo gerado, imprime massas e inércias, e simula
3 s parado na pose de repouso para verificar que o robô fica em pé.
