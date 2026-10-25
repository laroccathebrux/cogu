# sim/mjcf

O MJCF do Cogu **não vive aqui**: ele é gerado a partir do CAD por `uv run python -m cad.export`
e sai em `build/mjcf/cogu_lower_body.xml` (com as meshes em `build/mjcf/meshes/`).

Nunca edite um MJCF à mão. Mude `cad/parameters.py` ou a peça em `cad/parts/` e gere de novo.

`uv run python sim/check_mjcf.py` carrega o modelo gerado, imprime massas e inércias, e simula
3 s parado na pose de repouso para verificar que o robô fica em pé.
