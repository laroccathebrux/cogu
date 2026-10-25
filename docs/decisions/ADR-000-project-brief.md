# Cogu — Robô bípede cogumelo, open source e open hardware

> Documento de decisões de projeto. Serve como ponto de partida para bootstrap do repositório
> e como contexto completo para qualquer agente (Claude Code) ou pessoa que entre no projeto.
> Data: 2026-10-25. Autor: Alessandro La Rocca Silveira. Status: pré-CAD, visual validado.

---

## 1. Visão

Um robô bípede de mesa, **~28 cm**, em forma de **cogumelo**, que **anda, vê e se expressa**, no espírito do Open Duck Mini / Microduck (Pollen Robotics), mas com design **original** e **100% reproduzível por qualquer maker** com uma impressora FDM comum.

Objetivo central: **outras pessoas conseguirem recriar, modificar e redistribuir** o robô. Reprodutibilidade é requisito de engenharia, não um "nice to have".

Nome provisório: **Cogu** (curto, pronunciável em qualquer língua, claramente brasileiro). Alternativas descartadas: Fungo, Mushbot, Shroomy, Bolet, Amanita.

Roadmap de produto: o Cogu é a **plataforma** (pernas + stack). Um segundo modelo, um **dinossauro/terópode com cauda atuada** (apelo comercial maior, cauda como massa de reação para equilíbrio), é a v2 e deve **reaproveitar as mesmas pernas e o mesmo software**.

---

## 2. Por que este caminho (histórico das decisões)

1. Primeira ideia: bípede "profissional" com atuadores quasi-direct-drive (RobStride, DAMIAO, CubeMars). **Descartado**: US$ 1.200–2.400 só de atuadores, caro demais para hobby/comunidade.
2. Referência adotada: **Microduck** (Pollen, US$ 399, 15 motores, open source) e seu antecessor DIY **Open Duck Mini v2** (BOM < US$ 400, servos Feetech STS3215, Raspberry Pi Zero 2W, BNO055, RL em MuJoCo com sim2real comprovado).
3. Lição-chave: **na escala de 25–40 cm / ~1 kg, servos serial-bus de posição são suficientes**. A qualidade da caminhada vem do software (RL + identificação de atuador + domain randomization), não do atuador. QDD importa para robôs de 8 kg, não para um de 1 kg.
4. Decidido fazer um robô **original e completo** (não só pernas), para a comunidade.
5. Forma escolhida: **cogumelo** (mais simples que o dinossauro para o v1; engenharia favorável: chapéu abriga sensores e protege nas quedas, corpo redondo baixa o centro de massa).
6. Visual validado com renders gerados no Gemini (ver §5).

---

## 3. Constituição do hardware (regras invioláveis)

Estas regras definem o projeto. Qualquer PR que as viole deve ser recusado.

- **100% impresso em 3D**: toda peça estrutural sai de FDM. Zero usinagem, corte a laser, chapa dobrada, tubo de carbono, peça sob encomenda.
- **Impressora de referência: Bambu Lab A1 Mini** (mesa 180 × 180 × 180 mm, bico 0,4 mm). Nenhuma peça maior que **160 mm** na maior dimensão (margem para aba/brim). Suportes mínimos; orientação de impressão documentada por peça.
- **Materiais: só PLA e PETG.** Sem TPU (tirado deliberadamente: variação entre marcas e dificuldade para iniciantes), sem resina, sem nylon/CF.
  - PETG para estrutura (pernas, corpo, colar); PLA aceitável.
  - Flexibilidade/grip vem de peças de prateleira (O-rings) ou de geometria flexível impressa em PETG.
- **Fixação padronizada**:
  - Insertos roscados a quente **M3 × 5,7 mm (OD 4,6 mm)**. Furo para inserto: **Ø4,0 mm × 6,5 mm de profundidade**, chanfro de entrada 0,5 mm. Parede mínima 2 mm ao redor. Nunca em região com menos de 3 perímetros.
  - Parafusos **M3 × 8, × 12, × 16**, cabeça cilíndrica allen. Apenas esses três comprimentos no BOM inteiro.
  - Fixação dos servos: horn metálico e parafusos **originais do STS3215** (M2).
  - Peça de teste de insertos no repositório (bloco com 4 furos) para calibrar temperatura do ferro (PLA ~200 °C, PETG ~230 °C).
- **Rolamentos 688ZZ** (8 × 16 × 5 mm) nas articulações, no lado oposto ao horn do servo.
- **Grip dos pés: O-rings de nitrilo** em canais impressos na sola (presos mecanicamente, sem cola, trocáveis). Alternativa documentada: pés de borracha adesivos.
- **Anel de proteção do chapéu**: PETG com geometria flexível (nervuras/lattice), peça trocável. Alternativa: O-ring grande ou câmara de bicicleta no canal da aba.
- **Tolerâncias**: encaixes projetados com 0,2–0,3 mm de folga. Arquivo de calibração (pino/furo) no repositório para validar a impressora antes de imprimir as peças.
- **Nenhum cabo externo**, exceto o do próprio servo dentro da perna. Canais de cabo dentro da coxa/canela e passagem pelo corpo.
- **Componentes comprados**: apenas itens de prateleira disponíveis globalmente (AliExpress/Amazon/EU/US/BR), com alternativas por região no BOM.

---

## 4. Anatomia e dimensões (v1)

Altura total: **~280 mm**. Massa alvo: **< 1,2 kg**.

| Parte | Dimensão | Função |
|---|---|---|
| Chapéu | Ø **160 mm**, domo baixo (~55 mm de altura), coral vermelho com 3 manchas creme | Sensores (câmera frontal sob a aba, fenda de 360° na borda para LiDAR/ToF, microfones laterais), proteção em quedas, expressão (tilt), anel de LEDs na borda inferior (humor/status) |
| Colar | Curto (~15–20 mm), creme | Esconde os 2 servos de tilt do chapéu (pitch + roll). Decisão visual: mecanismo **escondido**, não aparente |
| Corpo | Esfera/elipsoide Ø **~95 mm**, creme | É torso **e** cabeça. Abriga Pi, IMU, bateria (o mais embaixo possível), placa de servos, regulador 5 V. Tela embutida na frente |
| Tela/rosto | CYD 2,8" horizontal, área visível 58 × 43 mm, moldura impressa com 4 parafusos à vista | Dois olhos grandes estilo cartoon; boca opcional como traço |
| Pernas | Coxa **60 mm**, canela **60 mm**, pé 15 mm de altura; **digitígradas** (coxa para frente e para baixo, joelho para trás, canela para trás e para baixo, tornozelo atrás do pé). Espaçamento entre quadris ~40 mm | Pernas **curtas de propósito**: corpo quase encostando nos pés quando parado; ~1/3 da altura total. CoM baixo, passada curta (favorece RL) |
| Pés | Largos, arredondados, planos; 2 O-rings na sola; chaves de contato (switch) | Base de apoio generosa |

**Graus de liberdade**: 5 por perna (hip yaw, hip roll, hip pitch, joelho, tornozelo pitch) + 2 no chapéu = **12 servos**. Sem pescoço, sem braços.

**Movimentos previstos desde o início**: andar (velocity tracking), virar, sentar/levantar, recuperar de queda (deitado → em pé), tilt de chapéu expressivo.

**Checagens geométricas obrigatórias no CAD**: chapéu não pode colidir com joelhos ao agachar/sentar (define o pitch máximo do chapéu e a altura do colar); verificar envelope de cada articulação.

Restrição de massa: cada 50 g no chapéu vale ~150 g no corpo. Chapéu em parede fina com nervuras; LiDAR real (ex.: RPLIDAR A1, ~170 g) é pesado demais para o v1 — usar **ToF multizona (VL53L5CX, ~2 g)** no v1 e deixar a fenda como identidade visual/preparação.

---

## 5. Visual validado (renders Gemini)

Prompt base e ajustes estão no histórico; itens **confirmados** pela sequência de imagens:

- Chapéu coral com 3 manchas creme e fenda escura na borda → **identidade visual, manter**.
- Corpo esférico creme, tela emoldurada por 4 parafusos → **manter**.
- Pés grandes arredondados → manter (mais largos que o esboço inicial; bom para estabilidade).
- Pernas curtas, corpo baixo → confirmado no segundo render.
- Colar escondendo mecanismo do chapéu → confirmado.
- Estados de rosto: olhos abertos (padrão), sorriso, "dormindo".

Não resolvido no render (gerador não consegue): **joelho para trás** — isso é decisão de engenharia, feita no CAD.

Proporção observada no render (~1,9× chapéu/corpo) ficou mais agradável que a planejada (1,6×). Como 160 mm é o teto da A1 Mini, a correção é **reduzir o corpo para ~95 mm**, não aumentar o chapéu.

---

## 6. Eletrônica (v1)

| Item | Qtd | Observação |
|---|---|---|
| Feetech STS3215 7,4 V (serial bus, encoder magnético 12 bit, feedback de posição/corrente) | 12 | 10 pernas + 2 chapéu. Alternativa de mais torque nos quadris: STS3250 |
| Placa controladora de servos Feetech (TTL/serial) | 1 | |
| Raspberry Pi Zero 2W (v1) — compartimento compatível com Pi 5 | 1 | Roda a política a 50 Hz |
| IMU BNO055 | 1 | Posição canônica no corpo, documentada |
| Tela **ESP32-2432S028 "Cheap Yellow Display" (CYD)**, TFT ILI9341 2,8" 320×240, touch resistivo, ESP32-WROOM | 1 | Rosto. O ESP32 **renderiza os olhos sozinho**; o Pi só envia estado de expressão por UART (conector P1) ou ESP-NOW/Wi-Fi. Touch = interação ("carinho"). ~50–60 g, ~150 mA a 5 V, regulador 5 V dedicado |
| ToF multizona VL53L5CX (ou similar) | 1 | Sob a aba, frontal |
| Câmera Raspberry Pi | 1 | Frontal sob a aba |
| Chaves de contato nos pés | 4 | 2 por pé |
| Bateria LiPo 2S (7,4 V) | 1 | No fundo do corpo |
| Anel de LEDs endereçáveis (WS2812) | 1 | Borda inferior do chapéu (opcional v1) |
| Microfone / speaker | opcional | |
| USB-C e chave liga/desliga | 1 | Atrás do corpo |

Variante documentada "mais barata": dois olhos de LED no lugar da CYD.

**Arquitetura de software embarcado**: locomoção (Pi) e expressão (ESP32) **desacopladas**, com um **protocolo serial de expressão** definido como parte da spec aberta (ex.: `EXPR <estado> <x> <y>`).

---

## 7. Software e simulação

- **Simulação/treino**: MuJoCo + **MuJoCo Playground** (JAX). Treino de locomoção por RL com domain randomization; opcionalmente imitação a partir de movimentos de referência (como o Open Duck).
- **Identificação de atuador**: **BAM (Rhoban)** nos STS3215, com os dados publicados no repositório, para o servo simulado se comportar como o real.
- **Deploy**: política ONNX rodando no Pi a 50 Hz; runtime inspirado no `Open_Duck_Mini_Runtime`.
- **Reprodutibilidade**: seed fixa, checkpoint publicado (Hugging Face), MJCF **gerado automaticamente a partir do CAD** (nunca editado à mão).
- **Compute de treino**: MacBook Pro M1 Max (CPU; horas por política) ou Hugging Face Jobs.
- Referências de stack: Open Duck Mini v2 (apirrone), Open Duck Playground, Microduck (pollen-robotics/microduck, Apache-2.0).

---

## 8. CAD: tudo em código (build123d)

Decisão: **fonte única de verdade em build123d** (Python). Blender fica **só para renders** a partir do STL exportado. Motivos: CAD paramétrico diffável em PR; STL/STEP/MJCF gerados do mesmo script em CI; o modelo de simulação nunca desatualiza em relação às peças.

- `parameters.py` central: `cap_diameter = 160`, `body_diameter = 95`, `thigh_length = 60`, `shin_length = 60`, `foot_height = 15`, `hip_spacing = 40`, folgas, dimensões de inserto, etc.
- Chapéu = spline revolucionada; corpo = elipsoide com corte para a tela e cavidade interna; pés = retângulo com fillet + canais de O-ring; manchas = rebaixos de 0,6 mm (troca de cor na A1 Mini) ou peças encaixadas.
- Fixar versão do build123d no `pyproject`. Visualizador: `ocp_vscode` (ou export GLB).
- Export automático: STL (por peça, com orientação de impressão), STEP, MJCF (massa/inércia de cada peça calculadas do CAD).
- Regras de design checadas em código: parede mínima ao redor de insertos, dimensão máxima 160 mm, folgas.

---

## 9. Licenças e governança

- Hardware: **CERN-OHL-S v2** (copyleft forte — quem modificar precisa abrir). Decisão pendente de confirmação final; alternativa CERN-OHL-P (permissiva).
- Software: **Apache-2.0**.
- Documentação: **CC-BY 4.0**.
- Licenças presentes desde o primeiro commit. Repositório **público desde o início**.

---

## 10. Estrutura de repositório proposta

```
cogu/
├── README.md               (PT + EN; visão, foto/render, "primeiros 60 min")
├── LICENSE-HARDWARE        (CERN-OHL-S)
├── LICENSE                 (Apache-2.0)
├── LICENSE-DOCS            (CC-BY)
├── docs/
│   ├── decisions/          (este documento e ADRs futuros)
│   ├── build-guide/        (impressão, insertos, montagem, fiação — com fotos)
│   ├── bom/                (BOM com alternativas por região)
│   └── expression-protocol.md
├── cad/
│   ├── parameters.py
│   ├── parts/              (leg_thigh.py, leg_shin.py, foot.py, body.py, cap.py, collar.py, bezel.py ...)
│   ├── assembly.py
│   ├── export.py           (STL / STEP / MJCF)
│   └── test_prints/        (calibração de inserto, pino/furo)
├── sim/
│   ├── mjcf/               (gerado)
│   ├── playground/         (envs, treino, seeds)
│   └── actuator_id/        (BAM, dados)
├── firmware/
│   ├── face-esp32/         (CYD: olhos, protocolo serial)
│   └── runtime-pi/         (loop 50 Hz, servos, IMU, política ONNX)
├── hardware/
│   └── wiring/             (esquemas)
└── .github/workflows/      (CI: gera STL/STEP/MJCF a cada push)
```

---

## 11. Roteiro

1. **Sim primeiro (semanas 1–2)**: pelve + perna em build123d → MJCF → política de caminhada em MuJoCo Playground com modelo aproximado de servo. Nada impresso ainda. Valida proporções e torques.
2. **Uma perna real (semanas 3–4)**: imprime, monta, roda BAM no STS3215, compara com a sim. Publica dados.
3. **Robô completo (semanas 5–8)**: corpo, chapéu, CYD, política treinada, primeiro vídeo, **release v0.1**.
4. **Depois**: documentação de ensino, variantes de perna, primeiro PR externo aceito, dinossauro (v2) sobre as mesmas pernas.

---

## 12. Decisões em aberto

- [ ] Nome final (Cogu?) e checagem de marca/domínio.
- [ ] Chapéu com 2 DOF já no v1 **ou** fixo no v1 (10 servos) e tilt no v1.1. Tendência: 2 DOF no v1, com colar.
- [ ] Pés: redondos grandes (render) vs achatados largos (esboço). Tendência: largos e planos (mais fáceis de imprimir, melhores para as chaves de contato), com cantos bem arredondados.
- [ ] Confirmar CERN-OHL-S vs P.
- [ ] Bateria exata (capacidade/dimensões) após fechar o volume interno do corpo.
- [ ] Manchas do chapéu: troca de cor vs peças encaixadas.

---

## 13. Instruções para o agente que fizer o bootstrap

1. Criar a estrutura de §10 com licenças e README (PT/EN) resumindo §1, §3 e §4.
2. Criar `cad/parameters.py` com os valores de §4 e §3.
3. Implementar a **perna paramétrica** (coxa, canela, pé) em build123d com: furos e encaixe do STS3215 (consultar datasheet para dimensões exatas do corpo e do horn), furo de rolamento 688ZZ, furos de inserto M3 conforme §3, canal de cabo, canais de O-ring na sola.
4. `export.py` gerando STL por peça e um MJCF mínimo da perna (massa/inércia do CAD).
5. Incluir `cad/test_prints/` (bloco de insertos e teste pino/furo).
6. Workflow de CI que executa `export.py` e publica os STLs como artefatos.
7. Não editar MJCF à mão; não introduzir material ou fixação fora de §3.

---

## 14. Referências

- Microduck — https://pollen-robotics.com/microduck/ · https://github.com/pollen-robotics/microduck
- Open Duck Mini v2 — https://github.com/apirrone/Open_Duck_Mini · MakerWorld (perfil de impressão) — https://www.makerworld.com/en/models/2755651
- Open Duck Mini Runtime — https://github.com/apirrone/Open_Duck_Mini_Runtime
- BAM (identificação de atuador, Rhoban) — https://github.com/Rhoban/bam
- MuJoCo Playground — https://github.com/google-deepmind/mujoco_playground
- Berkeley Humanoid Lite (referência de atuador impresso, não usada no v1) — https://arxiv.org/abs/2504.17249
- build123d — https://github.com/gumyr/build123d
- CYD (ESP32-2432S028) — https://github.com/witnessmenow/ESP32-Cheap-Yellow-Display
- Licenças: CERN-OHL v2 — https://ohwr.org/cern_ohl
