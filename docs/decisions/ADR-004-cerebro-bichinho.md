# ADR-004 — O cérebro do Cogu: um bichinho que conversa

Data: 2026-11-02. Status: **aceita a direção**; os itens marcados como revisáveis serão decididos por estudo no
simulador. Complementa o ADR-002 (o celular é o cérebro, a medula no ESP32 anda e levanta).

## Problema

A marcha (v10) e o levantar (v8) estão prontos no simulador, mas o robô ainda não pode ser montado (faltam peças).
Falta decidir o que o celular faz: o que o Cogu é, o que ele percebe, como pensa e o que pode ir para a nuvem.

## Decisão

1. **O Cogu é um bichinho de estimação inteligente que conversa.** Ele mantém um diálogo, articula o que pensa e
   explica o que vai fazer: "um Reachy Mini com pernas". Não é um robô de carinhas e sons sem sentido. A expressão
   vem da voz e do corpo inteiro: inclinar para a frente quando presta atenção, abaixar quando está com sono,
   balançar quando está animado, virar para quem fala.
2. **Camadas, cada uma mais lenta e mais "inteligente" que a de baixo; nenhuma depende da de cima para não cair:**

   | Camada | Onde | Ritmo | Faz |
   |---|---|---|---|
   | Medula | ESP32 | 50 Hz | marcha, levantar, sentar, deitar; proteções (torque desliga se tombar, se o comando sumir, bateria baixa) |
   | Reflexos e navegação | celular | 10–30 Hz | posição (ARKit), obstáculos (LiDAR), ir até um ponto, seguir, **parar** |
   | Percepção | celular | 1–5 Hz | pessoas, o dono, objetos (bola, sofá, porta), chão e paredes (ARKit, Vision, Core ML) |
   | Memória do mundo | celular | sempre | mapa com significado: o que está onde, quando foi visto, o que dá para fazer com cada coisa |
   | Raciocínio e conversa | celular ou nuvem | segundos | conversa, decide o que fazer e chama ferramentas |

3. **O LLM nunca controla motor.** Ele chama ferramentas do app (`o_que_estou_vendo`, `ir_ate`, `seguir`, `parar`,
   `sentar`, `deitar`, `empurrar_bola`, `lembrar`, ...); a navegação e a medula executam. "Modelo de mundo", por
   enquanto, é a memória do mundo acima (um mapa semântico que o LLM lê em texto). Modelos de mundo aprendidos (que
   imaginam o futuro em vídeo) ficam para depois, como mais uma ferramenta.
4. **Local primeiro; nuvem só onde o local ficar ruim.** Local tem menos atraso e mais privacidade.

   | Função | Local (iPhone) | Nuvem, se o local ficar ruim |
   |---|---|---|
   | Comandos simples: pare, siga, sente, deita, vem, fica | reconhecimento de fala no aparelho, sem LLM | não precisa |
   | Câmera, LiDAR, mapa da casa, rostos | ARKit, Vision, Core ML | **nunca sai do celular** |
   | Entender o que a pessoa disse | reconhecimento de fala no aparelho | ElevenLabs |
   | Pensar e conversar | modelo de linguagem da Apple no aparelho | LLM da plataforma de agentes da ElevenLabs |
   | Voz | frases pré-geradas com a voz do Cogu | ElevenLabs ao vivo |

   Para a nuvem vão só a voz de quem conversa com ele e um **resumo em texto** do que ele sabe ("estou na sala, o dono
   está a 1 m, a bola está perto do sofá"). Imagem, profundidade e mapa nunca saem.
5. **A voz é sempre a mesma**, a da ElevenLabs. Sem internet ele fala menos, com frases pré-geradas na mesma voz, em
   vez de trocar para uma voz sintética do aparelho.
6. **O "pare" não passa por LLM**: o reconhecimento local manda direto para a navegação e a medula.
7. **Profundidade: o LiDAR do iPhone no robô.** No simulador, que não tem LiDAR, a imagem de profundidade da câmera
   `phone_cam` faz o papel dele, degradada como o LiDAR do iPhone (256 × 192, alcance ~5 m, ruído, falhas em preto,
   espelho e vidro). A posição do ARKit é a verdadeira com erro e atraso.
8. **Movimentos novos, treinados no simulador como o levantar:**
   - **sentar** (2026-11-03): quase no chão. Primeiro agacha sobre os pés (35° para trás, pelve a 165 mm), depois as
     pernas desdobram para a frente: pelve a ~106 mm, bloco do quadril a ~2 cm do chão, corpo 44° para trás, olhando
     para cima como um pet sentado (ele continua vendo e conversando). Pede o quadril até −130° e o tornozelo até
     +82°, já livres no CAD. Antes disso o sentar parava no agachado, com o bloco a 7 cm do chão. Encostar
     o bloco dos servos do quadril (~96 mm abaixo do tambor) no chão e levantar de volta pede o quadril em ~−145°, o
     que mexe no quadril em volta do servo do roll (`sim/playground/README.md`, "Estudo: sentar no chão");
   - **deitar**: deita de costas devagar (no máximo 0,1 m/s), seguindo de trás para a frente o caminho que o levantar
     usa para subir, e no fim estica as pernas para o corpo ficar reto (90°); para levantar, o levantar v8;
   - **empurrar a bola**: primeiro com a marcha atual (anda até ficar atrás da bola, alinha e anda contra ela); uma
     política própria só se ficar desajeitado;
   - **postura expressiva**: altura e inclinação do corpo como comandos da marcha, além da velocidade.

## Revisável

- Deitar **de costas** (o celular olha para o teto). Alternativas: de bruços põe o celular no chão; "chocando"
  (pernas dobradas para trás sob o corpo, canelas no chão, corpo em pé, torque zero) existe como pose parada, mas
  não achei ainda uma transição que chegue nela sem tombar para trás.
- Sentar quase no chão, inclinado 44° para trás (bloco a ~2 cm): segura menos empurrões para a frente que o agachado
  (15–17 de 20 até 0,4–0,6 m/s, contra 20/20), e uma política treinada em volta do roteiro (pose v2) não mudou isso.
  Alternativas: o agachado de antes, ou o chão de verdade com o quadril indo até ~−145°.
- Qual LLM local e qual na nuvem: decidido medindo tempo até a primeira palavra e qualidade das respostas. Primeira
  medida no simulador (2026-11-03, `sim/brain/conversation.py`): um modelo de 3B, do tamanho do da Apple, responde
  em ~1 s, mas escolhe as ferramentas mal em frases livres (6 de 9). Modelos de 8B e 35B escolhem melhor, mas são
  lentos demais rodando no mesmo Mac. As frases fixas continuam sem LLM.
  A ligação com a ElevenLabs está pronta e testada contra um servidor falso (`sim/brain/eleven.py`): o agente usa
  Claude Haiku 5.5, as ferramentas do Cogu como ferramentas do cliente e só texto; a voz é o eleven_flash_v2_5 e o
  ouvido é o Scribe v2. Falta medir com a chave da conta.
- Bola: tamanho 5 (22 cm, ~430 g) é o pedido; no treino também 3 e 4. O Cogu tem ~40 cm de altura (câmera a 38,6 cm).

## Consequências

- No simulador: um apartamento (sala, sofá, mesa, porta, corredor, bola, uma pessoa que anda), o LiDAR simulado,
  navegação e comandos com percepção perfeita, depois só pela câmera; o motor de comportamento; a conversa por cima.
- Treinos novos no Colab: sentar e deitar, postura expressiva, empurrar a bola.
- O app em Swift vem depois, com a mesma lógica; só a ligação muda (simulador → BLE com o ESP32).
- `docs/expression-protocol.md` é da época do cogumelo (tela CYD) e não vale mais.
