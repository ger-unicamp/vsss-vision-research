# Metodologia

Como cada decisão experimental está implementada, e por quê. As
justificativas vêm do levantamento em
[`related-work.md`](related-work.md); os protocolos de cada proposta estão
em [`p1-illumination.md`](p1-illumination.md) e
[`p2-latency.md`](p2-latency.md).

---

## 1. Esquema de classes: por cor de time, não por robô

**Decisão.** Três classes: `ball`, `robot_yellow`, `robot_blue`. O
`robot_id` **não** vem da rede.

**Por quê.** As regras da categoria mandam que a cor de identificação do
time alterne entre partidas e que a etiqueta seja destacável. Um modelo com
classes por robô (`robot0`, `robot1`, `robot2` — o esquema do trabalho de
referência) fica preso às camisas de uma equipe e a uma cor: o modelo
público resultante detecta os três robôs da própria equipe e a bola, e
**nenhum robô adversário**. Num jogo, metade dos objetos em campo é
invisível para ele.

Com classes por cor:

- o mesmo modelo serve os dois lados; trocar de lado é mudar `team_color`
  na configuração, não retreinar;
- o adversário é detectado — e passa a ter métricas, o que hoje não existe
  na literatura de VSSS;
- o modelo aprende "robô com etiqueta amarela", conceito que generaliza
  entre equipes, em vez de "camisa rosa com um triângulo inscrito".

**Implementação.** `vision_config.json::yolo_class_map` mapeia nome de
classe do modelo para papel semântico. O mesmo arquivo alimenta o exportador
de dataset e o detector, então treino e inferência não podem discordar sobre
o que é a classe 1. Um mapa que não casa com `model.names` **levanta erro na
construção** do detector: um mapa errado não causa exceção durante a
inferência, apenas devolve zero detecções, e o experimento reportaria 100%
de detecções perdidas como se fosse resultado.

**Reprodução do esquema de referência.** Continua a uma configuração de
distância, acrescentando `robot_id` à entrada:

```json
"robot0": {"kind": "robot", "color": "blue", "robot_id": 0}
```

Comparar os dois esquemas sobre as mesmas cenas é, ele próprio, um
resultado defensável.

## 2. Identidade do robô

A classe diz o time; **quem** é cada robô vem de `vision/identity.py`:

| Estratégia | Como | Quando usar |
| :--- | :--- | :--- |
| `marker` | marcador de cor procurado **dentro da caixa detectada** | time próprio |
| `positional` | ordem por área da caixa | adversário (não há marcador) |

Restringir a busca do marcador à caixa é mais robusto que a busca global do
`ColorDetector`: um falso positivo de marcador no meio do campo não tem
caixa de robô ao redor e é descartado sem custo.

Uma caixa do time próprio **sem** marcador associado é descartada no frame,
não recebe id inventado. Inventar um id produziria exatamente a troca de
identidade que a métrica tenta medir. O `Tracker` decide se mantém o último
estado conhecido.

## 3. Orientação

Fonte primária: keypoints `front`/`back` de um modelo *pose*,
`θ = atan2(front − back)` — mesma convenção do trabalho de referência.
Fonte secundária, em modelos só de detecção: vetor centro-da-caixa →
centróide-do-marcador, como no `ColorDetector`.

As duas fontes **não devem ser misturadas na mesma tabela** sem dizer qual
foi usada: têm erros de natureza diferente. A do marcador é ruidosa quando
marcador e centro estão próximos.

Keypoint ausente no Ultralytics vem como `(0, 0)`. Isso é tratado como
"sem orientação" (`theta=None`), nunca como um ângulo apontando para o
canto superior esquerdo.

**A bola não tem orientação.** Os dois keypoints da bola são exportados com
visibilidade 0. O trabalho de referência anotou "quaisquer semirretas entre
duas extremidades da circunferência", o que ensina a rede a prever um eixo
que não existe.

## 4. Anotação e derivação das caixas

As anotações guardam **centro em metros e ângulo**, não caixas em pixel:

- é o que as métricas em escala real exigem;
- anotar um centro é muito mais barato que desenhar uma caixa;
- caixas derivadas são consistentes entre amostras, sem variância de
  anotador.

As caixas de treino saem das dimensões que a regra fixa — robô 7,5 cm,
bola 42,7 mm — convertidas por `pixels_per_meter` do frame retificado, com
folga de 15% para etiqueta e sombra. Os keypoints saem do centro e do
ângulo, meio corpo para cada lado.

Consequência a declarar no artigo: as caixas são **geradas**, não
anotadas. Isso muda o significado do mAP, que passa a medir concordância
com um modelo geométrico do objeto, não com o julgamento de um anotador.
É mais uma razão para o mAP não ser a métrica principal aqui.

## 5. Divisão treino/validação/teste: por arquivo de origem

**Nunca por frame.** Frames vizinhos de um mesmo vídeo são quase idênticos;
dividir aleatoriamente coloca quase-duplicatas dos dois lados e a métrica de
teste mede memorização. É um risco concreto no trabalho de referência: 653
imagens extraídas de vídeos de um único evento, sem menção a separação por
origem, com mAP50-95 de 0,99.

`benchmark/yolo_export.py::split_by_source` agrupa por arquivo de origem e
distribui grupos inteiros. Com poucos arquivos a proporção sai aproximada —
preferível a uma proporção exata com vazamento.

Validade externa exige mais: idealmente, **eventos diferentes** entre treino
e teste, e pelo menos um conjunto de teste com camisas de outra equipe.

## 6. Métricas

### 6.1 Erro de posição: distância euclidiana, em centímetros

```python
erro_cm = 100 * hypot(x_pred - x_gt, y_pred - y_gt)
```

O trabalho de referência usa `|(x − x₀) + (y − y₀)| / 2`, o módulo da
**soma** dos erros com sinal. Uma detecção deslocada em (+10, −10) px
recebe erro zero. Erros em X e Y não se cancelam fisicamente: o robô está
deslocado nas duas direções.

Unidade em centímetros, não em pixel. Erro em pixel não é comparável entre
resoluções nem entre montagens de câmera — limitação que a própria autora
declara.

### 6.2 Erro de orientação: diferença circular, em graus

```python
erro_rad = abs(atan2(sin(θ_pred - θ_gt), cos(θ_pred - θ_gt)))
```

Subtração crua reporta ~360° na virada 359°→1°. O sintoma aparece no
trabalho de referência como um histograma com valores em "(172,53; 215,53]
graus" — diferença angular verdadeira não passa de 180°. Parte dos 73°–97°
médios atribuídos ao baseline clássico é artefato de métrica.

### 6.3 Associação predição ↔ ground truth

Casamento guloso **por posição**, restrito ao mesmo time e a um raio máximo
(`--match-radius-cm`, padrão 10 cm ≈ um robô e meio), **ignorando o
`robot_id`**. Os ids são comparados **depois** da associação.

Se a associação usasse o id, uma troca de identidade apareceria como uma
detecção perdida somada a um falso positivo, e a taxa de trocas de
identidade seria sempre zero — apagando justamente o eixo em que se espera
que a rede neural leve vantagem sobre a cor.

O guloso basta com até 3 robôs por time e raio da ordem do robô. Para cenas
mais densas, trocar por atribuição ótima (Hungarian).

### 6.4 As métricas reportadas

| Métrica | Denominador |
| :--- | :--- |
| Erro de posição (cm): média, p95, máx | detecções associadas |
| Erro de orientação (graus): média, p95 | associadas **com** ângulo nos dois lados |
| Taxa de detecções perdidas | total anotado |
| Taxa de identificações trocadas | total **associado** |
| Taxa de falsos positivos | total anotado |
| Precisão / revocação / F1 | associação espacial, não IoU |
| Latência por estágio (ms): média, p50, p95, p99, máx | frames medidos |

Denominadores diferentes de propósito: dividir trocas de identidade pelo
total anotado misturaria o efeito de detecções perdidas.

**mAP entra como métrica secundária**, para comparabilidade com a
literatura de detecção. Não como métrica principal: um mAP alto é
compatível com um erro de vários centímetros, e é o erro em centímetros que
determina se o robô chega na bola.

## 7. Latência

Cronometragem com `time.perf_counter_ns` dentro do pipeline, não em script
à parte, e **idêntica** nos modos offline e ao vivo — a comparação entre
eles é então atribuível ao que de fato muda (câmera e transporte).

Estágios: `capture`, `decode`, `detect`, `track`, `convert`, `publish`,
`total`. O detector expõe seus sub-estágios via `Detector.profile()`; o
YOLO reporta `preprocess`, `inference` e `postprocess` separadamente. Isso
importa: se o custo estiver no pré-processamento e não na rede, a conclusão
sobre qual variante usar muda.

A medição de `detect` **inclui** pré-processamento e transferência para a
GPU. Medir só o *forward* subestima o que o laço de controle espera.

Reportar percentis, não só média: p99 de 80 ms com média de 12 ms significa
que o robô ocasionalmente age sobre um estado velho o bastante para ele já
ter saído da posição.

Frames de warm-up (`--warmup`) são descartados: os primeiros incluem
alocação de buffers e carga de pesos. `Detector.warmup()` é chamado antes de
qualquer medição.

**Referência externa:** na Small Size League, o atraso percepção-ação
medido foi de 53 ms em média (máx. 74) a 50 fps e 65 ms (máx. 99) a 30 fps
(Behnke et al., 2003). Qualquer número obtido aqui deve ser comparado a
essa ordem de grandeza.

## 8. Justiça da comparação

Uma comparação contra um pipeline clássico mal calibrado não sustenta
conclusão nenhuma sobre os métodos. As tabelas HSV do baseline no trabalho
de referência incluem faixas como *Saturation* [0:255] e *Value* [0:255] —
o eixo inteiro.

Regras adotadas aqui:

1. O `ColorDetector` é recalibrado com `tools/vision_calibrator.py` **na
   condição de referência** antes de qualquer comparação, e a calibração
   usada vai no resultado (`config_snapshot`).
2. Os dois detectores recebem **o mesmo frame retificado**, a mesma
   suavização e a mesma conversão para metros.
3. Na P1, os dois protocolos de calibração (única e por condição) são
   reportados separadamente — o clássico não é avaliado só no pior caso.
4. Para o YOLO, rodar os **dois braços de augmentation** (com e sem). Sem o
   braço de controle, não se distingue ganho de arquitetura de ganho de
   *augmentation*.

## 9. Reprodutibilidade

- Cada execução grava métricas, latência e um **recorte da configuração**
  (detector, pesos, `imgsz`, confiança, dispositivo, escala do campo, raio
  de casamento). O mesmo detector sobre o mesmo dataset dá números
  diferentes com outra resolução de entrada.
- Semente fixa no treino (`--seed`).
- `experiments/results/` fora do git: resultados são regeráveis. Versionar à
  mão só os citados, junto do commit que os gerou e do hardware usado.
- Pesos fora do git; publicar em release ou Hugging Face.

## 10. Ameaças à validade

| Ameaça | Mitigação |
| :--- | :--- |
| Erro residual da homografia entra em tudo | Medir com robô em pontos conhecidos e reportar como piso de erro (ver P1) |
| Dataset de um só ambiente | Coletar em mais de um local/evento; testar com camisas de outra equipe |
| Caixas derivadas, não anotadas | Declarado; mAP tratado como secundário |
| Ground truth anotado por humano tem erro próprio | Anotação em cenas estáticas com posições medidas fisicamente |
| Hardware diferente entre pontos da varredura | `sweep` grava o `device`; manter a máquina fixa é responsabilidade do operador |
| Poucas repetições nas partidas da P2 | Usar latência injetada com tarefa controlada como evidência principal |

## 11. Fluxo completo

```bash
# 1. calibrar câmera e cores na condição de referência
make camera-service && make calibrate-camera && make calibrate-colors

# 2. anotar as cenas (centro em metros, ângulo, conditions com lux)
#    ver datasets/vision/README.md

# 3. exportar para o formato Ultralytics (divisão por origem)
make export-yolo

# 4. treinar os dois braços, por variante
make train-yolo YOLO_MODEL=yolov8n-pose.pt
make train-yolo-sem-augmentation YOLO_MODEL=yolov8n-pose.pt

# 5. avaliar em escala real, mesmas cenas para todos
uv run vsss-vision compare --csv experiments/results/resumo.csv

# 6. curva precisão × latência (P2)
make sweep MODELS="runs/.../best.pt outra/best.pt" IMGSZ="320 480 640"

# 7. validar ao vivo, mesma instrumentação
make live
```
