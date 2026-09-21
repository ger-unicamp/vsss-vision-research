# Proposta P1 — Robustez à iluminação

## Contribuição

Primeira avaliação quantitativa, **em escala real**, de YOLO e de segmentação
por cor em VSSS sob condições de iluminação **medidas** e controladas,
incluindo o **esforço de calibração** de cada método. Estende diretamente o TCC
da UnB (Marques, 2025), que não testou variação de iluminação e reportou erro em
pixels.

## Hipóteses

- **H1.** O erro do detector por cor cresce com o desvio da iluminância em
  relação à condição em que foi calibrado; o do detector neural cresce menos.
- **H2.** Iluminação **não uniforme** (sombras, reflexos) degrada o método
  clássico mais que a redução uniforme de iluminância — o problema é o gradiente
  dentro do frame, não o nível absoluto.
- **H3.** Recalibrar por condição recupera boa parte do desempenho do método
  clássico, a um custo de tempo humano que é ele próprio mensurável e não
  desprezível em contexto de competição.

H3 é o que impede o artigo de ser um espantalho: comparar YOLO com um pipeline
clássico mal calibrado não prova nada.

## Desenho experimental

### Variáveis independentes

| Variável | Níveis |
| :--- | :--- |
| Iluminância (lux, medida com luxímetro no plano do campo) | **200, 400, 800, 1500** — a faixa típica de competição de futebol de robôs vai de 400 a 1500 lux, então 200 sonda abaixo do praticável e 1500 o teto |
| Tipo de luz | fria, quente |
| Uniformidade | uniforme; não uniforme (sombra); não uniforme (reflexo) |
| Método | `color` (calibração única); `color` (recalibrado por condição); YOLO com braço de *augmentation* fotométrico; YOLO sem *augmentation* |

Os dois braços de YOLO saem de `tools/train_yolo.py --augment brightness|none`.
Rodar só o braço com *augmentation* e comparar com o clássico não separa
ganho de arquitetura de ganho de *augmentation*.

Registrar cada nível no bloco `conditions` do ground truth (ver
`datasets/vision/README.md`) — é o que permite ao `bench` agrupar os resultados
por condição automaticamente.

### Dois protocolos para o método clássico

1. **Calibração única.** Calibrar na condição de referência e não tocar mais.
   Mede a *degradação* ao longo das condições.
2. **Recalibração por condição.** Recalibrar antes de cada condição,
   cronometrando a sessão. Mede o *esforço* — minutos de operador — necessário
   para manter o desempenho.

O segundo protocolo é o que dá substância ao argumento prático: se recalibrar
custa 12 minutos por condição de luz, isso é um custo real numa competição.

### Variáveis dependentes

Todas produzidas por `vsss-vision bench` (ver `benchmark/metrics.py`):

- erro de posição (cm): média, p95, máximo;
- erro de orientação (graus): média, p95;
- taxa de detecções perdidas;
- taxa de identificações trocadas;
- precisão, revocação e F1;
- (protocolo 2) tempo de calibração, em minutos, por condição.

Definições e denominadores em [`methodology.md`](methodology.md), seção 6.
As duas correções em relação ao trabalho de referência — distância
euclidiana em vez de soma com sinal, e diferença angular circular em vez de
subtração crua — mudam os números do baseline clássico e precisam ser
declaradas explicitamente no artigo, porque tornam os resultados **não
diretamente comparáveis** com os reportados lá.

### Linhas de base

- **Ballnet Pose** (`ayssag/BallnetPose`, Hugging Face) — modelo público do
  trabalho de referência. Avaliável sem re-treino, mas com duas ressalvas a
  declarar: as classes são por robô (`robot0/1/2`, camisas da UnBall), então
  ele **não detecta adversário nenhum**, e foi treinado com camisas de outra
  equipe. Configurar via `yolo_class_map` (ver
  [`methodology.md`](methodology.md), seção 1) e reportar a taxa de
  detecções perdidas separadamente para robôs próprios e adversários — do
  contrário o número fica ilegível.
- **Sistema de visão atual do GER** — `ColorDetector` deste repositório, o
  mesmo código que roda em competição, recalibrado na condição de
  referência (ver [`methodology.md`](methodology.md), seção 8).

## Requisitos de bancada

- Campo no padrão da categoria: **150 × 130 cm**; câmera fixa acima do campo, a
  no mínimo 2 m, conforme as regras.
- Luxímetro; fontes de luz com intensidade ajustável.
- **Ground truth em escala real**: cenas estáticas com posições e ângulos
  medidos fisicamente, ou anotação cuidadosa com homografia calibrada.
- Dataset próprio anotado com as camisas do GER; GPU para treino.
- Pipeline clássico **bem calibrado** — ver os dois protocolos acima.

### Sobre a homografia

`vision/geometry.py` assume que o frame retificado cobre exatamente o retângulo
físico do campo, isto é, que o ROI de 4 pontos foi posicionado nos cantos
reais. Essa hipótese precisa ser **verificada antes de qualquer coleta**, não
assumida: posicionar um robô em pontos conhecidos do campo (cantos, centro,
marcas de pênalti), comparar a saída do pipeline com a medida física e
registrar o erro residual. Esse erro é o piso do erro de posição de qualquer
detector — nenhum resultado pode ser menor que ele, e ele deve constar do
artigo.

## Coleta

1. Verificar a homografia (acima) e registrar o erro residual.
2. Para cada combinação de condição:
   a. medir a iluminância no plano do campo, em pelo menos 5 pontos;
   b. (protocolo 2) recalibrar as faixas HSV, cronometrando;
   c. capturar as cenas e anotar o ground truth com `units: "m"` e o bloco
      `conditions` preenchido.
3. Rodar `uv run vsss-vision compare --dataset datasets/vision` e versionar os
   JSONs citados no artigo junto do commit que os gerou.

## Análise

- Curva erro × iluminância, uma série por método. O eixo X vem de
  `conditions.illuminance_lux`; `RunResult.by_illuminance()` já produz o
  agrupamento.
- Comparar a **inclinação** das curvas, não apenas os valores absolutos: a
  afirmação de interesse é sobre robustez, isto é, sobre a derivada.
- Reportar o esforço de calibração como uma coluna própria, em minutos.
- Reportar o erro residual da homografia como piso de todas as curvas.

## Onde publicar

- **SBAI 2027** (SBA) — automação inteligente; aceita português. Edição anterior
  em jul/ago de 2025; datas de 2027 ainda não divulgadas.
- **SBR 2027** (SBC) — visão em robótica e futebol de robôs no escopo; revisão
  duplo-cega, apresentação em inglês, só presencial. Datas a confirmar.
