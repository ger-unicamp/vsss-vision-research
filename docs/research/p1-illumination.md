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
| Iluminância (lux, medida com luxímetro no plano do campo) | ≥ 4 níveis cobrindo do mínimo de competição ao excesso |
| Tipo de luz | fria, quente |
| Uniformidade | uniforme; não uniforme (sombra); não uniforme (reflexo) |
| Método | `color` (calibração única); `color` (recalibrado por condição); YOLO treinado com *augmentation* de brilho/contraste; YOLO sem *augmentation* |

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
- (protocolo 2) tempo de calibração, em minutos, por condição.

### Linhas de base

- **Ballnet Pose** (`ayssag/BallnetPose`, Hugging Face) — modelo público do
  trabalho de referência, avaliado sem re-treino sobre as cenas deste dataset.
- **Sistema de visão atual do GER** — `ColorDetector` deste repositório, que é
  o mesmo código que roda em competição.

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
