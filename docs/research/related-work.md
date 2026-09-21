# Trabalhos relacionados e posicionamento

## Trabalho de referência direta

**Marques, A. G. O. (2025).** *Detecção de objetos Very Small Size Soccer
utilizando redes neurais convolucionais YOLO.* TCC, Ciência da Computação, UnB.
Orientador: Prof. Marcus Vinicius Lamar.

- Treinou **YOLOv8-pose** (variantes nano a extra-large) com keypoints de frente
  e trás de cada robô, o que permite calcular a orientação.
- Comparou com o **Main System** da equipe UnBall (Python/OpenCV, segmentação
  por faixas HSV).
- O **YOLOv8x-pose** (69,4 M de parâmetros) teve menor erro de posição e de
  orientação que o sistema clássico.
- Dataset e modelo públicos: **`ballnet_dataset`** (Roboflow Universe) e
  **`ayssag/BallnetPose`** (Hugging Face). Servem de linha de base aqui.

### Limitações declaradas ou observáveis (lacunas exploráveis)

- Processamento **apenas offline** (arquivos de imagem/vídeo); sem detecção em
  tempo real, sem medição de FPS ou latência.
- **Métricas em pixels**; o próprio autor aponta que não bastam para avaliar
  erro em escala real no campo.
- **Sem teste sob variação de iluminação** e sem *data augmentation*.
- **Base pequena e pouco diversa**: 653 imagens (85 de teste), de um único
  evento (IronCup 2020) e com as camisas de uma única equipe.
- Parte da avaliação foi **qualitativa/visual**.
- **Não avalia o efeito da visão** sobre controle, estratégia ou desempenho em
  jogo.

## Busca de literatura pendente

Antes de escrever, fazer uma busca mais ampla para posicionar cada contribuição
e evitar reivindicar ineditismo indevido:

- **IEEE Xplore** — visão com *deep learning* em robótica móvel e em futebol de
  robôs;
- **SOL/SBC** — anais de SBR, SBAI, CBA; trabalhos brasileiros de VSSS e IEEE
  Very Small;
- **Anais do RoboCup** (Symposium e *team description papers*) — sobretudo da
  **SSL**, onde a visão global com câmera fixa é o padrão há mais tempo e boa
  parte dos problemas de calibração e latência já foi enfrentada;
- Trabalhos de **latência em laços de percepção-ação** fora do futebol de robôs
  (condução autônoma, teleoperação) para embasar a metodologia da P2.

Pontos a verificar na busca:

1. Alguém já mediu robustez à iluminação em VSSS/SSL **com iluminância medida**
   em lux? Se sim, a contribuição da P1 muda de "primeira avaliação" para
   "avaliação com escala real e esforço de calibração".
2. Existe trabalho relacionando latência de percepção a desempenho em tarefa no
   futebol de robôs? É o coração da P2.
3. Há dataset público de VSSS além do `ballnet_dataset`? Usar mais de um
   reforça a validade externa dos resultados.

## Como este repositório se posiciona

Ver [`p1-illumination.md`](p1-illumination.md) e [`p2-latency.md`](p2-latency.md)
para o desenho experimental completo. O resumo é que cada limitação da lista
acima vira ou uma variável medida, ou um requisito de infraestrutura já
implementado aqui (métricas em cm/graus, latência instrumentada, condições no
ground truth, comparação sob a mesma calibração).
