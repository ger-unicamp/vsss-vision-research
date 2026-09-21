"""vsss_vision — pipeline de visão para Very Small Size Soccer, em versão de pesquisa.

Derivado do subsistema de visão de `ger-unicamp/futebol-vsss` (repositório
de competição). Aqui o objetivo não é jogar, e sim **medir**: comparar
abordagens de detecção (segmentação por cor vs. redes neurais) sob
condições controladas, em unidades físicas e com latência instrumentada.

Pacotes:
  `camera`    — aquisição, correção de perspectiva, auto-tuning (inalterado).
  `vision`    — detectores plugáveis, suavização temporal, conversão px -> m.
  `benchmark` — dataset com ground truth, métricas de erro, latência, relatórios.

Ver `README.md` e `docs/research/` para o desenho experimental.
"""

__version__ = "0.1.0"
