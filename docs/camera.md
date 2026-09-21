# Módulo de Câmera

Este módulo é responsável pela aquisição bruta de imagem, ajuste de hardware e correção de perspectiva (recorte) do campo de jogo. Roda como um serviço independente, desacoplando a captura de câmera (dependente de hardware) do processamento de visão de alto nível.

## 🚀 Visão Geral

O módulo de câmera segue um pipeline:
**Captura de Hardware** $\rightarrow$ **Transformação de Perspectiva (Warp)** $\rightarrow$ **Distribuição via ZMQ**.

Ele fornece um stream em tempo real tanto do feed bruto da câmera quanto de uma visão "achatada" (vista de cima) do campo, permitindo que o módulo de visão subsequente opere em um sistema de coordenadas consistente independentemente do ângulo físico da câmera.

## 📂 Estrutura

| Arquivo | Responsabilidade |
| :--- | :--- |
| `service.py` | **Orquestrador**. Gerencia os sockets ZMQ, coordena a bridge, o recorte e o tuner, e trata comandos recebidos. |
| `camera_bridge.py` | **Camada de Hardware**. Lida com `VideoCapture` do OpenCV, descoberta de câmera (priorizando câmeras USB externas), e ajustes de propriedades de baixo nível. |
| `recorte.py` | **Processamento Geométrico**. Implementa `warpPerspective` usando seleção de ROI por 4 pontos para transformar a imagem distorcida do campo num retângulo visto de cima. |
| `tuning.py` | **Auto-Otimização**. Ajusta sequencialmente Foco $\rightarrow$ Exposição $\rightarrow$ Contraste com base em estatísticas da imagem (Média/Desvio) para garantir condições ideais de detecção. |
| `config.json` | **Persistência**. Armazena os pontos de ROI, resolução de saída e configurações de hardware para persistir entre reinicializações. |

## 📡 Protocolo de Comunicação (ZMQ)

O módulo expõe dois sockets ZMQ para comunicação assíncrona.

### 1. Publisher (PUB) - `tcp://*:5555`
Usado para streaming de dados em alta frequência.

| Tópico | Formato dos Dados | Descrição |
| :--- | :--- | :--- |
| `raw` | binário `.jpg` | O frame de imagem original da câmera. |
| `cropped` | binário `.jpg` | A imagem do campo, vista de cima, com a perspectiva corrigida. |
| `metrics` | JSON | Estatísticas em tempo real: `{"mean": float, "std": float, "phase": string/null}`. |
| `settings` | JSON | Configurações de hardware após um ciclo de auto-tuning. |

### 2. Request-Reply (REP) - `tcp://*:5556`
Usado para comandos de configuração e controle.

**Formato da Requisição:** `{"cmd": "nome_do_comando", ...args}`

| Comando | Argumentos | Descrição |
| :--- | :--- | :--- |
| `set_roi` | `{"points": [[x,y], ...]}` | Atualiza os 4 pontos de origem para a transformação de perspectiva. |
| `set_property` | `{"prop": int, "val": int}` | Define uma propriedade específica de hardware do OpenCV. |
| `save` | N/A | Persiste as configurações e o ROI atuais em `config.json`. |
| `reset_roi` | N/A | Limpa os pontos de ROI atuais. |
| `auto_config` | N/A | Dispara o processo sequencial de auto-tuning. |
| `get_status` | N/A | Retorna as configurações de hardware e o ROI atuais. |

## ⚙️ Lógica de Auto-Tuning

O `CameraTuner` garante que a imagem não fique nem muito escura nem muito clara, e que tenha contraste suficiente para a segmentação de cor.

### Sequência de Tuning
1. **Reset**: Brilho, Contraste e Saturação são resetados para 128.
2. **Foco**: O autofoco de hardware é ativado por um curto período.
3. **Exposição**: Ajusta `CAP_PROP_EXPOSURE` até que o brilho médio (Média) entre na faixa alvo.
4. **Contraste**: Ajusta `CAP_PROP_CONTRAST` até que a variância da imagem (Desvio) entre na faixa alvo.

### Faixas Alvo
- **Média (Brilho)**: $127.5 \pm 5.5$ $\rightarrow$ **[122, 133]**
- **Desvio (Contraste)**: $55 \pm 5$ $\rightarrow$ **[50, 60]**

## 🛠 Uso

### Rodando o Serviço
```bash
make camera-service
# Ou manualmente
uv run python -m vsss_vision.camera.service
```

### Calibração
Use a ferramenta de configuração para definir os pontos de ROI e verificar o tuning:
```bash
make calibrate-camera
# Ou manualmente
uv run python tools/camera_configurator.py
```
