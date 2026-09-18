# CNN para Classificação de Células Sanguíneas

Implementação didática de uma CNN com **Residual Blocks**, **Squeeze-and-Excitation** e **Grad-CAM** aplicada ao dataset [BloodMNIST](https://medmnist.com) — 8 tipos de células sanguíneas, ~12k imagens, balanceado.

Desenvolvido para o grupo de estudos de Visão Computacional · UFG.

---

## Estrutura do projeto

```
cnn-blood-cells/
├── modelo.py     # Arquitetura da CNN (SEBlock, ResBlock, BloodCNN)
├── treino.py     # Dataset, augmentation, loop de treinamento, curvas
├── metricas.py   # Avaliação: métricas, matriz de confusão, Grad-CAM
├── outputs/      # Gerado automaticamente (gráficos + modelo salvo)
├── data/         # Gerado automaticamente (download do MedMNIST)
└── .gitignore
```

### Divisão de responsabilidades

| Arquivo | Responsável | Conteúdo |
|---|---|---|
| `modelo.py` | Pessoa 1 | O que é a CNN, blocos residuais, SE attention |
| `treino.py` | Pessoa 2 | Augmentation, loss, otimizador, curvas de aprendizado |
| `metricas.py` | Pessoa 3 | Accuracy, F1, matriz de confusão, Grad-CAM |

---

## Dataset — BloodMNIST

8 classes de células sanguíneas humanas vistas em microscópio:

| Índice | Classe | Descrição |
|---|---|---|
| 0 | Basófilo | Granulócito com grânulos grandes e escuros |
| 1 | Eosinófilo | Granulócito com grânulos alaranjados |
| 2 | Eritroblasto | Precursor das hemácias, núcleo visível |
| 3 | Gran. imaturo | Granulócitos em desenvolvimento |
| 4 | Linfócito | Célula do sistema imune, núcleo grande |
| 5 | Monócito | Maior leucócito, núcleo em formato de rim |
| 6 | Neutrófilo | Granulócito mais abundante no sangue |
| 7 | Plaqueta | Fragmento celular, muito pequeno |

- **Treino:** 11.959 imagens · **Validação:** 1.712 · **Teste:** 3.421
- Imagens RGB 28×28 pixels
- Dataset balanceado (~1.500 por classe no treino)

---

## Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/SEU_USUARIO/cnn-blood-cells.git
cd cnn-blood-cells

# 2. Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install torch torchvision medmnist scikit-learn matplotlib seaborn
```

---

## Como executar

Os arquivos devem ser rodados **nessa ordem**:

```bash
# Passo 1 — define a arquitetura (sem saída visual, só importações)
python modelo.py

# Passo 2 — treina e salva o modelo
python treino.py
# → gera: outputs/cnn_blood_treinado.pth
# → gera: outputs/augmentation_exemplos.png
# → gera: outputs/curvas_aprendizado.png

# Passo 3 — avalia o modelo treinado
python metricas.py
# → gera: outputs/metricas_blood.png
# → gera: outputs/gradcam_blood.png
```

---

## Arquitetura

```
Input (3 × 28 × 28)
    │
    ▼
Stem Conv (32 filtros)
    │
    ▼
Stage 1: ResBlock(32) → Conv stride 2 → 64 filtros  [14×14]
    │
    ▼
Stage 2: ResBlock(64) → Conv stride 2 → 128 filtros  [7×7]
    │
    ▼
Stage 3: ResBlock(128) → ResBlock(128)               [7×7]
    │
    ▼
Global Average Pooling → Dropout(0.4) → Linear(128→8)
    │
    ▼
Output (8 classes)
```

**ResBlock** = Conv → BN → ReLU → Conv → BN → SE → + skip connection → ReLU

**SEBlock** = comprime cada channel num número (squeeze) → aprende peso 0–1 pra cada channel (excitation) → reescala

---

## Resultados esperados

| Métrica | Valor aproximado |
|---|---|
| Acurácia (teste) | ~93–96% |
| F1 Macro | ~93–95% |
| F1 Micro | ~93–96% |
| Baseline aleatório | 12.5% (1/8) |

---

## Referências

- Yang et al. (2023). [MedMNIST v2](https://www.nature.com/articles/s41597-022-01721-8) — Nature Scientific Data
- He et al. (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — ResNet
- Hu et al. (2018). [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) — SENet
- Selvaraju et al. (2017). [Grad-CAM](https://arxiv.org/abs/1610.02391) — Visual Explanations from Deep Networks
