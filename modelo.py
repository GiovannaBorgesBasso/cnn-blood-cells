"""
modelo.py — Arquitetura da CNN
================================
Define os blocos e o modelo completo.
Este arquivo não treina nem avalia nada — só descreve a estrutura da rede.

É importado por treino.py e metricas.py assim:
    from modelo import BloodCNN
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────
# BLOCO 1 — SEBlock (Squeeze-and-Excitation)
# ─────────────────────────────────────────────────────────────────────
# Problema que resolve: nem todos os canais (filtros) de uma camada
# são igualmente úteis para a decisão atual.
#
# Solução: para cada canal, calcula um número entre 0 e 1
# que representa sua "importância" — e reescala o canal por esse valor.
#
# É como a rede perguntar: "desses 64 mapas de feature que tenho,
# quais realmente importam para identificar esse linfócito?"
#
# Paper: Hu et al., 2018 — ganhou o ImageNet naquele ano.

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        """
        channels  : número de canais de entrada (ex: 64)
        reduction : fator de compressão da camada intermediária
                    (64 canais → 8 → 64)
        """
        super().__init__()
        self.se = nn.Sequential(
            # "Squeeze": resume cada mapa de feature num único número
            # (média global de cada canal) → shape: (batch, channels, 1, 1)
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),  # → (batch, channels)

            # Camada densa pequena que aprende quais canais importam
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),

            # Sigmoid: saída entre 0 e 1 para cada canal
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Calcula o peso de cada canal e reescala
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale  # multiplicação canal a canal


# ─────────────────────────────────────────────────────────────────────
# BLOCO 2 — ResBlock (Bloco Residual)
# ─────────────────────────────────────────────────────────────────────
# Problema que resolve: em redes muito profundas, o gradiente
# enfraquece ao se propagar pelas camadas — "vanishing gradient".
# Os pesos das primeiras camadas praticamente param de aprender.
#
# Solução: adiciona um "atalho" (skip connection) que soma a entrada
# diretamente à saída do bloco:
#
#     saída = F(x) + x
#
# Isso garante que o gradiente sempre tem um caminho direto de volta,
# independente de quantas camadas existam.
#
# Paper: He et al., 2015 (ResNet) — revolucionou visão computacional.

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Caminho principal: duas convoluções com BN e ReLU
        self.path = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),  # normaliza as ativações → treino mais estável
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # SE após o caminho principal
        self.se   = SEBlock(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x               # guarda a entrada original (o "atalho")
        out = self.path(x)         # passa pelas convoluções
        out = self.se(out)         # recalibra os canais
        out = out + residual       # ← skip connection: soma com a entrada
        return self.relu(out)


# ─────────────────────────────────────────────────────────────────────
# MODELO COMPLETO — BloodCNN
# ─────────────────────────────────────────────────────────────────────
# Entrada: imagem RGB 3 × 28 × 28
# Saída:   vetor de 8 logits (um por classe de célula)
#
# Pipeline:
#   Stem → Stage1 → Stage2 → Stage3 → Cabeça de classificação

class BloodCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        # Stem: primeira convolução — extrai features iniciais
        # Entrada: 3 × 28 × 28  →  Saída: 32 × 28 × 28
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage 1: refina features e reduz resolução pela metade
        # 32 × 28 × 28  →  64 × 14 × 14
        self.stage1 = nn.Sequential(
            ResBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 2: features mais abstratas, resolução cai de novo
        # 64 × 14 × 14  →  128 × 7 × 7
        self.stage2 = nn.Sequential(
            ResBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Stage 3: dois blocos residuais sem reduzir resolução
        # 128 × 7 × 7  →  128 × 7 × 7
        # (este é o último mapa de features — usado pelo Grad-CAM)
        self.stage3 = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
        )

        # Cabeça de classificação
        self.head = nn.Sequential(
            # Global Average Pooling: resume cada mapa 7×7 num único número
            # 128 × 7 × 7  →  128
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            # Dropout: durante o treino, desativa 40% dos neurônios aleatoriamente
            # Força a rede a não depender de poucos neurônios → menos overfitting
            nn.Dropout(0.4),

            # Camada final: 128 → 8 (uma saída por classe)
            nn.Linear(128, num_classes),
        )

        # Hooks para Grad-CAM (usados em metricas.py)
        self.gradients   = None
        self.activations = None

    def _salva_gradiente(self, grad):
        self.gradients = grad

    def forward(self, x, cam=False):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        if cam:
            # Registra as ativações e gradientes para o Grad-CAM
            x.register_hook(self._salva_gradiente)
            self.activations = x

        return self.head(x)


# ─────────────────────────────────────────────────────────────────────
# SANITY CHECK — rode este arquivo diretamente para verificar
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = BloodCNN(num_classes=8)

    # Conta parâmetros
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parâmetros treináveis: {total:,}")

    # Passa uma imagem falsa para verificar os shapes
    x = torch.randn(1, 3, 28, 28)  # batch=1, RGB, 28×28
    out = model(x)
    print(f"Shape de entrada : {x.shape}")
    print(f"Shape de saída   : {out.shape}  ← deve ser [1, 8]")
    print("\nModelo ok! Pronto para ser importado por treino.py")