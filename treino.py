"""
treino.py — Como a rede aprende
=========================

Cobre: data augmentation, loss function, otimizador, scheduler e curvas.
Precisa que modelo.py esteja na mesma pasta (importa BloodCNN de lá).

Gera (dentro de outputs/):
    augmentation_exemplos.png
    curvas_aprendizado.png
    cnn_blood_treinado.pth  ← carregado por metricas.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import BloodMNIST
import numpy as np
import matplotlib.pyplot as plt

# Importa a arquitetura definida pela Pessoa 1
from modelo import BloodCNN

os.makedirs("outputs", exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 1e-3

CLASS_NAMES = [
    "Basófilo", "Eosinófilo", "Eritroblasto", "Gran. imaturo",
    "Linfócito", "Monócito", "Neutrófilo", "Plaqueta",
]

print(f"Dispositivo: {DEVICE}")


# ════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — DATA AUGMENTATION
# "Por que aumentar os dados?"
# ════════════════════════════════════════════════════════════════════
#
# A rede aprende a partir dos exemplos que vê. Se ela sempre vê
# um neutrófilo exatamente na mesma orientação, ela não vai
# reconhecer o mesmo neutrófilo girado 90°.
#
# Augmentation cria variações artificiais de cada imagem durante
# o treino — o modelo vê a "mesma" célula em condições diferentes,
# ficando mais robusto.
#
# IMPORTANTE: augmentation só vai no treino, NUNCA no teste.
# No teste queremos medir o modelo nas imagens como elas são.

print("=" * 60)
print("SEÇÃO 1 — DATA AUGMENTATION")
print("=" * 60)

train_transform = transforms.Compose([
    # Espelha horizontalmente com 50% de chance
    # → célula vista "de outro lado" ainda é a mesma célula
    transforms.RandomHorizontalFlip(),

    # Espelha verticalmente com 50% de chance
    transforms.RandomVerticalFlip(),

    # Rotação aleatória até 15°
    # → microscópio pode capturar a célula em qualquer ângulo
    transforms.RandomRotation(15),

    # Varia brilho, contraste e saturação levemente
    # → simula variações de corante entre laboratórios diferentes
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# Teste e validação: sem augmentation — queremos a imagem real
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# Carrega os três conjuntos
train_ds = BloodMNIST(split="train", transform=train_transform, download=True)
val_ds   = BloodMNIST(split="val",   transform=test_transform,  download=True)
test_ds  = BloodMNIST(split="test",  transform=test_transform,  download=True)

print(f"\nTamanho dos conjuntos:")
print(f"  Treino    : {len(train_ds):>6} imagens  ← augmentation ativo")
print(f"  Validação : {len(val_ds):>6} imagens  ← sem augmentation")
print(f"  Teste     : {len(test_ds):>6} imagens  ← sem augmentation")

# ── Visualização: mesma célula, 8 versões aumentadas ──────────────
print("\nGerando visualização de augmentation...")

# Original (sem augmentation)
ds_orig = BloodMNIST(split="train", transform=test_transform, download=False)
img_orig = ds_orig[0][0]
classe   = int(ds_orig[0][1])

# Desfaz a normalização para exibir as cores reais
inv = transforms.Normalize([-1, -1, -1], [2, 2, 2])

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle(
    f"Data Augmentation — {CLASS_NAMES[classe]}: mesma célula, "
    "versões diferentes que o modelo vê durante o treino",
    fontsize=11, fontweight="bold",
)

axes[0, 0].imshow(inv(img_orig).permute(1, 2, 0).clamp(0, 1))
axes[0, 0].set_title("Original", fontweight="bold", color="#2563eb")
axes[0, 0].axis("off")
axes[1, 0].axis("off")
axes[1, 0].text(0.5, 0.5, f"Classe:\n{CLASS_NAMES[classe]}",
                ha="center", va="center", fontsize=10)

for i in range(8):
    row, col = divmod(i, 4)
    aug_img = train_ds[0][0]   # nova chamada → novo augmentation aleatório
    axes[row, col + 1].imshow(inv(aug_img).permute(1, 2, 0).clamp(0, 1))
    axes[row, col + 1].set_title(f"versão {i+1}", fontsize=9, color="#6b7280")
    axes[row, col + 1].axis("off")

plt.tight_layout()
plt.savefig("outputs/augmentation_exemplos.png", dpi=130, bbox_inches="tight")
plt.show()
print("→ salvo: outputs/augmentation_exemplos.png\n")


# ════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — LOSS FUNCTION
# "Como a rede sabe se errou?"
# ════════════════════════════════════════════════════════════════════
#
# A loss function mede o quanto a predição do modelo está errada.
# O treinamento inteiro é uma corrida para minimizar esse número.
#
# Para classificação multiclasse usamos Cross-Entropy Loss.
# Ela penaliza o modelo quando a probabilidade da classe CORRETA é baixa.
#
# Exemplo intuitivo:
#   - Modelo deu 95% para a classe certa → loss pequeno  ✓
#   - Modelo deu 10% para a classe certa → loss grande   ✗
#
# Matematicamente: Loss = -log(probabilidade da classe correta)

print("=" * 60)
print("SEÇÃO 2 — CROSS-ENTROPY LOSS (intuição)")
print("=" * 60)

probs = np.linspace(0.01, 0.99, 200)
losses = -np.log(probs)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(probs, losses, color="#2563eb", lw=2.5)
ax.fill_between(probs, losses, alpha=0.1, color="#2563eb")
ax.axvline(0.10, color="#dc2626", ls="--", lw=1.5,
           label="modelo inseguro (10%) → loss alto")
ax.axvline(0.95, color="#16a34a", ls="--", lw=1.5,
           label="modelo confiante (95%) → loss baixo")
ax.set_xlabel("Probabilidade atribuída à classe correta", fontsize=11)
ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
ax.set_title("Quanto mais confiante e certo, menor o loss",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 5)
plt.tight_layout()
plt.savefig("outputs/crossentropy_intuicao.png", dpi=130, bbox_inches="tight")
plt.show()
print("→ salvo: outputs/crossentropy_intuicao.png\n")

# CrossEntropyLoss já inclui o Softmax internamente —
# o modelo só precisa retornar logits (valores brutos)
criterion = nn.CrossEntropyLoss()


# ════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — OTIMIZADOR E LEARNING RATE SCHEDULER
# "Como a rede ajusta os pesos para errar menos?"
# ════════════════════════════════════════════════════════════════════
#
# Após calcular o loss, o backpropagation calcula o gradiente:
# a direção em que cada peso deve se mover para reduzir o loss.
#
# O otimizador decide o tamanho do passo nessa direção — o learning rate.
#
#   LR muito grande → pula o mínimo, não converge
#   LR muito pequeno → converge, mas demora muito
#
# AdamW é o otimizador padrão atual:
#   • adapta o LR individualmente para cada parâmetro (momentum adaptativo)
#   • weight_decay penaliza pesos grandes → reduz overfitting
#
# CosineAnnealingLR: começa com LR alto e diminui suavemente
# ao longo das épocas — exploração ampla no início, refinamento no final.

print("=" * 60)
print("SEÇÃO 3 — OTIMIZADOR + LEARNING RATE SCHEDULER")
print("=" * 60)

model     = BloodCNN(num_classes=8).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Simula as épocas para mostrar a curva do LR antes de treinar
lrs = []
temp_opt   = optim.AdamW([torch.zeros(1)], lr=LR)
temp_sched = optim.lr_scheduler.CosineAnnealingLR(temp_opt, T_max=EPOCHS)
for _ in range(EPOCHS):
    lrs.append(temp_sched.get_last_lr()[0])
    temp_sched.step()

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(range(1, EPOCHS + 1), lrs, color="#7c3aed", lw=2.5,
        marker="o", markersize=4)
ax.fill_between(range(1, EPOCHS + 1), lrs, alpha=0.15, color="#7c3aed")
ax.set_xlabel("Época", fontsize=11)
ax.set_ylabel("Learning Rate", fontsize=11)
ax.set_title("Cosine Annealing — LR alto no começo, refinamento suave no final",
             fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/learning_rate_schedule.png", dpi=130, bbox_inches="tight")
plt.show()
print("→ salvo: outputs/learning_rate_schedule.png\n")


# ════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — LOOP DE TREINAMENTO
# "O que acontece a cada época?"
# ════════════════════════════════════════════════════════════════════
#
# Uma época = o modelo viu todas as imagens de treino uma vez.
# A cada batch (lote de 64 imagens):
#   1. Forward pass:   imagem → predição
#   2. Loss:           mede o erro
#   3. Backward pass:  calcula gradientes (backpropagation)
#   4. Optimizer step: atualiza os pesos
#
# Ao final de cada época, avaliamos no conjunto de VALIDAÇÃO
# (imagens que o modelo nunca viu) para saber se está generalizando.

print("=" * 60)
print("SEÇÃO 4 — LOOP DE TREINAMENTO")
print("=" * 60)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

history = {"train_loss": [], "val_loss": [], "val_acc": []}

for epoch in range(EPOCHS):
    # ── TREINO ──────────────────────────────────────────────────────
    model.train()   # ativa Dropout e BatchNorm no modo treino
    total_loss = 0.0

    for imgs, labels in train_loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.squeeze().long().to(DEVICE)

        optimizer.zero_grad()               # zera gradientes do batch anterior
        outputs = model(imgs)               # 1. forward pass
        loss    = criterion(outputs, labels) # 2. calcula o erro
        loss.backward()                     # 3. backpropagation
        optimizer.step()                    # 4. atualiza os pesos
        total_loss += loss.item()

    # ── VALIDAÇÃO ───────────────────────────────────────────────────
    model.eval()    # desativa Dropout; BatchNorm usa estatísticas fixas
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():   # sem cálculo de gradientes → mais rápido
        for imgs, labels in val_loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)
            out    = model(imgs)
            val_loss += criterion(out, labels).item()
            correct  += (out.argmax(1) == labels).sum().item()
            total    += labels.size(0)

    scheduler.step()    # ajusta o LR para a próxima época

    tl = total_loss / len(train_loader)
    vl = val_loss   / len(val_loader)
    ac = correct / total

    history["train_loss"].append(tl)
    history["val_loss"].append(vl)
    history["val_acc"].append(ac)

    print(f"  Época {epoch+1:02d}/{EPOCHS} | "
          f"loss treino: {tl:.4f} | loss val: {vl:.4f} | acc val: {ac:.2%}")

# Salva os pesos — a Pessoa 3 vai carregar esse arquivo
torch.save(model.state_dict(), "outputs/cnn_blood_treinado.pth")
print("\n→ Modelo salvo em: outputs/cnn_blood_treinado.pth")
print("   (metricas.py vai carregar esse arquivo para as métricas finais)\n")


# ════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — CURVAS DE APRENDIZADO
# "O que essas curvas nos dizem?"
# ════════════════════════════════════════════════════════════════════
#
# As curvas de loss são o "diário" do treinamento.
# Elas revelam se o modelo está aprendendo bem ou com problemas.
#
# Padrão saudável:
#   → loss de treino e validação caem juntos
#   → acurácia de val sobe de forma estável
#
# Overfitting (decoreba):
#   → loss treino cai, mas val sobe ou para de cair
#   → o modelo memorizou o treino em vez de generalizar
#
# Underfitting:
#   → ambas as losses ficam altas
#   → modelo muito simples ou muito poucas épocas

print("=" * 60)
print("SEÇÃO 5 — CURVAS DE APRENDIZADO")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Curvas de Aprendizado — o que o modelo aprendeu ao longo das épocas",
    fontsize=12, fontweight="bold",
)

# ── Loss ──────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(history["train_loss"], label="Treino",    color="#2563eb", lw=2)
ax.plot(history["val_loss"],   label="Validação", color="#dc2626", lw=2)
ax.set_title("Cross-Entropy Loss")
ax.set_xlabel("Época")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# Diagnóstico automático
gap = history["train_loss"][-1] - history["val_loss"][-1]
if abs(gap) < 0.1:
    nota = "✓ treino e val próximos → boa generalização"
elif gap < -0.1:
    nota = "⚠ val < treino → pode aumentar epochs"
else:
    nota = "⚠ val > treino → possível overfitting"

ax.text(0.05, 0.05, nota, transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9fafb",
                  edgecolor="#d1d5db"))

# ── Acurácia ──────────────────────────────────────────────────────
ax = axes[1]
ax.plot(history["val_acc"], color="#16a34a", lw=2, marker="o", markersize=3)
ax.axhline(1/8, color="#9ca3af", ls="--", lw=1.2,
           label=f"Aleatório: {1/8:.0%} (8 classes)")
ax.set_title("Acurácia na Validação")
ax.set_xlabel("Época")
ax.set_ylabel("Acurácia")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)

# Marca a melhor época
best_epoch = int(np.argmax(history["val_acc"]))
best_acc   = history["val_acc"][best_epoch]
offset = min(best_epoch + 2, EPOCHS - 1)
ax.annotate(
    f"melhor: {best_acc:.2%}\n(época {best_epoch + 1})",
    xy=(best_epoch, best_acc),
    xytext=(offset, best_acc - 0.12),
    arrowprops=dict(arrowstyle="->", color="#374151"),
    fontsize=9, color="#374151",
)

plt.tight_layout()
plt.savefig("outputs/curvas_aprendizado.png", dpi=150, bbox_inches="tight")
plt.show()
print("→ salvo: outputs/curvas_aprendizado.png")

print("\n" + "=" * 60)
print("RESUMO — o que esta seção apresentou:")
print("=" * 60)
n_aug = len([t for t in train_transform.transforms
             if not isinstance(t, (transforms.ToTensor, transforms.Normalize))])
print(f"  • Data augmentation : {n_aug} transformações aplicadas só no treino")
print(f"  • Loss function     : CrossEntropyLoss (multiclasse, inclui Softmax)")
print(f"  • Otimizador        : AdamW (lr={LR}, weight_decay=1e-4)")
print(f"  • Scheduler         : CosineAnnealingLR por {EPOCHS} épocas")
print(f"  • Melhor acc val    : {max(history['val_acc']):.2%}"
      f" (época {int(np.argmax(history['val_acc'])) + 1})")
print(f"\n  Arquivo para a Pessoa 3: outputs/cnn_blood_treinado.pth")