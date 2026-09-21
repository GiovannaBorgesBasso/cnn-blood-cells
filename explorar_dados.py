"""
explorar_dados.py — Visualização do BloodMNIST
================================================
Rode este arquivo ANTES de treinar para entender o dataset.
Não depende de modelo treinado — só carrega os dados.

Gera:
    outputs/exemplos_celulas.png     — uma célula de cada tipo
    outputs/distribuicao_classes.png — quantas imagens por classe
    outputs/grid_variacoes.png       — variações dentro de uma mesma classe
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms
from medmnist import BloodMNIST

os.makedirs("outputs", exist_ok=True)

CLASS_NAMES = [
    "Basófilo", "Eosinófilo", "Eritroblasto", "Gran. imaturo",
    "Linfócito", "Monócito", "Neutrófilo", "Plaqueta",
]

# Descrição curta de cada célula (2 linhas para caber abaixo da imagem)
CLASS_DESC = [
    "Grânulos escuros\nraro no sangue normal",
    "Grânulos alaranjados\nresposta a alérgenos",
    "Precursor da hemácia\nnúcleo ainda presente",
    "Granulócito imaturo\nnúcleo não segmentado",
    "Sistema imune\nnúcleo grande e redondo",
    "Maior leucócito\nnúcleo em forma de rim",
    "Leucócito abundante\nelimina bactérias",
    "Fragmento celular\ncoagulação sanguínea",
]

# Carrega sem augmentation para ver as imagens reais
transform = transforms.Compose([transforms.ToTensor()])

train_ds = BloodMNIST(split="train", transform=transform, download=True)
test_ds  = BloodMNIST(split="test",  transform=transform, download=True)

print(f"Dataset carregado: {len(train_ds)} treino | {len(test_ds)} teste")

# Organiza índices por classe
indices_por_classe = {c: [] for c in range(8)}
for i in range(len(train_ds)):
    label = int(train_ds[i][1].item())
    indices_por_classe[label].append(i)

print("\nDistribuição no treino:")
for c in range(8):
    n = len(indices_por_classe[c])
    bar = "█" * (n // 100)
    print(f"  {CLASS_NAMES[c]:20s}: {n:5d}  {bar}")


# ─────────────────────────────────────────────
# FIGURA 1 — Um exemplo de cada célula
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 8, figsize=(20, 5.5))
fig.suptitle("BloodMNIST — 8 tipos de células sanguíneas\n"
             "Microscopia de sangue periférico · Acevedo et al., 2020",
             fontsize=13, fontweight="bold")
fig.subplots_adjust(top=0.82)

# Cores para cada classe (para identidade visual consistente)
CORES = ["#6366f1","#f59e0b","#ef4444","#10b981",
         "#3b82f6","#8b5cf6","#14b8a6","#f97316"]

for c, ax in enumerate(axes):
    idx = indices_por_classe[c][0]
    img, _ = train_ds[idx]
    img_np = img.permute(1, 2, 0).numpy()

    ax.imshow(img_np)
    ax.set_title(CLASS_NAMES[c], fontsize=9, fontweight="bold",
                 color=CORES[c], pad=5)
    ax.set_xlabel(CLASS_DESC[c], fontsize=7.5, color="#4b5563",
                  labelpad=6, linespacing=1.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Borda colorida
    for spine in ax.spines.values():
        spine.set_edgecolor(CORES[c])
        spine.set_linewidth(2.5)
        spine.set_visible(True)

plt.tight_layout()
plt.savefig("outputs/exemplos_celulas.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n→ salvo: outputs/exemplos_celulas.png")


# ─────────────────────────────────────────────
# FIGURA 2 — Distribuição de classes
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle("Distribuição de classes — BloodMNIST treino\n"
             "Desbalanceamento moderado: Neutrófilo (2330) tem ~3× mais amostras que Basófilo (852)",
             fontsize=12, fontweight="bold")

contagens = [len(indices_por_classe[c]) for c in range(8)]
bars = ax.bar(CLASS_NAMES, contagens, color=CORES, edgecolor="white", linewidth=1.2)

# Linha da média
media = np.mean(contagens)
ax.axhline(media, color="#374151", ls="--", lw=1.5,
           label=f"Média: {media:.0f} imagens/classe")

# Anotações em cima de cada barra
for bar, n in zip(bars, contagens):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            str(n), ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Número de imagens", fontsize=11)
ax.set_ylim(0, max(contagens) * 1.18)
ax.tick_params(axis="x", labelsize=9)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis="y")
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/distribuicao_classes.png", dpi=150, bbox_inches="tight")
plt.show()
print("→ salvo: outputs/distribuicao_classes.png")


# ─────────────────────────────────────────────
# FIGURA 3 — Variações dentro de uma classe
# (mostra por que augmentation é necessária)
# ─────────────────────────────────────────────
CLASSE_DESTAQUE = 6  # Neutrófilo — mais abundante, fácil de reconhecer

fig, axes = plt.subplots(2, 8, figsize=(18, 5))
fig.suptitle(f"16 exemplos de {CLASS_NAMES[CLASSE_DESTAQUE]} — mesma classe, variações naturais\n"
             "A rede precisa reconhecer todas como a mesma célula",
             fontsize=11, fontweight="bold")

exemplos = indices_por_classe[CLASSE_DESTAQUE][:16]
for i, idx in enumerate(exemplos):
    row, col = divmod(i, 8)
    img, _ = train_ds[idx]
    axes[row, col].imshow(img.permute(1, 2, 0).numpy())
    axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("outputs/grid_variacoes.png", dpi=150, bbox_inches="tight")
plt.show()
print("→ salvo: outputs/grid_variacoes.png")

print("\nExplore outros datasets MedMNIST em: https://medmnist.com")
print("Paper original BloodMNIST: Acevedo et al. (2020), Data in Brief")
print("Benchmark arquiteturas: Yang et al. (2023), Nature Scientific Data")