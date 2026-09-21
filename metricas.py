"""
metricas.py — Avaliação do modelo treinado
===========================================

Precisa que treino.py já tenha rodado e gerado:
    outputs/cnn_blood_treinado.pth

Gera (dentro de outputs/):
    metricas_resumo.png     — accuracy, precision, recall, F1 por classe
    matriz_confusao.png     — onde o modelo erra e acerta
    gradcam_exemplos.png    — o que a rede "olha" para decidir
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
from medmnist import BloodMNIST
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
import warnings
warnings.filterwarnings("ignore")

from modelo import BloodCNN

os.makedirs("outputs", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Basófilo", "Eosinófilo", "Eritroblasto", "Gran. imaturo",
    "Linfócito", "Monócito", "Neutrófilo", "Plaqueta",
]
CORES = [
    "#6366f1", "#f59e0b", "#ef4444", "#10b981",
    "#3b82f6", "#8b5cf6", "#14b8a6", "#f97316",
]

# ── Carrega modelo treinado ───────────────────────────────────────
model = BloodCNN(num_classes=8).to(DEVICE)
model.load_state_dict(torch.load("outputs/cnn_blood_treinado.pth",
                                  map_location=DEVICE))
model.eval()
print(f"Modelo carregado · dispositivo: {DEVICE}")

# ── Dataset de teste (sem augmentation) ──────────────────────────
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
test_ds = BloodMNIST(split="test", transform=test_transform, download=True)
loader  = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

print(f"Conjunto de teste: {len(test_ds)} imagens\n")


# ════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — COLETANDO AS PREDIÇÕES
# ════════════════════════════════════════════════════════════════════
#
# Antes de calcular qualquer métrica, precisamos rodar o modelo
# em TODAS as imagens de teste e guardar:
#   • y_true : o rótulo correto (dado pelo dataset)
#   • y_pred : o que o modelo achou que era
#
# O modelo retorna logits (valores brutos) — o argmax dá a classe
# com maior pontuação, que é a predição final.

print("=" * 60)
print("SEÇÃO 1 — COLETANDO PREDIÇÕES NO TESTE")
print("=" * 60)

y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.squeeze().long()
        preds  = model(imgs).argmax(1).cpu()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
print(f"Predições coletadas: {len(y_true)} amostras\n")


# ════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — MÉTRICAS GLOBAIS
# "Accuracy basta? Quando ela mente?"
# ════════════════════════════════════════════════════════════════════
#
# ACCURACY: de todas as predições, quantas estavam certas?
#   → simples, mas enganosa quando as classes são desbalanceadas.
#   → se 90% do dataset for classe A, um modelo que sempre chuta A
#     tem 90% de accuracy sem aprender nada.
#
# PRECISION: dos que o modelo disse "é classe X", quantos eram X?
#   → mede falsos positivos
#
# RECALL: de todos os X reais, quantos o modelo encontrou?
#   → mede falsos negativos
#
# F1: média harmônica de precision e recall
#   → penaliza quando um dos dois é muito baixo
#   → melhor resumo único do desempenho por classe
#
# MACRO vs MICRO:
#   Macro = calcula por classe e tira a média → cada classe vale igual
#   Micro = agrega todos os acertos/erros → classes maiores pesam mais

print("=" * 60)
print("SEÇÃO 2 — MÉTRICAS GLOBAIS")
print("=" * 60)

acc      = accuracy_score(y_true, y_pred)
f1_mac   = f1_score(y_true, y_pred, average="macro")
f1_mic   = f1_score(y_true, y_pred, average="micro")
prec_mac = precision_score(y_true, y_pred, average="macro")
rec_mac  = recall_score(y_true, y_pred, average="macro")

print(f"  Acurácia          : {acc:.4f}  ({acc:.2%})")
print(f"  F1 Macro          : {f1_mac:.4f}")
print(f"  F1 Micro          : {f1_mic:.4f}")
print(f"  Precision Macro   : {prec_mac:.4f}")
print(f"  Recall Macro      : {rec_mac:.4f}")
print()
print("  Relatório completo por classe:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))


# ════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — MÉTRICAS POR CLASSE + GRÁFICO
# "Quais classes são mais difíceis?"
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("SEÇÃO 3 — MÉTRICAS POR CLASSE")
print("=" * 60)

prec_cls = precision_score(y_true, y_pred, average=None)
rec_cls  = recall_score(y_true, y_pred, average=None)
f1_cls   = f1_score(y_true, y_pred, average=None)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"Métricas por Classe — BloodMNIST · Acurácia global: {acc:.2%}",
    fontsize=13, fontweight="bold",
)

metrics = [
    ("Precision", prec_cls, "Dos que o modelo disse 'é X', quantos eram X?"),
    ("Recall",    rec_cls,  "De todos os X reais, quantos o modelo encontrou?"),
    ("F1-Score",  f1_cls,   "Média harmônica de Precision e Recall"),
]

for ax, (nome, vals, desc) in zip(axes, metrics):
    bars = ax.barh(CLASS_NAMES, vals, color=CORES, edgecolor="white", linewidth=0.8)
    ax.set_xlim(0, 1.1)
    ax.set_title(nome, fontsize=12, fontweight="bold")
    ax.set_xlabel(desc, fontsize=8, color="#6b7280")
    ax.axvline(vals.mean(), color="#374151", ls="--", lw=1.2,
               label=f"Média: {vals.mean():.3f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="x")
    ax.spines[["top", "right"]].set_visible(False)

    # Valor dentro de cada barra
    for bar, v in zip(bars, vals):
        ax.text(min(v + 0.01, 1.05), bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("outputs/metricas_resumo.png", dpi=150, bbox_inches="tight")
plt.show()
print("→ salvo: outputs/metricas_resumo.png\n")


# ════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — MATRIZ DE CONFUSÃO
# "Onde o modelo erra? Com o que ele confunde cada célula?"
# ════════════════════════════════════════════════════════════════════
#
# Cada linha = classe real.  Cada coluna = o que o modelo previu.
# Diagonal principal = acertos.
# Fora da diagonal = erros.
#
# Exemplo de leitura:
#   Linha "Basófilo", coluna "Eosinófilo" = quantos basófilos
#   o modelo chamou erroneamente de eosinófilo.
#
# Normalizamos por linha (divide pela contagem real de cada classe)
# para comparar classes com tamanhos diferentes.

print("=" * 60)
print("SEÇÃO 4 — MATRIZ DE CONFUSÃO")
print("=" * 60)

cm_abs  = confusion_matrix(y_true, y_pred)
cm_norm = cm_abs.astype(float) / cm_abs.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Matriz de Confusão — onde o modelo acerta e onde erra",
             fontsize=13, fontweight="bold")

for ax, (data, title, fmt) in zip(axes, [
    (cm_abs,  "Valores absolutos (contagem)", "d"),
    (cm_norm, "Normalizada por classe (proporção)", ".2f"),
]):
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predito", fontsize=10)
    ax.set_ylabel("Real", fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

    thresh = data.max() / 2
    for i in range(8):
        for j in range(8):
            val = f"{data[i,j]:{fmt}}"
            color = "white" if data[i, j] > thresh else "#1f2937"
            ax.text(j, i, val, ha="center", va="center",
                    fontsize=7.5, color=color)

plt.tight_layout()
plt.savefig("outputs/matriz_confusao.png", dpi=150, bbox_inches="tight")
plt.show()
print("→ salvo: outputs/matriz_confusao.png\n")


# ════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — GRAD-CAM
# "O que a rede realmente olha para tomar a decisão?"
# ════════════════════════════════════════════════════════════════════
#
# Grad-CAM (Gradient-weighted Class Activation Map) é uma técnica
# de explicabilidade: ela mostra quais regiões da imagem mais
# influenciaram a decisão do modelo.
#
# Como funciona:
#   1. Forward pass com cam=True → salva as ativações do último stage
#   2. Backpropagation da classe predita → calcula os gradientes
#   3. Média dos gradientes por canal → peso de cada mapa de feature
#   4. Soma ponderada dos mapas → mapa de calor (heatmap)
#   5. Redimensiona para 28×28 e sobrepõe na imagem original
#
# Resultado: regiões quentes = onde a rede "prestou atenção"
# Ideal: foco no núcleo e citoplasma da célula, não no fundo

print("=" * 60)
print("SEÇÃO 5 — GRAD-CAM")
print("=" * 60)


def grad_cam(model, img_tensor):
    """Retorna o mapa de calor Grad-CAM e a classe predita."""
    model.eval()
    img = img_tensor.unsqueeze(0).to(DEVICE)

    # Forward com cam=True → registra hook de gradiente
    output = model(img, cam=True)
    pred   = output.argmax(1).item()

    # Backward na classe predita
    model.zero_grad()
    output[0, pred].backward()

    # Pesos = média global dos gradientes por canal
    grads  = model.gradients                          # (1, 128, 7, 7)
    acts   = model.activations                        # (1, 128, 7, 7)
    weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, 128, 1, 1)

    # Mapa de calor: soma ponderada + ReLU
    heatmap = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    heatmap = heatmap.squeeze().detach().cpu().numpy()

    # Normaliza para [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap, pred


def overlay_cam(img_tensor, heatmap, alpha=0.45):
    """Sobrepõe o heatmap colorido na imagem original."""
    inv = transforms.Normalize([-1, -1, -1], [2, 2, 2])
    img_np = inv(img_tensor).permute(1, 2, 0).clamp(0, 1).numpy()

    # Redimensiona heatmap de 7×7 para 28×28
    heatmap_big = np.array(
        plt.cm.jet(heatmap)[:, :, :3]  # colormap RGB
    )
    # Upscale simples por repetição
    heatmap_big = np.repeat(np.repeat(heatmap, 4, axis=0), 4, axis=1)
    heatmap_big = plt.cm.jet(heatmap_big)[:, :, :3]

    return np.clip((1 - alpha) * img_np + alpha * heatmap_big, 0, 1)


# Coleta exemplos: 1 correto por classe + até 8 erros de qualquer classe
print("Gerando Grad-CAM para exemplos do conjunto de teste...")

ds_plain = BloodMNIST(split="test", transform=test_transform, download=False)
correct_by_class = {c: None for c in range(8)}
wrong_examples   = []   # lista de (img, true_label, pred_label)

for i in range(len(ds_plain)):
    img, label = ds_plain[i]
    label  = int(label.item())
    pred_i = y_pred[i]

    if pred_i == label and correct_by_class[label] is None:
        correct_by_class[label] = (img, label)
    elif pred_i != label and len(wrong_examples) < 8:
        wrong_examples.append((img, label, pred_i))

    done_correct = all(v is not None for v in correct_by_class.values())
    done_wrong   = len(wrong_examples) >= 8
    if done_correct and done_wrong:
        break

inv = transforms.Normalize([-1, -1, -1], [2, 2, 2])

# ── FIGURA 1: acertos — 8 colunas (uma por classe), 2 linhas (original + cam) ──
fig, axes = plt.subplots(2, 8, figsize=(20, 6))
fig.suptitle("Grad-CAM — o que a rede olha ao acertar cada tipo de célula\n"
             "🔴 Vermelho/amarelo = maior atenção   🔵 Azul = menor atenção",
             fontsize=11, fontweight="bold")
fig.subplots_adjust(top=0.83, hspace=0.05, wspace=0.05)

for c in range(8):
    entry = correct_by_class[c]
    ax_orig = axes[0, c]
    ax_cam  = axes[1, c]

    if entry is None:
        ax_orig.axis("off"); ax_cam.axis("off"); continue

    img, label = entry
    heatmap, _ = grad_cam(model, img)
    blended    = overlay_cam(img, heatmap)
    img_np     = inv(img).permute(1, 2, 0).clamp(0, 1).numpy()

    ax_orig.imshow(img_np)
    ax_orig.set_title(CLASS_NAMES[c], fontsize=8.5,
                      fontweight="bold", color=CORES[c], pad=4)
    ax_orig.axis("off")

    ax_cam.imshow(blended)
    ax_cam.axis("off")

# Rótulos de linha
axes[0, 0].set_ylabel("Original", fontsize=8, color="#374151")
axes[1, 0].set_ylabel("Grad-CAM", fontsize=8, color="#374151")

plt.savefig("outputs/gradcam_acertos.png", dpi=150, bbox_inches="tight")
plt.show()
print("→ salvo: outputs/gradcam_acertos.png")

# ── FIGURA 2: erros — mostra os erros que existem ───────────────────────────
n_wrong = len(wrong_examples)
if n_wrong == 0:
    print("→ Nenhum erro encontrado no teste (modelo perfeito nessa amostra)!")
else:
    cols = min(n_wrong, 8)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 2.8, 7))
    if cols == 1:
        axes = np.array(axes).reshape(2, 1)
    fig.suptitle(f"Grad-CAM — onde a rede errou ({n_wrong} exemplos)\n"
                 "Região vermelha = onde a rede olhou para decidir",
                 fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.87, bottom=0.15, hspace=0.04, wspace=0.06)

    for i, (img, true_label, pred_label) in enumerate(wrong_examples[:cols]):
        heatmap, _ = grad_cam(model, img)
        blended    = overlay_cam(img, heatmap)
        img_np     = inv(img).permute(1, 2, 0).clamp(0, 1).numpy()

        # ── Linha 0: imagem original + título "Real: X" em cima
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"Real: {CLASS_NAMES[true_label]}",
                              fontsize=8.5, color=CORES[true_label],
                              fontweight="bold", pad=5)
        axes[0, i].axis("off")

        # ── Linha 1: Grad-CAM + label "Predito: Y" abaixo da imagem
        axes[1, i].imshow(blended)
        axes[1, i].axis("off")
        axes[1, i].text(
            0.5, -0.06,
            f"Predito:\n{CLASS_NAMES[pred_label]}",
            transform=axes[1, i].transAxes,
            ha="center", va="top",
            fontsize=8, color="#dc2626", fontweight="bold",
            linespacing=1.4, clip_on=False,
        )

    plt.savefig("outputs/gradcam_erros.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("→ salvo: outputs/gradcam_erros.png")



# ════════════════════════════════════════════════════════════════════
# RESUMO FINAL
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("RESUMO FINAL — desempenho do modelo")
print("=" * 60)
print(f"  Acurácia  : {acc:.2%}")
print(f"  F1 Macro  : {f1_mac:.2%}  (cada classe vale igual)")
print(f"  F1 Micro  : {f1_mic:.2%}  (classes maiores pesam mais)")
print()

melhor = int(np.argmax(f1_cls))
pior   = int(np.argmin(f1_cls))
print(f"  Melhor classe : {CLASS_NAMES[melhor]} (F1 = {f1_cls[melhor]:.3f})")
print(f"  Pior classe   : {CLASS_NAMES[pior]}  (F1 = {f1_cls[pior]:.3f})")
print()
print("  Arquivos gerados em outputs/:")
print("    metricas_resumo.png")
print("    matriz_confusao.png")
print("    gradcam_exemplos.png")