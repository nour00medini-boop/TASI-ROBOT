"""Trace les fonctions d'appartenance et les surfaces de commande de FIS-A.
Cree deux images PNG dans le dossier courant."""
import numpy as np
import matplotlib.pyplot as plt
from archive.fis_a import _VARS, score_collecte

# ------------------------------------------------ 1) fonctions d'appartenance
titres = {"conf": "conf (confiance YOLO)", "dist": "dist (m)",
          "batt": "batt (%)", "bac": "bac (%)", "score": "score (sortie)"}
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for ax, (nom, var) in zip(axes.flat, _VARS.items()):
    for terme, mf in var.terms.items():
        ax.plot(var.universe, mf.mf, label=terme, linewidth=2)
    ax.set_title(titres[nom])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("degre d'appartenance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
axes.flat[-1].axis("off")
fig.suptitle("FIS-A : fonctions d'appartenance")
fig.tight_layout()
fig.savefig("fis_a_appartenance.png", dpi=150)
print("ecrit : fis_a_appartenance.png")


# ------------------------------------------------ 2) surfaces de commande
def surface(ax, f, x, y, xl, yl, titre):
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(a, b) for a in x] for b in y])
    im = ax.contourf(X, Y, Z, levels=np.arange(0, 101, 10), cmap="viridis")
    ax.contour(X, Y, Z, levels=[50], colors="red", linewidths=2)  # seuil 50
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(titre)
    return im


fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
im = surface(axes[0], lambda c, d: score_collecte(c, d, 80, 20),
             np.linspace(0, 1, 26), np.linspace(0, 5, 26),
             "conf", "dist (m)", "score(conf, dist), batt=80, bac=20")
surface(axes[1], lambda d, z: score_collecte(0.9, d, 80, z),
        np.linspace(0, 5, 26), np.linspace(0, 100, 26),
        "dist (m)", "bac (%)", "score(dist, bac), conf=0.9, batt=80")
surface(axes[2], lambda d, b: score_collecte(0.9, d, b, 20),
        np.linspace(0, 5, 26), np.linspace(0, 100, 26),
        "dist (m)", "batt (%)", "score(dist, batt), conf=0.9, bac=20")
fig.colorbar(im, ax=axes, label="score (ligne rouge = seuil 50)")
fig.savefig("fis_a_surfaces.png", dpi=150, bbox_inches="tight")
print("ecrit : fis_a_surfaces.png")
plt.show()