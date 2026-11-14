# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 2. Ejercicios manuales: verificación numérica (Sección 4.8.1–4.8.17)
#
# Los dos ejercicios de esta sección **se resuelven a mano** (lápiz, papel y calculadora; la guía prohíbe Python y
# hojas de cálculo) y se entregan por separado. Este capítulo no sustituye ese trabajo: reproduce los cálculos con
# las fórmulas del enunciado para **comprobar** la solución manual (tolerancia ±0.002), verifica cada gradiente
# analítico contra diferencias finitas y comprueba los criterios de convergencia exigidos.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dl03 import manual_exercises as mx
from dl03.viz import savefig, setup

cfg = setup()
h1, h2 = mx.run_lstm(), mx.run_w2v()

# %% [markdown]
# ## 2.1 LSTM escalar (BPTT, η = 3)

# %%
t2 = pd.DataFrame([{"k": it["k"], "z_A": it["seq"]["A"]["z"], "z_B": it["seq"]["B"]["z"],
                    "ŷ_A": it["seq"]["A"]["yh"], "ŷ_B": it["seq"]["B"]["yh"], "L_A": it["seq"]["A"]["L"],
                    "L_B": it["seq"]["B"]["L"], "L(k)": it["L"], "converge": it["conv"],
                    "error máx. gradiente vs dif. finitas": it["fd_err"]} for it in h1]).set_index("k")
t2.round(4)

# %%
t3 = pd.DataFrame({it["k"]: it["theta"] for it in h1}).rename(columns=lambda k: f"k={k}")
t3.round(4)

# %%
assert not h1[3]["conv"] and h1[4]["conv"], "el criterio debe fallar en k=3 y cumplirse en k=4"
print(f"k=3: L={h1[3]['L']:.4f} (≥ 0.20) → no converge; k=4: L={h1[4]['L']:.4f} → converge (K* = 4)")

# %% [markdown]
# ## 2.2 Skip-gram tipo word2vec (η = 4)

# %%
W = pd.DataFrame([{"k": it["k"], "L(k)": it["L"], "L−Lmin": it["L"] - mx.LMIN,
                   **{f"W1[{w},{j + 1}]": it["W1"][i, j] for i, w in enumerate(mx.VOC) for j in range(2)},
                   "cos(cat,dog)": it["cos"]} for it in h2]).set_index("k")
D = np.abs(np.diff(np.array([it["W1"] for it in h2]), axis=0)).reshape(len(h2) - 1, 6)
kstar = [int(np.argmax(D[:, c] < 0.06)) + 1 for c in range(6)]
print("L_min = ½·ln2 =", round(mx.LMIN, 4), "| k* por peso:", kstar, "| K* =", max(kstar))
W.round(4)

# %%
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
ax[0].plot([it["L"] for it in h1], "o-")
ax[0].axhline(0.20, ls="--", c="gray")
ax[0].set(title="LSTM: L(k) y umbral 0.20", xlabel="k")
ax[1].plot(W["L(k)"], "o-")
ax[1].axhline(mx.LMIN, ls="--", c="gray")
ax[1].set(title="word2vec: L(k) y L_min", xlabel="k")
ax[2].plot(W["cos(cat,dog)"], "o-")
ax[2].set(title="cos(cat, dog)", xlabel="k")
savefig(cfg, "02_ejercicios_manuales")
plt.show()
