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
# # 1. Fundamentos teóricos
#
# Seis resultados que el enunciado exige demostrar analíticamente y verificar numéricamente (4.8.23). Cada uno
# justifica una decisión del protocolo: qué métrica usar, qué pérdida, cómo validar y cómo calibrar.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from dl03 import calibration as cal
from dl03 import seg_metrics as sm
from dl03.viz import savefig, setup

cfg = setup()
rng = np.random.default_rng(cfg.seed)

# %% [markdown]
# ## 1.1 Relación Dice–IoU
#
# Sea $I = |A \cap B|$ y $U = |A \cup B|$. Como $|A| + |B| = U + I$,
#
# $$D = \frac{2I}{|A| + |B|} = \frac{2I}{U + I} = \frac{2\,(I/U)}{1 + I/U} = \frac{2J}{1 + J}.$$
#
# $f(J) = 2J/(1+J)$ tiene derivada $f'(J) = 2/(1+J)^2 > 0$: es estrictamente creciente, así que **para un mismo
# par (predicción, referencia)** ordenar por Dice u ordenar por IoU da lo mismo. Pero $f$ es cóncava, y el promedio
# de una función cóncava no es la función del promedio (desigualdad de Jensen): al promediar sobre imágenes los
# rankings pueden invertirse.

# %%
def dice_from_iou(j):
    return 2 * j / (1 + j)


a = rng.random((64, 64)) > 0.5
b = rng.random((64, 64)) > 0.4
d, j = sm.dice_iou(a, b)
print(f"Dice medido = {d:.6f}; 2J/(1+J) = {dice_from_iou(j):.6f}")

modelo_A = np.array([0.50, 0.50])          # IoU por imagen
modelo_B = np.array([0.10, 0.95])
contra = pd.DataFrame({"IoU medio": [modelo_A.mean(), modelo_B.mean()],
                       "Dice medio": [dice_from_iou(modelo_A).mean(), dice_from_iou(modelo_B).mean()]},
                      index=["Modelo A", "Modelo B"])
contra

# %% [markdown]
# **Contraejemplo.** B gana por IoU medio (0.525 > 0.500) y pierde por Dice medio (0.578 < 0.667): la
# concavidad de $f$ premia más la regularidad de A. Por eso el protocolo fija **una** métrica primaria antes de ver
# resultados.

# %% [markdown]
# ## 1.2 Gradiente de la pérdida Dice suave
#
# Con $S = \sum_i p_i g_i$, $P = \sum_i p_i$, $G = \sum_i g_i$ y $L = 1 - \dfrac{2S + \epsilon}{P + G + \epsilon}$:
#
# $$\frac{\partial L}{\partial p_i} = -\frac{2 g_i (P + G + \epsilon) - (2S + \epsilon)}{(P + G + \epsilon)^2}.$$
#
# * Píxel de célula ($g_i = 1$): gradiente negativo (sube $p_i$) de magnitud $\approx (2(P+G) - 2S)/(P+G)^2$.
# * Píxel de fondo ($g_i = 0$): gradiente positivo **igual para todos** los píxeles de fondo, $(2S+\epsilon)/(P+G+\epsilon)^2$.
#
# Cuando el primer plano ocupa una fracción mínima, $P + G$ es pequeño y el gradiente por píxel crece como
# $1/(P+G)$: la pérdida no se ahoga en el fondo (a diferencia de la entropía cruzada promediada, cuyo gradiente
# total está dominado por los $N(1-\pi)$ píxeles de fondo), pero se vuelve **inestable** en imágenes casi
# vacías. La pérdida focal, $-(1-p_t)^\gamma \log p_t$, ataca el mismo desbalance de otra forma: reduce el peso
# de los píxeles ya bien clasificados (casi todo el fondo). El protocolo compara BCE, Dice, BCE + Dice y
# Tversky/focal en la ablación (capítulo 5).

# %%
def soft_dice(p, g, eps=1.0):
    return 1 - (2 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps)


def soft_dice_grad(p, g, eps=1.0):
    S, P, G = (p * g).sum(), p.sum(), g.sum()
    return -(2 * g * (P + G + eps) - (2 * S + eps)) / (P + G + eps) ** 2


p = rng.random(200)
g = (rng.random(200) < 0.1).astype(float)
h = 1e-6
fd = np.array([(soft_dice(p + h * np.eye(200)[i], g) - soft_dice(p - h * np.eye(200)[i], g)) / (2 * h)
               for i in range(200)])
print("error máximo analítico vs diferencias finitas:", np.abs(fd - soft_dice_grad(p, g)).max())

rows = []
N = 10_000
for pi in [0.5, 0.1, 0.01, 0.001]:
    g = (rng.random(N) < pi).astype(float)
    p = np.clip(0.5 + 0.1 * (g - 0.5) + rng.normal(0, 0.05, N), 0.01, 0.99)
    gd = soft_dice_grad(p, g)
    gce = (-(g / p) + (1 - g) / (1 - p)) / N
    rows.append({"π": pi, "|∂Dice| medio en célula": np.abs(gd[g == 1]).mean(),
                 "|∂CE| medio en célula": np.abs(gce[g == 1]).mean(),
                 "fracción del gradiente CE que viene del fondo": np.abs(gce[g == 0]).sum() / np.abs(gce).sum()})
pd.DataFrame(rows).set_index("π")

# %% [markdown]
# A medida que $\pi \to 0$, el gradiente Dice por píxel de célula crece (amplifica la señal escasa) y el de la
# entropía cruzada queda dominado por el fondo (última columna → 1).

# %% [markdown]
# ## 1.3 Desbalance y métricas de píxel
#
# Sea $\pi$ la proporción de píxeles de célula. Un clasificador trivial que dice «fondo» siempre tiene
#
# * exactitud $= 1 - \pi \to 1$;
# * balanced accuracy $= \tfrac12(\text{recall} + \text{especificidad}) = \tfrac12(0 + 1) = 0.5$;
# * AUC-ROC $= 0.5$ (un puntaje constante no ordena).
#
# Un clasificador «casi trivial» con puntajes $s \sim \mathcal N(\mu_1, 1)$ en células y $\mathcal N(\mu_0, 1)$ en
# fondo tiene $\text{AUC} = \Phi\big((\mu_1-\mu_0)/\sqrt 2\big)$, **independiente de $\pi$**: AUC solo mide el
# orden, no cuántos falsos positivos hay en números absolutos. La precisión, en cambio, es
# $\text{TPR}\,\pi / (\text{TPR}\,\pi + \text{FPR}(1-\pi)) \to 0$ cuando $\pi \to 0$ a TPR/FPR fijos, y el área
# bajo la curva Precision–Recall tiende a $\pi$ para un clasificador aleatorio. En el régimen de células escasas la
# curva PR es la que muestra el costo real de los falsos positivos.

# %%
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score  # noqa: E402

rows = []
for pi in [0.5, 0.1, 0.01, 0.001]:
    n = 200_000
    y = rng.random(n) < pi
    s = rng.normal(np.where(y, 1.0, 0.0), 1.0)
    trivial = np.zeros(n, bool)
    rows.append({"π": pi, "exactitud trivial": (trivial == y).mean(),
                 "bal. acc. trivial": balanced_accuracy_score(y, trivial),
                 "AUC casi trivial": roc_auc_score(y, s), "AUC teórica": stats.norm.cdf(1 / np.sqrt(2)),
                 "AUC-PR casi trivial": average_precision_score(y, s)})
pd.DataFrame(rows).set_index("π")

# %% [markdown]
# ## 1.4 Sesgo optimista de la validación cruzada no anidada
#
# Si se evalúan $K$ configuraciones con estimaciones $\hat\mu_k = \mu_k + \varepsilon_k$ (ruido de validación) y se
# reporta $\max_k \hat\mu_k$, entonces $\mathbb E[\max_k \hat\mu_k] \ge \max_k \mu_k$: el máximo de variables
# ruidosas está sesgado hacia arriba, más cuanto más configuraciones y menos datos de validación. La validación
# anidada separa **seleccionar** (bucle interno) de **estimar** (bucle externo, datos que no participaron en la
# selección), de modo que la cifra externa es insesgada para el procedimiento completo «buscar + reentrenar».
# Estimación empírica con un SVM RBF en datos sintéticos, comparando con un test grande independiente:

# %%
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

grid = {"C": np.logspace(-2, 3, 6), "gamma": np.logspace(-4, 1, 6)}
reps = 3 if cfg.is_smoke else 15
res = []
for r in range(reps):
    X, y = make_classification(n_samples=2200, n_features=30, n_informative=5, flip_y=0.1, random_state=r)
    Xd, yd, Xt, yt = X[:200], y[:200], X[200:], y[200:]
    gs = GridSearchCV(SVC(), grid, cv=StratifiedKFold(5, shuffle=True, random_state=r)).fit(Xd, yd)
    nested = cross_val_score(GridSearchCV(SVC(), grid, cv=StratifiedKFold(3, shuffle=True, random_state=r)), Xd, yd,
                             cv=StratifiedKFold(5, shuffle=True, random_state=100 + r)).mean()
    res.append({"no anidada (mejor CV)": gs.best_score_, "anidada": nested,
                "test grande (verdad)": gs.best_estimator_.score(Xt, yt)})
cvb = pd.DataFrame(res)
bias = (cvb["no anidada (mejor CV)"] - cvb["test grande (verdad)"]).agg(["mean", "std"])
print(f"sesgo de la CV no anidada: {bias['mean']:+.3f} ± {bias['std']:.3f};",
      f"anidada: {(cvb['anidada'] - cvb['test grande (verdad)']).mean():+.3f}")
cvb.describe().loc[["mean", "std"]]

# %% [markdown]
# ## 1.5 Calibración
#
# **ECE** (expected calibration error): con $B$ intervalos de confianza $I_b$,
# $\mathrm{ECE} = \sum_b \tfrac{|I_b|}{n}\,|\mathrm{acc}(I_b) - \mathrm{conf}(I_b)|$.
# **Brier** (multiclase): $\tfrac1n \sum_i \lVert \hat{\mathbf p}_i - \mathbf e_{y_i}\rVert^2$, puntaje propio que
# combina calibración y discriminación.
#
# **La temperatura no cambia el argmax.** Para $T > 0$, $z \mapsto z/T$ es estrictamente creciente en cada
# coordenada y la softmax también lo es, luego $\arg\max_k \operatorname{softmax}(z/T)_k = \arg\max_k z_k$: la
# exactitud es idéntica. En cambio la NLL $= -\log \operatorname{softmax}(z/T)_y$ sí depende de $T$
# (su derivada es $\tfrac{1}{T^2}(z_y - \mathbb E_{p_T}[z])$, nula solo en el óptimo), y por eso $T$ se ajusta
# minimizando la NLL en validación.

# %%
z = rng.normal(0, 1, (5000, 5))
y = np.array([rng.choice(5, p=q) for q in cal.softmax(z)])
logits = 2.5 * z
T = cal.fit_temperature(logits[:2500], y[:2500])
P0, PT = cal.softmax(logits[2500:]), cal.softmax(logits[2500:] / T)
pd.DataFrame({"sin escalar": [(P0.argmax(1) == y[2500:]).mean(), cal.nll(P0, y[2500:]),
                              cal.expected_calibration_error(P0, y[2500:]), cal.brier_score(P0, y[2500:])],
              f"T = {T:.2f}": [(PT.argmax(1) == y[2500:]).mean(), cal.nll(PT, y[2500:]),
                               cal.expected_calibration_error(PT, y[2500:]), cal.brier_score(PT, y[2500:])]},
             index=["exactitud", "NLL", "ECE", "Brier"])

# %% [markdown]
# ## 1.6 Complejidad computacional
#
# * **U-Net++** con profundidad $L$, anchos $C_i$ y entrada $H \times W$: una convolución $k\times k$ en el nivel $i$
#   cuesta $O(\tfrac{HW}{4^i} k^2 C_i^2)$ y el nivel $i$ tiene $L-i$ nodos, así que el tiempo es
#   $O\big(HW k^2 \sum_i (L-i) C_i^2 / 4^i\big) = O(HW)$ para una arquitectura fija; la memoria de activaciones
#   también es $O(HW)$ (los nodos anidados la multiplican por una constante mayor que la U-Net).
# * **BiLSTM** con secuencia $T$, embeddings de dimensión $e$ y $d$ unidades: $O(T\,d\,(d + e))$ en tiempo, con
#   $T$ pasos **secuenciales** (no paralelizables en el tiempo), y $O(T d)$ de memoria.
# * **Auto-atención** con $T$ tokens de dimensión $d$: $O(T^2 d)$ en tiempo y $O(T^2)$ en memoria por cabeza
#   (la matriz de atención), totalmente paralelizable.
#
# Contraste empírico (CPU, una pasada hacia adelante, mediana de 3): la pendiente en escala log–log estima el
# exponente.

# %%
import keras  # noqa: E402

from dl03.seg_models.unetpp import build_unetpp  # noqa: E402


def timeit(f, n=3):
    f()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


unet = build_unetpp((8, 16, 32, 64, 128))
sizes = [64, 128, 256] if cfg.is_smoke else [64, 128, 256, 512]
t_unet = [timeit(lambda s=s: unet.predict(np.zeros((1, s, s, 1), np.float32), verbose=0)) for s in sizes]
lens = [32, 64, 128, 256] if cfg.is_smoke else [64, 128, 256, 512, 1024]
inp = keras.Input((None, 64))
lstm = keras.Model(inp, keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(inp))
mha_in = keras.Input((None, 64))
mha = keras.Model(mha_in, keras.layers.MultiHeadAttention(4, 16)(mha_in, mha_in))
t_lstm = [timeit(lambda t=t: lstm.predict(np.zeros((1, t, 64), np.float32), verbose=0)) for t in lens]
t_mha = [timeit(lambda t=t: mha.predict(np.zeros((1, t, 64), np.float32), verbose=0)) for t in lens]


def slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]


fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].loglog(np.square(sizes), t_unet, "o-", label=f"U-Net++ (pendiente {slope(np.square(sizes), t_unet):.2f})")
ax[0].set_xlabel("píxeles (H·W)")
ax[0].set_ylabel("segundos")
ax[0].legend()
ax[1].loglog(lens, t_lstm, "o-", label=f"BiLSTM ({slope(lens, t_lstm):.2f})")
ax[1].loglog(lens, t_mha, "s-", label=f"auto-atención ({slope(lens, t_mha):.2f})")
ax[1].set_xlabel("longitud de secuencia T")
ax[1].legend()
fig.suptitle("Tiempo de inferencia vs tamaño de entrada (pendientes log–log)")
savefig(cfg, "01_complejidad")
plt.show()

# %% [markdown]
# Las pendientes esperadas son ≈ 1 (U-Net++ respecto de píxeles y BiLSTM respecto de T) y → 2 para la
# auto-atención cuando $T$ es grande; con $T$ pequeño domina el costo lineal de las proyecciones y la pendiente
# medida queda entre 1 y 2. Sobrecostos fijos (llamada a `predict`) aplanan las curvas en tamaños pequeños.
