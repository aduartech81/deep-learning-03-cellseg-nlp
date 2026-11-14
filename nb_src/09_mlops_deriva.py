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
# # 9. MLOps: exportación, registro, API, carga y experimento de deriva (RQ8)
#
# 1. Se elige el modelo a desplegar por dominio con un criterio **declarado aquí** (cabe en 512 MB de RAM y está
#    a menos de δ del mejor, o es el más cercano), y la elección se revisa en el análisis multicriterio del
#    capítulo 10.
# 2. Se exporta a formatos de inferencia ligeros (ONNX para U-Net++, joblib para el modelo de texto) con
#    verificación de paridad numérica, se registran en MLflow (si está disponible) y en un registro de archivos
#    con etapas y reversión, y se versionan los datos (DVC o, en su defecto, el MANIFEST con SHA-256).
# 3. Se levanta la API real, se prueba y se mide su latencia bajo carga concurrente.
# 4. Se simula deriva controlada y gradual en ambos dominios y se mide el poder de detección.

# %%
import base64
import io
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dl03 import drift
from dl03 import nlp_data as nd
from dl03 import nlp_experiments as ne
from dl03 import seg_data as sd
from dl03 import seg_experiments as se
from dl03.viz import PALETTE, save_json, save_table, savefig, setup

cfg = setup(seed_offset=9)
ROOT = Path("..").resolve()
APPM = ROOT / "app" / "models"
APPM.mkdir(parents=True, exist_ok=True)
VERSION = time.strftime("1.0.%Y%m%d")

# %% [markdown]
# ## 9.1 Elección y exportación del modelo de texto

# %%
df, y, classes, groups, sp = ne.prepare(cfg)
X, K = df["text"].values, len(classes)
te = np.array(sp["test"])
finals = pd.read_csv(cfg.out_dir / "nlp" / "finals.csv")
f1_by = finals.groupby("model")["test_f1"].mean().sort_values(ascending=False)
LIGHT_DEPLOY = ["tfidf_lr", "nbsvm", "tfidf_xgb"]          # caben en 512 MB sin PyTorch/TensorFlow
cand = [m for m in f1_by.index if m in LIGHT_DEPLOY]
text_choice = cand[0]
gap = f1_by.iloc[0] - f1_by[text_choice]
print(f"mejor en test: {f1_by.index[0]} ({f1_by.iloc[0]:.4f}); desplegable: {text_choice} ({f1_by[text_choice]:.4f});"
      f" brecha {gap:.4f} ({'< δ' if gap < 0.02 else '≥ δ: se declara la pérdida'})")
import joblib  # noqa: E402

tm = joblib.load(cfg.out_dir / "nlp" / "models" / text_choice / "model.joblib")
joblib.dump(tm, APPM / "text_model.joblib", compress=3)
(APPM / "text_model.json").write_text(json.dumps({"classes": classes, "version": f"text-{text_choice}-{VERSION}"}))
parity = float(np.abs(joblib.load(APPM / "text_model.joblib").predict_proba(X[te[:50]]) -
                      tm.predict_proba(X[te[:50]])).max())
print("paridad joblib (máx |Δp|):", parity)

# %% [markdown]
# ## 9.2 Elección y exportación del modelo de imágenes (U-Net++ → ONNX) con paridad y cuantización

# %%
img_metrics = pd.read_csv(cfg.out_dir / "seg" / "metrics_test.csv")
ap_by = img_metrics.groupby("model")["ap"].mean().sort_values(ascending=False)
image_choice = "unetpp"
print(f"mejor en test: {ap_by.index[0]} ({ap_by.iloc[0]:.4f}); desplegado: U-Net++ ({ap_by['unetpp']:.4f}); "
      f"brecha {ap_by.iloc[0] - ap_by['unetpp']:.4f}")
S0 = cfg.seg["seeds_heavy"][0]
umodel = se.load_keras(se.run_dir(cfg, "models") / f"unetpp_s{S0}.keras")
pp = json.loads((cfg.out_dir / "seg" / "params" / f"unetpp_s{S0}.json").read_text())["params"]
onnx_path = APPM / "unetpp.onnx"
umodel.predict(np.zeros((1, 64, 64, 1), np.float32), verbose=0)   # Keras exige una llamada antes de exportar
umodel.export(str(onnx_path), format="onnx", verbose=False)
import onnxruntime as ort  # noqa: E402

images, instances = sd.build_index(cfg)
splits = sd.Splits.from_json((cfg.out_dir / "seg" / "splits.json").read_text())
RES = cfg.seg["img_size"]
x01 = np.clip(sd.normalize(se.native_image(images, splits.test[0])), 0, 1)
gm = sd.Geometry.for_resolution(x01.shape, RES)
xin = gm.to_model(x01)[None, ..., None].astype(np.float32)
ref = np.mean(umodel.predict(xin, verbose=0), 0)
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
out = sess.run(None, {sess.get_inputs()[0].name: xin})
onx = np.mean(out, 0) if len(out) > 1 else out[0]
TOL = 1e-3
par_img = float(np.abs(onx - ref).max())
print(f"paridad Keras vs ONNX: máx |Δp| = {par_img:.2e} (tolerancia declarada {TOL})")
assert par_img < TOL
from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402

q_path = ROOT / "results_q.onnx"
quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QUInt8)
sq = ort.InferenceSession(str(q_path), providers=["CPUExecutionProvider"])


def ort_lat(s, n=10):
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        s.run(None, {s.get_inputs()[0].name: xin})
        t.append(1000 * (time.perf_counter() - t0))
    return np.percentile(t, [50, 95])


oq = sq.run(None, {sq.get_inputs()[0].name: xin})
oq = np.mean(oq, 0) if len(oq) > 1 else oq[0]
qrep = {"ONNX FP32": {"MB": onnx_path.stat().st_size / 2 ** 20, "p50 ms": ort_lat(sess)[0], "máx |Δp|": par_img},
        "ONNX INT8 dinámico": {"MB": q_path.stat().st_size / 2 ** 20, "p50 ms": ort_lat(sq)[0],
                               "máx |Δp|": float(np.abs(oq - ref).max())}}
q_path.unlink()
(APPM / "image_model.json").write_text(json.dumps({"postproc": pp, "resolution": RES,
                                                   "version": f"image-unetpp-{VERSION}"}))
save_table(cfg, "mlops_onnx", pd.DataFrame(qrep).T)
pd.DataFrame(qrep).T.round(4)

# %% [markdown]
# ## 9.3 Versionado de datos, registro de modelos y *model_info*

# %%
manifests = {k: json.loads(p.read_text()) for k, p in
             {"sartorius": cfg.data_dir / "raw" / "sartorius" / "MANIFEST.json",
              "resumes": cfg.data_dir / "raw" / "resumes" / "MANIFEST.json"}.items() if p.exists()}
dvc_ok = False
try:
    if shutil.which("dvc"):
        subprocess.run(["dvc", "init", "-q"], cwd=ROOT, check=False, capture_output=True)
        for d in ["sartorius", "resumes"]:
            p = (cfg.data_dir / "raw" / d).relative_to(ROOT)
            subprocess.run(["dvc", "add", str(p)], cwd=ROOT, check=True, capture_output=True)
        dvc_ok = True
except Exception as e:
    print("DVC no disponible, se usa el MANIFEST con SHA-256:", e)
NAMES_NLP = {"tfidf_lr": "TF-IDF + LR", "nbsvm": "NB-SVM", "tfidf_xgb": "TF-IDF + XGBoost"}
seg_tab = pd.read_csv(cfg.out_dir / "tables" / "seg_resultados_test.csv", index_col=0)
nlp_tab = pd.read_csv(cfg.out_dir / "tables" / "nlp_resultados_test.csv", index_col=0)
info = {"version": VERSION, "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                                     text=True).stdout.strip(),
        "data": manifests, "dvc": dvc_ok,
        "text": {"model": text_choice, "test_f1_macro_mean_seeds": float(f1_by[text_choice]),
                 "test_metrics": nlp_tab.loc[NAMES_NLP[text_choice]].to_dict() if NAMES_NLP[text_choice] in
                 nlp_tab.index else {},
                 "classes": classes},
        "image": {"model": "U-Net++ + watershed (ONNX)", "test_ap": float(ap_by["unetpp"]),
                  "resolution": RES, "postproc": pp, "onnx_parity_max_abs": par_img}}
(APPM / "model_info.json").write_text(json.dumps(info, indent=1, default=str))
sys.path.insert(0, str(ROOT))
from app import registry  # noqa: E402

registry.register(APPM, "dl03", VERSION, {"f1_macro": info["text"]["test_f1_macro_mean_seeds"],
                                          "ap": info["image"]["test_ap"]})
registry.promote("dl03", VERSION, "production")
try:
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("dl03")
    with mlflow.start_run(run_name=f"deploy-{VERSION}"):
        mlflow.log_params({"text_model": text_choice, "image_model": "unetpp", "resolution": RES})
        mlflow.log_metrics({"test_f1_macro": info["text"]["test_f1_macro_mean_seeds"], "test_ap": info["image"]["test_ap"]})
        mlflow.log_artifacts(str(APPM), artifact_path="models")
    print("MLflow: corrida registrada en mlflow.db")
except Exception as e:
    print("MLflow no disponible:", e)
{k: v for k, v in info.items() if k != "data"}

# %% [markdown]
# ### Conjunto dorado para la prueba de regresión de CI
# 100 hojas de vida de test (licencia CC0) y 4 imágenes **sintéticas** con referencia conocida (las imágenes de
# Sartorius no se redistribuyen). Se registra el desempeño del modelo desplegado sobre ese conjunto: la CI exige
# que una nueva versión no caiga más de ε = 0.02.

# %%
from skimage import io as skio  # noqa: E402

from app.inference import ImageModel, TextModel  # noqa: E402
from dl03.config import load_config  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

G = ROOT / "app" / "golden"
G.mkdir(parents=True, exist_ok=True)
gsel = te[:100]
pd.DataFrame({"text": X[gsel], "label": [classes[k] for k in y[gsel]]}).to_csv(G / "texts.csv", index=False)
tm_ = TextModel(APPM)
g_pred = [max(p, key=p.get) for p in map(tm_.predict_proba, X[gsel])]
syn_cfg = load_config("smoke")
syn_raw = sd.make_synthetic(syn_cfg)
syn = pd.read_csv(syn_raw / "train.csv")
im_ = ImageModel(APPM)
g_aps = []
for k, (iid, g_) in enumerate(list(syn.groupby("id"))[:4]):
    from dl03.rle import instances_to_label_map
    lbl_ = instances_to_label_map(g_["annotation"], (int(g_["height"].iloc[0]), int(g_["width"].iloc[0])))
    img_ = skio.imread(syn_raw / "train" / f"{iid}.png")
    skio.imsave(G / f"img_{k}.png", img_, check_contrast=False)
    np.savez_compressed(G / f"img_{k}.npz", lbl=lbl_)
    g_aps.append(se.sm.sartorius_score(lbl_, im_.predict(img_.astype(np.float32))))
golden = {"text_f1_macro": float(f1_score([classes[k] for k in y[gsel]], g_pred, average="macro")),
          "image_ap": float(np.mean(g_aps)), "version": VERSION}
(G / "golden.json").write_text(json.dumps(golden, indent=1))
import sklearn  # noqa: E402

(APPM / "versions.json").write_text(json.dumps({"scikit-learn": sklearn.__version__, "numpy": np.__version__,
                                                "onnxruntime": ort.__version__}))
golden

# %% [markdown]
# ## 9.4 La API real: prueba funcional y prueba de carga
# Se levanta `uvicorn app.main:app` en segundo plano con los artefactos recién exportados.

# %%
import httpx  # noqa: E402

port = 8765
api_log = open(cfg.out_dir / "api_carga.log", "w")       # registro estructurado de la API (no ensucia el notebook)
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level",
                        "warning"], cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": f"{ROOT}/src:{ROOT}"},
                       stdout=api_log, stderr=subprocess.STDOUT)
base = f"http://127.0.0.1:{port}"
for _ in range(60):
    try:
        if httpx.get(f"{base}/ready", timeout=2).json().get("ready"):
            break
    except Exception:
        time.sleep(1)
print(httpx.get(f"{base}/ready").json())
r_t = httpx.post(f"{base}/predict_text", json={"text": X[te[0]]}, timeout=60).json()
buf = io.BytesIO()
from PIL import Image  # noqa: E402

Image.fromarray(se.native_image(images, splits.test[0]).astype(np.uint8)).save(buf, "PNG")
b64 = base64.b64encode(buf.getvalue()).decode()
r_i = httpx.post(f"{base}/predict_image", json={"image_png_base64": b64}, timeout=120).json()
print("texto:", r_t["label"], "(verdadera:", classes[y[te[0]]], ") ·", round(r_t["latency_ms"], 1), "ms")
print("imagen:", r_i["n_instances"], "instancias (referencia:", int(sd.load_label(cfg, splits.test[0]).max()), ") ·",
      round(r_i["latency_ms"], 1), "ms")
bad = httpx.post(f"{base}/predict_text", json={"text": ""}).status_code
print("validación de entrada (texto vacío) →", bad)


def load_test(endpoint, payload, conc, n):
    def one(_):
        t0 = time.perf_counter()
        try:
            ok = httpx.post(f"{base}{endpoint}", json=payload, timeout=120).status_code == 200
        except Exception:
            ok = False
        return (time.perf_counter() - t0) * 1e3, ok
    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(one, range(n)))
    wall = time.perf_counter() - t0
    lat = np.array([r[0] for r in res])
    return {"concurrencia": conc, "p50 ms": np.percentile(lat, 50), "p95 ms": np.percentile(lat, 95),
            "p99 ms": np.percentile(lat, 99), "throughput (req/s)": n / wall,
            "errores": 1 - np.mean([r[1] for r in res])}


lt = []
for conc in [1, 2, 4, 8, 16]:
    lt.append({"endpoint": "/predict_text", **load_test("/predict_text", {"text": X[te[1]]}, conc, 40 * conc // 2 + 20)})
    lt.append({"endpoint": "/predict_image", **load_test("/predict_image", {"image_png_base64": b64}, conc,
                                                         4 * conc + 4)})
lt = pd.DataFrame(lt)
srv.terminate()
api_log.close()
print("líneas de registro JSON de la API:", sum(1 for _ in open(cfg.out_dir / "api_carga.log")))
save_table(cfg, "mlops_carga", lt)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for k, (e, g) in enumerate(lt.groupby("endpoint")):
    ax[0].plot(g["concurrencia"], g["throughput (req/s)"], "o-", color=PALETTE[k], label=e)
    ax[1].plot(g["concurrencia"], g["p95 ms"], "o-", color=PALETTE[k], label=e)
ax[0].set(xscale="log", xlabel="concurrencia", ylabel="req/s", title="Rendimiento (saturación cuando se aplana)")
ax[1].set(xscale="log", yscale="log", xlabel="concurrencia", ylabel="p95 (ms)", title="Latencia p95")
ax[0].legend()
savefig(cfg, "09_carga")
plt.show()
lt.round(2)

# %% [markdown]
# ## 9.5 Experimento de deriva (RQ8)
# **Diseño.** Referencia = validación. Flujo simulado en ventanas de 40 documentos / 8 imágenes tomadas del test.
# Deriva inyectada con intensidad creciente (0 = sin deriva):
# * Texto: (a) cambio de vocabulario (errores tipográficos y palabras fuera de vocabulario), (b) cambio de tema
#   (la mezcla de categorías se concentra en 3 categorías), (c) cambio de longitud (truncamiento).
# * Imágenes: (a) desenfoque, (b) ruido, (c) iluminación, (d) cambio de tipo celular (la ventana se llena de un solo
#   tipo).
#
# **Detectores.** PSI y KS sobre rasgos escalares (longitud, confianza máxima, nº de instancias, intensidad media,
# varianza del Laplaciano) y MMD con prueba de permutación sobre embeddings (TF-IDF→SVD en texto; vector de rasgos
# en imágenes). **Medidas.** Poder (AUC del estadístico entre ventanas sin y con deriva, por intensidad), tasa de
# falsas alarmas al umbral del percentil 95 de las ventanas sin deriva, retardo de detección en un flujo con deriva
# gradual y caída de la métrica asociada.

# %%
from sklearn.decomposition import TruncatedSVD  # noqa: E402

rng = np.random.default_rng(cfg.seed)
va = np.array(sp["val"])
text_model = joblib.load(APPM / "text_model.joblib")
vec = text_model.vec
svd = TruncatedSVD(20 if not cfg.is_smoke else 5, random_state=0).fit(vec.transform([nd.CLEANERS[text_model.prep](t)
                                                                                     for t in X[va]]))


def text_feats(texts):
    P = text_model.predict_proba(list(texts))
    E = svd.transform(vec.transform([nd.CLEANERS[text_model.prep](t) for t in texts]))
    return {"longitud": np.array([len(t.split()) for t in texts]), "confianza": P.max(1), "emb": E, "pred": P.argmax(1)}


def corrupt(t, level):
    w = t.split()
    k = int(level * 0.3 * len(w))
    for i in rng.choice(len(w), k, replace=False) if k else []:
        w[i] = "".join(rng.permutation(list(w[i]))) + "zq"
    return " ".join(w)


def text_window(level, kind, n=40):
    if kind == "tema" and level > 0:
        focus = rng.choice(K, 3, replace=False)
        pool = te[np.isin(y[te], focus)] if rng.random() < level else te
        idx = rng.choice(pool, n)
    else:
        idx = rng.choice(te, n)
    texts = list(X[idx])
    if kind == "vocabulario":
        texts = [corrupt(t, level) for t in texts]
    if kind == "longitud" and level > 0:
        texts = [" ".join(t.split()[: max(5, int(len(t.split()) * (1 - 0.9 * level)))]) for t in texts]
    return texts, y[idx]


ref_t = text_feats(X[va])
LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_WIN = 30 if not cfg.is_smoke else 6
rows = []
for kind in ["vocabulario", "tema", "longitud"]:
    for lev in LEVELS:
        for w in range(N_WIN):
            texts, yy = text_window(lev, kind)
            f = text_feats(texts)
            rows.append({"dominio": "texto", "deriva": kind, "intensidad": lev, "ventana": w,
                         "PSI longitud": drift.psi(ref_t["longitud"], f["longitud"]),
                         "KS confianza": drift.drift_report(ref_t["confianza"], f["confianza"])["ks_stat"],
                         "MMD emb": drift.mmd2(ref_t["emb"][rng.choice(len(va), 120)], f["emb"]),
                         "F1 ventana": ne.f1m(yy, np.eye(K)[f["pred"]]) if len(set(yy)) > 1 else np.nan,
                         "exactitud ventana": float((f["pred"] == yy).mean())})
dt = pd.DataFrame(rows)

# %%
from scipy import ndimage as ndi  # noqa: E402
from skimage import filters  # noqa: E402

sys.path.insert(0, str(ROOT))
from app.inference import ImageModel  # noqa: E402

im_model = ImageModel(APPM)
meta = images.set_index("id")
img_pool = splits.test if not cfg.is_smoke else splits.test[:8]
cache = {}


def img_feats(x255, iid=None, key=None):
    lbl = im_model.predict(x255)
    lap = filters.laplace(x255 / 255.0).var()
    out = {"intensidad": x255.mean(), "laplaciano": lap, "n_inst": lbl.max(),
           "ap": se.sm.sartorius_score(sd.load_label(cfg, iid), lbl) if iid else np.nan}
    return out


def perturb(x, kind, lev):
    if lev == 0 or kind == "tipo celular":
        return x
    if kind == "desenfoque":
        return filters.gaussian(x, 3 * lev, preserve_range=True)
    if kind == "ruido":
        return np.clip(x + rng.normal(0, 30 * lev, x.shape), 0, 255)
    return np.clip(x * (1 - 0.6 * lev * np.linspace(0, 1, x.shape[1])[None, :]), 0, 255)


ref_i = pd.DataFrame([img_feats(se.native_image(images, i), i) for i in (splits.val if not cfg.is_smoke
                                                                           else splits.val[:8])])
irows = []
N_IW = 12 if not cfg.is_smoke else 3
for kind in ["desenfoque", "ruido", "iluminación", "tipo celular"]:
    for lev in [0.0, 0.5, 1.0]:
        for w in range(N_IW):
            if kind == "tipo celular" and lev > 0:
                ct = rng.choice(sorted(meta.cell_type.unique()))
                pool = [i for i in img_pool if meta.loc[i, "cell_type"] == ct] or img_pool
                ids = list(rng.choice(pool, 8)) if rng.random() < lev else list(rng.choice(img_pool, 8))
            else:
                ids = list(rng.choice(img_pool, 8))
            fs = []
            for i in ids:
                k_ = (i, kind, lev) if kind != "ruido" else None
                if k_ is not None and k_ in cache:
                    fs.append(cache[k_])
                    continue
                f_ = img_feats(perturb(se.native_image(images, i), kind, lev), i)
                if k_ is not None:
                    cache[k_] = f_
                fs.append(f_)
            fs = pd.DataFrame(fs)
            vec_ref = ref_i[["intensidad", "laplaciano", "n_inst"]].values
            vec_cur = fs[["intensidad", "laplaciano", "n_inst"]].values
            mu, sdv = vec_ref.mean(0), vec_ref.std(0) + 1e-9
            irows.append({"dominio": "imagen", "deriva": kind, "intensidad": lev, "ventana": w,
                          "PSI intensidad": drift.psi(ref_i["intensidad"], fs["intensidad"]),
                          "KS laplaciano": drift.drift_report(ref_i["laplaciano"], fs["laplaciano"])["ks_stat"],
                          "MMD rasgos": drift.mmd2((vec_ref - mu) / sdv, (vec_cur - mu) / sdv),
                          "AP ventana": fs["ap"].mean()})
di = pd.DataFrame(irows)
dt.to_csv(cfg.out_dir / "tables" / "drift_texto.csv", index=False)
di.to_csv(cfg.out_dir / "tables" / "drift_imagen.csv", index=False)

# %%
def power_table(d, stats_):
    out = []
    for (kind), g in d.groupby("deriva"):
        base = g[g.intensidad == 0]
        for s_ in stats_:
            thr = np.nanpercentile(base[s_], 95)
            for lev in sorted(g.intensidad.unique()):
                cur = g[g.intensidad == lev]
                row = {"deriva": kind, "detector": s_, "intensidad": lev,
                       "tasa de alarma (umbral p95 sin deriva)": float((cur[s_] > thr).mean())}
                if lev > 0:
                    row["AUC del detector"] = drift.detection_roc(base[s_].dropna(), cur[s_].dropna())
                out.append(row)
    return pd.DataFrame(out)


pt = power_table(dt, ["PSI longitud", "KS confianza", "MMD emb"])
pi = power_table(di, ["PSI intensidad", "KS laplaciano", "MMD rasgos"])
power = pd.concat([pt.assign(dominio="texto"), pi.assign(dominio="imagen")])
save_table(cfg, "drift_poder", power)
fig, ax = plt.subplots(2, 4, figsize=(20, 7))
for j, (kind, g) in enumerate(pt.groupby("deriva")):
    for k, (s_, h) in enumerate(g.groupby("detector")):
        ax[0, j].plot(h.intensidad, h["tasa de alarma (umbral p95 sin deriva)"], "o-", color=PALETTE[k], label=s_)
    m_ = dt[dt.deriva == kind].groupby("intensidad")["exactitud ventana"].mean()
    ax[0, j].plot(m_.index, m_.values, "k--", label="exactitud del modelo")
    ax[0, j].set(title=f"texto · {kind}", xlabel="intensidad", ylim=(-0.05, 1.05))
ax[0, 0].legend(fontsize=7)
ax[0, 3].axis("off")
for j, (kind, g) in enumerate(pi.groupby("deriva")):
    for k, (s_, h) in enumerate(g.groupby("detector")):
        ax[1, j].plot(h.intensidad, h["tasa de alarma (umbral p95 sin deriva)"], "o-", color=PALETTE[k], label=s_)
    m_ = di[di.deriva == kind].groupby("intensidad")["AP ventana"].mean()
    ax[1, j].plot(m_.index, m_.values, "k--", label="AP del modelo")
    ax[1, j].set(title=f"imagen · {kind}", xlabel="intensidad", ylim=(-0.05, 1.05))
ax[1, 0].legend(fontsize=7)
plt.tight_layout()
savefig(cfg, "09_deriva")
plt.show()
power[power.intensidad == 1.0].pivot_table(index=["dominio", "deriva"], columns="detector",
                                           values="AUC del detector").round(3)

# %% [markdown]
# ### Retardo de detección en un flujo con deriva gradual
# 60 ventanas de texto: las 20 primeras sin deriva y luego la intensidad del cambio de vocabulario crece
# linealmente. Alarma cuando el estadístico supera el percentil 95 de las ventanas sin deriva en 2 ventanas
# consecutivas (para reducir falsas alarmas).

# %%
thr = np.nanpercentile(dt[(dt.deriva == "vocabulario") & (dt.intensidad == 0)]["MMD emb"], 95)
stream = []
n_s = 60 if not cfg.is_smoke else 12
onset = n_s // 3
for t in range(n_s):
    lev = 0.0 if t < onset else min(1.0, (t - onset) / (n_s - onset))
    texts, yy = text_window(lev, "vocabulario")
    f = text_feats(texts)
    stream.append({"t": t, "intensidad": lev, "MMD": drift.mmd2(ref_t["emb"][rng.choice(len(va), 120)], f["emb"]),
                   "exactitud": float((f["pred"] == yy).mean())})
stream = pd.DataFrame(stream)
over = (stream["MMD"] > thr).astype(int)
alarm = stream.t[(over & over.shift(1, fill_value=0)).astype(bool)]
first = int(alarm.iloc[0]) if len(alarm) else None
false_alarms = int(((over & over.shift(1, fill_value=0)).astype(bool) & (stream.t < onset)).sum())
rq8 = {"inicio de la deriva (ventana)": onset, "primera alarma": first,
       "retardo (ventanas)": None if first is None else first - onset,
       "intensidad al detectar": None if first is None else float(stream.loc[first, "intensidad"]),
       "exactitud antes": float(stream[stream.t < onset].exactitud.mean()),
       "exactitud al detectar": None if first is None else float(stream.loc[first, "exactitud"]),
       "falsas alarmas antes del inicio": false_alarms}
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(stream.t, stream["MMD"], label="MMD²")
ax.axhline(thr, ls="--", c="gray", label="umbral (p95 sin deriva)")
ax.axvline(onset, c=PALETTE[1], label="inicio de deriva")
if first is not None:
    ax.axvline(first, c=PALETTE[3], label="alarma")
ax2 = ax.twinx()
ax2.plot(stream.t, stream.exactitud, color="k", alpha=0.4, label="exactitud")
ax.set(xlabel="ventana", title="Flujo con deriva gradual de vocabulario")
ax.legend(fontsize=7, loc="upper left")
savefig(cfg, "09_flujo_deriva")
plt.show()
save_json(cfg, "drift_RQ8", rq8)
rq8

# %% [markdown]
# **Política de alerta y reentrenamiento propuesta.**
# * Umbral: percentil 95 del estadístico en ventanas sin deriva, con confirmación en 2 ventanas consecutivas
#   (reduce falsas alarmas a costa de una ventana de retardo).
# * Costos: una falsa alarma cuesta una revisión humana; una deriva no detectada cuesta decisiones con exactitud
#   degradada. En reclutamiento el segundo costo es mayor, por eso se prefiere un detector sensible (MMD sobre
#   embeddings) con confirmación, más PSI de longitud como señal barata.
# * Despliegue: nueva versión primero en **sombra** (predice sin efecto, se comparan distribuciones), luego
#   **canario** (5–10 % del tráfico con monitoreo de la métrica con etiquetas tardías) y **reversión** inmediata con
#   el registro de modelos (`python -m app.registry rollback dl03`).

# %% [markdown]
# ## 9.6 Model cards y datasheets
# Se generan a partir de las tablas de resultados (no se escriben cifras a mano).

# %%
rep = ROOT / "reports"
(rep / "model_cards").mkdir(parents=True, exist_ok=True)
(rep / "datasheets").mkdir(parents=True, exist_ok=True)
eq = pd.read_csv(cfg.out_dir / "tables" / "nlp_equidad_subgrupos.csv", index_col=0)
cf = pd.read_csv(cfg.out_dir / "tables" / "nlp_contrafactual.csv", index_col=0)
seg_ct = img_metrics[img_metrics.model == "unetpp"].groupby("cell_type")["ap"].mean()
(rep / "model_cards" / "texto.md").write_text(f"""# Model card · Clasificador de categoría de cargo ({text_choice})

* **Versión:** text-{text_choice}-{VERSION} · **commit:** {info['commit'][:10]}
* **Uso previsto:** sugerir la categoría de cargo de una hoja de vida en inglés para enrutarla; siempre con revisión
  humana. **Usos no previstos:** descartar, puntuar o rankear candidatos; decidir contrataciones; otros idiomas.
* **Datos:** Jarvis Calling Hiring Contest (Kaggle, CC0), {len(df)} documentos, 24 categorías; partición 70/15/15
  agrupando casi-duplicados.
* **Desempeño en test (F1 macro, media de semillas):** {f1_by[text_choice]:.3f}. Mejor modelo evaluado:
  {f1_by.index[0]} ({f1_by.iloc[0]:.3f}).
* **Equidad:** sin atributos protegidos en los datos; pruebas contrafactuales (cambio de argmax por atributo):
  {cf[cf['modelo'].str.contains('LR|XGB|NB', regex=True)].to_dict('records')}.
  Métricas por subgrupo con proxies de pronombres (débiles): {eq.to_dict('records')}.
* **Limitaciones y riesgos:** atajo del título del cargo (capítulo 8.2), etiquetas ruidosas (capítulo 6), texto
  truncado, contexto de alto riesgo según el Reglamento (UE) 2024/1689 (anexo III). Mitigación: revisión humana,
  monitoreo de deriva, auditoría periódica con atributos reales recogidos con consentimiento.
""")
(rep / "model_cards" / "imagen.md").write_text(f"""# Model card · Segmentación de instancias celulares (U-Net++ + watershed, ONNX)

* **Versión:** image-unetpp-{VERSION} · **commit:** {info['commit'][:10]}
* **Uso previsto:** segmentar y contar células en microscopía de contraste de fase similar a Sartorius (SH-SY5Y,
  astrocitos, neuronas corticales), para investigación. **No previsto:** diagnóstico clínico; otros microscopios o
  protocolos sin revalidación.
* **Desempeño en test (AP@[.5:.95] estilo Sartorius):** {ap_by['unetpp']:.3f}; por tipo celular:
  {seg_ct.round(3).to_dict()}. Mejor modelo evaluado: {ap_by.index[0]} ({ap_by.iloc[0]:.3f}).
* **Exportación:** ONNX, paridad máx |Δp| = {par_img:.1e} (tolerancia {TOL}).
* **Limitaciones:** sensibilidad a densidad (fusiones), a desenfoque e iluminación (capítulo 5.6) y a tipos celulares
  no vistos (capítulo 5.5).
""")
for name, path in [("sartorius", "sartorius"), ("resumes", "resumes")]:
    m = manifests.get(name, {})
    (rep / "datasheets" / f"{name}.md").write_text(
        f"# Datasheet · {name}\n\n* **Origen:** {m.get('source', 'sintético')}\n"
        f"* **Hash:** {m.get('zip_sha256', m.get('csv_sha256', 'n/a'))}\n"
        "* **Composición, recolección, usos y riesgos:** ver capítulos 3 y 6 del libro (tablas en results/tables).\n")
print(sorted(p.name for p in (rep / "model_cards").iterdir()))
