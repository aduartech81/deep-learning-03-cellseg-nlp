"""Tablero del proyecto 03: segmentación de instancias celulares y clasificación de hojas de vida.

Lee solo resultados ya calculados (results/tables, results/figs, results/seg, results/nlp); no entrena nada.
Si API_URL apunta a la API desplegada (app/main.py), la página «Probar los modelos» consulta en vivo.

Local:   RESULTS_DIR=results python dashboard/app.py  -> http://127.0.0.1:8050
Render:  gunicorn --chdir dashboard app:server
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html
from flask import abort, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / os.environ.get("RESULTS_DIR", "results")
TAB, FIGS = RES / "tables", RES / "figs"
BOOK = "https://aduartech81.github.io/deep-learning-03-cellseg-nlp/"
REPO = "https://github.com/aduartech81/deep-learning-03-cellseg-nlp"
API_URL = os.environ.get("API_URL", "").rstrip("/")
SMOKE = RES.name.endswith("smoke")

S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#52514e"]
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e8e7e1", "#c3c2b7", "#ffffff"
CT = {"shsy5y": S[0], "astro": S[1], "cort": S[2]}
SEG_NAMES = {"classical": "Baseline clásico", "maskrcnn": "Mask R-CNN (R50-FPN)", "hovernet": "Estilo Hover-Net",
             "unetpp": "U-Net++ + watershed", "cellpose_ft": "Cellpose ajustado",
             "cellpose_zero": "Cellpose cyto2 cero-shot", "sam_auto": "SAM cero-shot", "sam_boxes": "SAM + cajas",
             "sam_ft": "SAM decod. ajustado"}


# ------------------------------------------------------------------------------------------------ utilidades
def csv(name, **kw):
    p = TAB / f"{name}.csv"
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


def js(name):
    p = TAB / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def ci(s):
    """'0.123 [0.100, 0.150]' -> (0.123, 0.100, 0.150)"""
    try:
        a, rest = str(s).split(" [")
        lo, hi = rest.rstrip("]").split(", ")
        return float(a), float(lo), float(hi)
    except ValueError:
        return np.nan, np.nan, np.nan


def fmt(x, d=3):
    try:
        return f"{float(x):,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return str(x)


def base_layout(fig, height=380, **kw):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=16, b=10),
                      font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=13, color=INK2),
                      paper_bgcolor=SURF, plot_bgcolor=SURF, separators=",.",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None, font=dict(size=12)),
                      hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK, size=12)), **kw)
    fig.update_xaxes(gridcolor=GRID, linecolor=BASE, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=BASE, zeroline=False)
    return fig


def graph(fig):
    return dcc.Graph(figure=fig, style={"height": f"{fig.layout.height or 380}px"}, config={"displayModeBar": False})


def kpi(value, label, accent=False):
    return html.Div([html.Div(value, className="kpi-value" + (" accent" if accent else "")),
                     html.Div(label, className="kpi-label")], className="kpi")


def head(eyebrow, title, lead):
    return html.Div([html.Div(eyebrow, className="eyebrow"), html.H1(title), html.P(lead, className="lead")],
                    className="section-head")


def card(title, body, note=None):
    return html.Div([html.H3(title), body] + ([html.P(note, className="note")] if note else []), className="viz-card")


def fig_img(name, caption):
    if not (FIGS / f"{name}.png").exists():
        return html.P(f"(figura {name} pendiente)", className="note")
    return html.Figure([html.Img(src=f"/figs/{name}.png", alt=caption), html.Figcaption(caption)],
                       className="static-fig")


def table(df, page_size=12):
    if df is None or len(df) == 0:
        return html.P("(tabla pendiente)", className="note")
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: fmt(v, 3))
    return dash_table.DataTable(
        data=df.to_dict("records"), columns=[{"name": str(c), "id": str(c)} for c in df.columns], page_size=page_size,
        sort_action="native", style_as_list_view=True, style_table={"overflowX": "auto"},
        style_header={"fontWeight": 600, "backgroundColor": "#f6f6f3", "color": INK, "border": "none", "fontSize": 12.5},
        style_cell={"fontSize": 13, "padding": "7px 10px", "color": INK2, "textAlign": "left",
                    "borderBottom": f"1px solid {GRID}", "whiteSpace": "normal", "height": "auto",
                    "fontFamily": "Inter, system-ui, sans-serif", "maxWidth": 420})


def ci_bars(df, col, title_x, color=S[0], height=None):
    v = df[col].map(ci).apply(pd.Series)
    v.columns = ["est", "lo", "hi"]
    v = v.sort_values("est")
    fig = go.Figure(go.Bar(y=v.index, x=v["est"], orientation="h", marker_color=color,
                           error_x=dict(type="data", symmetric=False, array=v["hi"] - v["est"],
                                        arrayminus=v["est"] - v["lo"], color=INK2, thickness=1.2),
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    base_layout(fig, height=height or 60 + 34 * len(v))
    fig.update_xaxes(title=title_x)
    return fig


# ------------------------------------------------------------------------------------------------ datos
seg_res = csv("seg_resultados_test", index_col=0)
nlp_res = csv("nlp_resultados_test", index_col=0)
seg_img = pd.read_csv(RES / "seg" / "metrics_test.csv") if (RES / "seg" / "metrics_test.csv").exists() else \
    pd.DataFrame()
best_seg = seg_res["ap"].map(lambda s: ci(s)[0]).idxmax() if len(seg_res) else "—"
best_nlp = nlp_res["f1_macro"].map(lambda s: ci(s)[0]).idxmax() if len(nlp_res) else "—"
hyp = csv("hipotesis", index_col=0)


# ------------------------------------------------------------------------------------------------ páginas
def page_inicio():
    ap = ci(seg_res.loc[best_seg, "ap"])[0] if len(seg_res) else np.nan
    f1 = ci(nlp_res.loc[best_nlp, "f1_macro"])[0] if len(nlp_res) else np.nan
    n_img = len(seg_img["id"].unique()) if len(seg_img) else 0
    qs = [(r.Index, f"{r.veredicto}. {str(r.evidencia)[:220]}") for r in hyp.itertuples()] if len(hyp) else []
    return html.Div([
        html.Div([html.Div([
            html.Div("Andrés Duarte Chaves · Deep Learning · Maestría en Ingeniería Biomédica, Uninorte",
                     className="eyebrow"),
            html.H1("Contar células y leer hojas de vida: qué modelo, cuándo y con qué riesgo", className="hero-title"),
            html.P(["Una investigación empírica pre-registrada en dos dominios: ", html.B("segmentación de instancias "
                    "celulares"), " en microscopía de contraste de fase (Sartorius) con seis paradigmas, y ",
                    html.B("clasificación de hojas de vida"), " en 24 categorías con seis familias de modelos. Mismo "
                    "estándar para ambos: test sellado, varias semillas, intervalos de confianza, pruebas pareadas con "
                    "corrección de Holm, ablaciones, robustez, calibración, equidad y despliegue monitoreado."],
                   className="lead"),
            html.Div([html.Span(t, className="chip") for t in ["U-Net++ propia", "Mask R-CNN", "Cellpose", "SAM",
                                                               "DistilBERT", "BiLSTM", "Equidad", "MLOps"]],
                     className="chips"),
            html.Div([dbc.Button("Resultados de segmentación →", href="/segmentacion", className="btn-primary-soft"),
                      dbc.Button("Jupyter Book completo", href=BOOK, target="_blank", className="btn-ghost")],
                     className="hero-actions")], className="hero-text"),
            html.Div(fig_img("03_rle_overlay", "Instancias anotadas sobre las tres líneas celulares"),
                     className="hero-art")], className="hero"),
        html.Div([kpi(fmt(ap), f"AP@[.5:.95] del mejor modelo de segmentación ({best_seg})", accent=True),
                  kpi(fmt(f1), f"F1 macro del mejor clasificador de texto ({best_nlp})", accent=True),
                  kpi(str(n_img), "imágenes en el test sellado (agrupado por placa/fecha)"),
                  kpi(str(len(seg_res) + len(nlp_res)), "modelos evaluados con el mismo protocolo")],
                 className="kpi-row"),
        html.H2("Hipótesis pre-registradas", className="sub"),
        html.Div([html.Div([html.P(q, className="q-text"), html.P(a, className="q-answer")], className="q-card")
                  for q, a in qs], className="q-grid") if qs else html.P("(pendiente)", className="note"),
    ])


def page_datos():
    return html.Div([
        head("Auditoría antes de modelar", "Los datos", "Qué hay en cada conjunto, qué sesgos tiene y qué decisiones "
             "de diseño se derivan de ello."),
        html.H2("Imágenes de microscopía (Sartorius)", className="sub"),
        fig_img("03_cualitativa", "Fila superior: 5 imágenes al azar. Inferior: casos extremos (más/menos células, "
                "peor enfoque, menor contraste, mayor gradiente de iluminación)."),
        fig_img("03_distribuciones", "Tamaño de imagen, células por imagen, área (log), densidad por tipo celular."),
        card("Tamaños de célula y resolución", table(csv("seg_sizes").round(2)),
             "Fracción de instancias que caen por debajo de 10 px al reescalar el ancho a 512 o 256."),
        fig_img("03_batch_effects", "Efectos de lote: embeddings de un codificador preentrenado por tipo y por grupo."),
        html.H2("Hojas de vida (Jarvis Calling)", className="sub"),
        fig_img("06_clases", "Documentos por categoría."),
        fig_img("06_longitud", "Longitud en tokens y por categoría: casi todos superan las longitudes de la rejilla."),
        card("Atajos y calidad", html.Pre(json.dumps(js("nlp_atajos") | js("nlp_duplicados") | js("nlp_ruido_etiquetas"),
                                                     indent=1, ensure_ascii=False)[:2500], className="mono")),
    ])


def page_seg():
    blocks = [head("Bloque 1", "Segmentación de instancias celulares",
                   "Seis paradigmas evaluados con el mismo código sobre el mismo test sellado. Unidad de análisis: "
                   "imagen; IC bootstrap de 10 000 remuestreos.")]
    if len(seg_res):
        blocks += [card("AP@[.5:.95] en test (media [IC 95 %])", graph(ci_bars(seg_res, "ap", "AP@[.5:.95]"))),
                   card("Tabla de resultados", table(seg_res.reset_index().rename(columns={"index": "modelo"})))]
    if len(seg_img):
        h = seg_img.groupby(["model", "cell_type"])["ap"].mean().unstack()
        h.index = [SEG_NAMES.get(m, m) for m in h.index]
        fig = go.Figure(go.Heatmap(z=h.values, x=h.columns, y=h.index, colorscale="Blues", text=np.round(h.values, 2),
                                   texttemplate="%{text}", hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>"))
        blocks.append(card("AP por tipo celular (¿cambia el ranking? H3)", graph(base_layout(fig, 60 + 34 * len(h)))))
    blocks += [fig_img("04_cd_diagram", "Diagrama de diferencia crítica (Friedman + Wilcoxon–Holm)."),
               fig_img("04_taxonomia", "Composición de los errores de instancia por modelo y tipo celular."),
               fig_img("04_casos_peores", "Los 5 peores casos del mejor modelo: rojo = falso positivo, azul = falso "
                       "negativo."),
               fig_img("04_comparacion_modelos", "Mismas imágenes, todos los modelos (mapas de error)."),
               card("Conteo celular (utilidad práctica)", table(csv("seg_conteo")))]
    return html.Div(blocks)


def page_abl():
    rob = pd.read_csv(RES / "seg" / "robustness.csv") if (RES / "seg" / "robustness.csv").exists() else pd.DataFrame()
    fr = pd.read_csv(RES / "seg" / "data_efficiency.csv") if (RES / "seg" / "data_efficiency.csv").exists() else \
        pd.DataFrame()
    out = [head("Por qué funciona", "Ablaciones, eficiencia de datos y robustez",
                "Qué decisiones de diseño importan, cuántos datos hacen falta y cómo se degrada cada modelo.")]
    out.append(card("Ablaciones (U-Net++, ≥ 3 semillas, validación B)", table(csv("seg_ablaciones"))))
    if len(fr):
        fig = go.Figure()
        for k, (m, g) in enumerate(fr.groupby("model")):
            c = g.groupby("frac")["ap"].agg(["mean", "std"])
            fig.add_trace(go.Scatter(x=c.index * 100, y=c["mean"], error_y=dict(array=c["std"]), name=m,
                                     mode="lines+markers", line=dict(color=S[k])))
        base_layout(fig, 340)
        fig.update_xaxes(title="% del train", type="log")
        fig.update_yaxes(title="AP (test)")
        out.append(card("Eficiencia de datos (H1)", graph(fig), json.dumps(js("seg_H1").get("H1", ""))))
    if len(rob):
        fig = go.Figure()
        for k, ((p, m), g) in enumerate(rob.groupby(["perturbación", "modelo"])):
            c = g.groupby("severidad")["ap"].mean()
            fig.add_trace(go.Scatter(x=c.index, y=c.values, name=f"{p} · {m}", mode="lines+markers",
                                     line=dict(color=S[k % len(S)], dash="solid" if "U-Net" in m else "dot")))
        base_layout(fig, 420)
        fig.update_xaxes(title="severidad")
        fig.update_yaxes(title="AP")
        out.append(card("Degradación bajo perturbaciones", graph(fig)))
    out += [card("Dejar un tipo celular fuera", table(csv("seg_loco"))),
            fig_img("05_calibracion_incertidumbre", "Calibración de píxel y riesgo–cobertura por incertidumbre."),
            fig_img("05_gradcam", "Grad-CAM de U-Net++."), fig_img("05_pareto", "Frente de Pareto AP vs latencia.")]
    return html.Div(out)


def page_nlp():
    out = [head("Bloque 2", "Clasificación de hojas de vida",
                "Mismo presupuesto de búsqueda para los modelos ligeros, validación anidada, 5 semillas y bootstrap "
                "jerárquico (semillas × documentos).")]
    if len(nlp_res):
        out += [card("F1 macro en test (media [IC 95 %])", graph(ci_bars(nlp_res, "f1_macro", "F1 macro", S[1]))),
                card("Tabla de resultados", table(nlp_res.reset_index().rename(columns={"index": "modelo"})[
                    ["modelo", "accuracy", "f1_macro", "mcc", "roc_auc_ovr", "ece", "latencia p50 (ms)",
                     "parámetros"]]))]
    out += [card("Validación anidada: desempeño del procedimiento y sesgo optimista",
                 table(csv("nlp_anidada").round(4))),
            card("H4: DistilBERT vs TF-IDF + XGBoost", html.Pre(json.dumps(js("nlp_H4"), indent=1, ensure_ascii=False),
                                                                className="mono")),
            fig_img("07_bilstm_sensibilidad", "Sensibilidad del BiLSTM a sus hiperparámetros (rejilla de 576)."),
            card("Preprocesamiento por familia", table(csv("nlp_preprocesado"))),
            fig_img("07_confusion", "Matriz de confusión normalizada del mejor modelo."),
            card("Calibración y temperature scaling (H5)", table(csv("nlp_calibracion"))),
            fig_img("08_riesgo_cobertura", "Clasificación selectiva: cuántos casos derivar a revisión humana.")]
    return html.Div(out)


def page_equidad():
    return html.Div([
        head("Contexto de reclutamiento", "Equidad, atajos y riesgo",
             "Un modelo que asigna categorías de cargo a hojas de vida opera en un dominio de alto riesgo."),
        card("Atajo del título: F1 con y sin la primera línea", table(csv("nlp_atajo_titulo"))),
        card("Pruebas contrafactuales (cambio de predicción al alterar un marcador)", table(csv("nlp_contrafactual"))),
        card("Métricas por subgrupo con proxies (débiles, declarados)", table(csv("nlp_equidad_subgrupos"))),
        card("Invarianza (CheckList)", table(csv("nlp_checklist_invarianza"))),
        html.Div([html.H3("¿Debería decidir automáticamente sobre personas?"),
                  html.P("No. Como mucho, sugerir la categoría para enrutar la hoja de vida, con revisión humana de "
                         "los casos de baja confianza, auditoría periódica con atributos reales recogidos con "
                         "consentimiento y prohibición de usarlo para descartar candidatos. El Reglamento (UE) "
                         "2024/1689 clasifica estos sistemas como de alto riesgo (anexo III).")], className="viz-card"),
    ])


def page_mlops():
    lt = csv("mlops_carga")
    out = [head("Del notebook al servicio", "MLOps, carga y deriva (RQ8)",
                "Modelos exportados (ONNX, joblib) con paridad numérica, API probada en CI, pruebas de carga y un "
                "experimento de deriva con poder y retardo de detección medidos.")]
    if len(lt):
        fig = go.Figure()
        for k, (e, g) in enumerate(lt.groupby("endpoint")):
            fig.add_trace(go.Scatter(x=g["concurrencia"], y=g["throughput (req/s)"], name=e, mode="lines+markers",
                                     line=dict(color=S[k])))
        base_layout(fig, 320)
        fig.update_xaxes(title="concurrencia", type="log")
        fig.update_yaxes(title="req/s")
        out.append(card("Rendimiento bajo carga concurrente", graph(fig)))
    out += [fig_img("09_deriva", "Tasa de alarma de cada detector y métrica del modelo vs intensidad de la deriva."),
            fig_img("09_flujo_deriva", "Flujo con deriva gradual: inicio, alarma y caída de la exactitud."),
            card("RQ8", html.Pre(json.dumps(js("drift_RQ8"), indent=1, ensure_ascii=False), className="mono")),
            card("Exportación ONNX", table(csv("mlops_onnx")))]
    return html.Div(out)


def page_api():
    return html.Div([
        head("En vivo", "Probar los modelos desplegados",
             "La API (FastAPI) sirve el clasificador de texto (scikit-learn) y la U-Net++ exportada a ONNX. En el plan "
             "gratuito el servicio se duerme: la primera llamada puede tardar ~1 minuto."),
        html.Div([html.H3("Texto: categoría de una hoja de vida (en inglés)"),
                  dcc.Textarea(id="t-in", value="Experienced registered nurse with ICU and patient care background, "
                               "medication administration, charting and family education.",
                               style={"width": "100%", "height": 120}),
                  dbc.Button("Clasificar", id="t-go", className="btn-primary-soft"), html.Div(id="t-out")],
                 className="viz-card"),
        html.Div([html.H3("Imagen: segmentar células (PNG en escala de grises)"),
                  dcc.Upload(id="i-in", children=html.Div(["Arrastre o ", html.A("elija un PNG")]),
                             className="upload", multiple=False, accept="image/png"),
                  html.Div(id="i-out")], className="viz-card"),
        html.P("Ejercicio académico. El clasificador no debe usarse para decidir sobre candidatos reales.",
               className="note"),
    ])


def page_concl():
    return html.Div([
        head("Cierre", "Decisión y conclusiones", "Análisis multicriterio con sensibilidad a los pesos, veredicto de "
             "las hipótesis y revisión crítica."),
        card("Hipótesis pre-registradas", table(hyp.reset_index().rename(columns={"index": "hipótesis"}))),
        card("Decisión multicriterio · imágenes", table(csv("decision_seg").reset_index())),
        card("Decisión multicriterio · texto", table(csv("decision_nlp").reset_index())),
        card("Revisión por pares (cinco objeciones)", table(csv("revision_pares"), page_size=5)),
        html.P(["Todo el detalle, el código y la reproducción: ", html.A("Jupyter Book", href=BOOK, target="_blank"),
                " · ", html.A("repositorio", href=REPO, target="_blank"), "."], className="note"),
    ])


PAGES = {"/": ("Inicio", page_inicio), "/datos": ("Datos", page_datos), "/segmentacion": ("Segmentación", page_seg),
         "/ablaciones": ("Ablaciones y robustez", page_abl), "/texto": ("Texto", page_nlp),
         "/equidad": ("Equidad", page_equidad), "/mlops": ("MLOps y deriva", page_mlops),
         "/api": ("Probar", page_api), "/conclusiones": ("Conclusiones", page_concl)}

app = dash.Dash(__name__, title="Células y hojas de vida · Deep Learning", suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP,
                                      "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"],
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server


@server.route("/figs/<path:name>")
def figs(name):
    if not name.endswith(".png") or not (FIGS / name).exists():
        abort(404)
    return send_from_directory(FIGS, name, max_age=86400)


@server.route("/healthz")
def healthz():
    return "ok"


banner = [html.Div("Vista de prueba con datos SINTÉTICOS: estas cifras no son resultados del proyecto.",
                   className="smoke-banner")] if SMOKE else []
app.layout = html.Div([
    dcc.Location(id="url"),
    html.Header(html.Div([html.A([html.Span(className="brand-dot"), "Células · Texto"], href="/", className="brand"),
                          html.Nav(id="nav", className="topnav"),
                          html.A("Jupyter Book ↗", href=BOOK, target="_blank", className="nav-book")],
                         className="topbar-inner"), className="topbar"),
    *banner,
    html.Main(id="page", className="page"),
    html.Footer(html.Div([
        html.Span("Andrés Duarte Chaves · Tarea 3 (4.8) de Deep Learning · Maestría en Ingeniería Biomédica, "
                  "Universidad del Norte · 2026"),
        html.Span(["Datos: Kaggle (Sartorius; Jarvis Calling, CC0). ", html.A("Código", href=REPO, target="_blank")]),
    ], className="footer-inner"), className="footer"),
])


@app.callback(Output("page", "children"), Output("nav", "children"), Input("url", "pathname"))
def render(path):
    path = path if path in PAGES else "/"
    nav = [html.A(n, href=p, className="topnav-link" + (" active" if p == path else "")) for p, (n, _) in PAGES.items()]
    return PAGES[path][1](), nav


@app.callback(Output("t-out", "children"), Input("t-go", "n_clicks"), State("t-in", "value"),
              prevent_initial_call=True)
def live_text(_, text):
    if not API_URL:
        return html.P("La API no está configurada para este tablero (falta API_URL).", className="note")
    import requests
    try:
        r = requests.post(f"{API_URL}/predict_text", json={"text": text}, timeout=90).json()
    except Exception as e:  # noqa: BLE001
        return html.P(f"No se pudo consultar la API: {e}", className="note")
    p = pd.Series(r["probabilities"]).sort_values().tail(6)
    fig = go.Figure(go.Bar(x=p.values, y=p.index, orientation="h", marker_color=S[1]))
    base_layout(fig, 260)
    fig.update_xaxes(title="probabilidad", range=[0, 1])
    return html.Div([html.P([html.B(r["label"]), f" · {fmt(r['latency_ms'], 1)} ms · modelo {r['model_version']}"]),
                     graph(fig)])


@app.callback(Output("i-out", "children"), Input("i-in", "contents"), prevent_initial_call=True)
def live_image(contents):
    if not API_URL:
        return html.P("La API no está configurada para este tablero (falta API_URL).", className="note")
    import io

    import requests
    from PIL import Image
    b64 = contents.split(",", 1)[1]
    try:
        r = requests.post(f"{API_URL}/predict_image", json={"image_png_base64": b64}, timeout=180).json()
    except Exception as e:  # noqa: BLE001
        return html.P(f"No se pudo consultar la API: {e}", className="note")
    if "rles" not in r:
        return html.P(str(r), className="note")
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("L"))
    h, w = r["height"], r["width"]
    lbl = np.zeros(h * w, np.int32)
    for k, rle in enumerate(r["rles"], 1):
        s = np.asarray(rle.split(), dtype=np.int64)
        for a, ln in zip(s[0::2] - 1, s[1::2]):
            lbl[a:a + ln] = k
    lbl = lbl.reshape(h, w)
    edge = (lbl != np.roll(lbl, 1, 0)) | (lbl != np.roll(lbl, 1, 1))
    rgb = np.stack([img] * 3, -1).astype(np.uint8)
    rgb[edge & (lbl > 0)] = [235, 104, 52]
    fig = go.Figure(go.Image(z=rgb))
    base_layout(fig, 460)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return html.Div([html.P([html.B(f"{r['n_instances']} células"), f" · {fmt(r['latency_ms'], 0)} ms · modelo "
                                                                    f"{r['model_version']}"]), graph(fig)])


if __name__ == "__main__":
    app.run(debug=False, port=8050)
