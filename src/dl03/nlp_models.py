"""Modelos de clasificación de texto con una interfaz común.

Cada modelo se construye con ``fit_model(name, Xtr, ytr, Xva, yva, params, seed, n_classes)`` y expone
``predict_proba(textos) -> (n, K)``. El preprocesado (familia mínimo/estándar/agresivo) va en ``params['prep']``
y TODO lo que se ajusta (vocabulario, IDF, Word2Vec, tokenizer) se ajusta SOLO con los textos de entrenamiento.

Espacios de búsqueda: el del BiLSTM es exactamente la rejilla de 576 configuraciones del enunciado (4.8.44.1);
los demás modelos ligeros reciben el MISMO número de trials de Optuna.
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .nlp_data import CLEANERS


def _prep(texts, prep: str) -> list[str]:
    f = CLEANERS[prep]
    return [f(t) for t in texts]


# ============================================================== sklearn ========
@dataclass
class SkModel:
    """Vectorizador + clasificador. ``kind``: 'sk' (predict_proba con clases presentes), 'nbsvm' o 'xgb'."""
    vec: Any
    clf: Any
    prep: str
    K: int
    kind: str = "sk"
    extra: dict = field(default_factory=dict)

    def predict_proba(self, texts) -> np.ndarray:
        X = self.vec.transform(_prep(texts, self.prep))
        if self.kind == "nbsvm":
            return _nbsvm_proba(self.clf, X, self.extra["r"])
        if self.kind == "xgb":
            return self.clf.predict_proba(X)
        return _full_proba(self.clf, X, self.K)

    def logits(self, texts) -> np.ndarray:
        return np.log(np.clip(self.predict_proba(texts), 1e-9, 1))


def _tfidf(params):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(max_features=params.get("max_features", 50_000), sublinear_tf=True,
                           ngram_range=(1, params.get("ngram", 2)), min_df=params.get("min_df", 2))


def fit_majority(Xtr, ytr, Xva, yva, params, seed, K):
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(max_features=10).fit(_prep(Xtr, "minimal"))
    clf = DummyClassifier(strategy="prior").fit(vec.transform(_prep(Xtr, "minimal")), ytr)
    return SkModel(vec, clf, "minimal", K)


def _full_proba(clf, X, K):
    p = clf.predict_proba(X)
    out = np.zeros((p.shape[0], K))
    out[:, clf.classes_] = p
    return out


def fit_tfidf_lr(Xtr, ytr, Xva, yva, params, seed, K):
    from sklearn.linear_model import LogisticRegression
    prep = params.get("prep", "standard")
    vec = _tfidf(params).fit(_prep(Xtr, prep))
    clf = LogisticRegression(C=params.get("C", 10.0), max_iter=3000, random_state=seed)
    clf.fit(vec.transform(_prep(Xtr, prep)), ytr)
    return SkModel(vec, clf, prep, K)


def _nbsvm_proba(clfs, X, r):
    from .calibration import softmax
    scores = np.column_stack([c.decision_function(X.multiply(rk).tocsr()) for c, rk in zip(clfs, r)])
    return softmax(scores)


def fit_nbsvm(Xtr, ytr, Xva, yva, params, seed, K):
    """NB-SVM (Wang & Manning, 2012) uno-contra-resto con regresión logística sobre rasgos escalados por la razón
    de log-conteos; las puntuaciones se convierten en probabilidades con softmax (no calibradas)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    prep = params.get("prep", "standard")
    vec = TfidfVectorizer(max_features=params.get("max_features", 50_000), ngram_range=(1, params.get("ngram", 2)),
                          min_df=2, use_idf=False, binary=True, norm=None).fit(_prep(Xtr, prep))
    X = vec.transform(_prep(Xtr, prep)).tocsr()
    ytr = np.asarray(ytr)
    clfs, rs = [], []
    for k in range(K):
        pos = X[ytr == k].sum(0) + 1
        neg = X[ytr != k].sum(0) + 1
        r = np.asarray(np.log((pos / pos.sum()) / (neg / neg.sum()))).ravel()
        c = LogisticRegression(C=params.get("C", 4.0), max_iter=2000, random_state=seed)
        yk = (ytr == k).astype(int)
        if yk.sum() == 0:
            yk[0] = 1
        c.fit(X.multiply(r).tocsr(), yk)
        clfs.append(c)
        rs.append(r)
    return SkModel(vec, clfs, prep, K, "nbsvm", {"r": rs})


def fit_xgb(Xtr, ytr, Xva, yva, params, seed, K):
    import xgboost as xgb
    prep = params.get("prep", "standard")
    vec = _tfidf(params).fit(_prep(Xtr, prep))
    dev = "cuda" if _cuda() else "cpu"
    clf = xgb.XGBClassifier(n_estimators=params.get("n_estimators", 600), max_depth=params.get("max_depth", 6),
                            learning_rate=params.get("lr", 0.1), subsample=params.get("subsample", 0.8),
                            colsample_bytree=params.get("colsample", 0.5),
                            min_child_weight=params.get("min_child_weight", 1), reg_lambda=params.get("lambda", 1.0),
                            tree_method="hist", device=dev, random_state=seed, early_stopping_rounds=30,
                            eval_metric="mlogloss", num_class=K, objective="multi:softprob")
    clf.fit(vec.transform(_prep(Xtr, prep)), ytr, eval_set=[(vec.transform(_prep(Xva, prep)), yva)], verbose=False)
    return SkModel(vec, clf, prep, K, "xgb")


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ============================================================== fastText =======
@dataclass
class FTModel:
    model: Any
    prep: str
    K: int

    def predict_proba(self, texts) -> np.ndarray:
        lines = [t.replace("\n", " ") for t in _prep(texts, self.prep)]
        # llamada de bajo nivel: FastText.predict usa np.array(copy=False), que falla con numpy >= 2
        labs, probs = self.model.f.multilinePredict(lines, self.K, 0.0, "strict")
        out = np.zeros((len(texts), self.K))
        for i, (ls, ps) in enumerate(zip(labs, probs)):
            for lab, p in zip(ls, ps):
                out[i, int(lab.replace("__label__", ""))] = p
        return out / np.clip(out.sum(1, keepdims=True), 1e-12, None)


def fit_fasttext(Xtr, ytr, Xva, yva, params, seed, K):
    import fasttext
    prep = params.get("prep", "standard")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for t, y in zip(_prep(Xtr, prep), ytr):
            fh.write(f"__label__{int(y)} {t}\n")
        path = fh.name
    try:
        m = fasttext.train_supervised(path, lr=params.get("lr", 0.5), epoch=params.get("epoch", 25),
                                      wordNgrams=params.get("wordNgrams", 2), dim=params.get("dim", 100),
                                      minn=params.get("minn", 0), maxn=params.get("maxn", 0), loss="softmax",
                                      thread=1, seed=seed, verbose=0)
    finally:
        os.unlink(path)
    return FTModel(m, prep, K)


# ================================================================ Keras ========
def train_word2vec(token_lists, dim: int, sg: int, seed: int, window: int = 5):
    from gensim.models import Word2Vec
    return Word2Vec(token_lists, vector_size=dim, window=window, min_count=2, sg=sg, workers=1, seed=seed, epochs=10)


@dataclass
class KerasTextModel:
    model: Any
    vectorizer: Any
    prep: str

    def predict_proba(self, texts) -> np.ndarray:
        x = self.vectorizer(np.array(_prep(texts, self.prep), dtype=object)).numpy()
        return self.model.predict(x, batch_size=128, verbose=0).astype(np.float64)


def _vectorizer(texts, seq_len: int, max_tokens: int = 30_000):
    import keras
    v = keras.layers.TextVectorization(max_tokens=max_tokens, output_sequence_length=seq_len,
                                       standardize=None, split="whitespace")
    v.adapt(np.array(texts, dtype=object))
    return v


def _embedding_matrix(vocab, w2v, dim):
    E = np.random.default_rng(0).normal(0, 0.05, (len(vocab), dim)).astype(np.float32)
    E[0] = 0
    hit = 0
    for i, w in enumerate(vocab):
        if w in w2v.wv:
            E[i] = w2v.wv[w]
            hit += 1
    return E, hit / max(1, len(vocab))


def build_bilstm(vocab_size, emb_dim, E, units, n_layers, dropout, K, pooling="max", trainable=True):
    import keras
    from keras import layers
    inp = keras.Input((None,), dtype="int64")
    x = layers.Embedding(vocab_size, emb_dim, mask_zero=True, trainable=trainable,
                         embeddings_initializer=keras.initializers.Constant(E) if E is not None else "uniform")(inp)
    for k in range(n_layers):
        last = k == n_layers - 1
        x = layers.Bidirectional(layers.LSTM(units, return_sequences=not (last and pooling == "last"),
                                             dropout=dropout))(x)
    if pooling == "max":
        x = layers.GlobalMaxPooling1D()(x)
    elif pooling == "attention":
        x = AttentionPool()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(K, activation="softmax", dtype="float32")(x)
    return keras.Model(inp, out)


def _attention_pool_cls():
    import keras
    from keras import layers, ops

    @keras.saving.register_keras_serializable(package="dl03")
    class _AttentionPool(layers.Layer):
        """Pooling aditivo con máscara de padding: softmax(vᵀ tanh(W h)) sobre posiciones válidas."""

        def build(self, s):
            self.W = self.add_weight(shape=(s[-1], s[-1]), initializer="glorot_uniform", name="W")
            self.v = self.add_weight(shape=(s[-1], 1), initializer="glorot_uniform", name="v")

        def call(self, h, mask=None):
            e = ops.squeeze(ops.matmul(ops.tanh(ops.matmul(h, self.W)), self.v), -1)
            if mask is not None:
                e = ops.where(mask, e, ops.full_like(e, -1e9))
            a = ops.softmax(e, axis=-1)
            return ops.sum(h * ops.expand_dims(a, -1), axis=1)

        def compute_mask(self, inputs, mask=None):
            return None
    return _AttentionPool


class _Lazy:
    cls = None


def AttentionPool():
    if _Lazy.cls is None:
        _Lazy.cls = _attention_pool_cls()
    return _Lazy.cls()


def build_cnn1d(vocab_size, emb_dim, E, filters, dropout, K, kernels=(2, 3, 4, 5)):
    import keras
    from keras import layers
    inp = keras.Input((None,), dtype="int64")
    x = layers.Embedding(vocab_size, emb_dim,
                         embeddings_initializer=keras.initializers.Constant(E) if E is not None else "uniform")(inp)
    pooled = [layers.GlobalMaxPooling1D()(layers.Conv1D(filters, k, activation="relu")(x)) for k in kernels]
    x = layers.Dropout(dropout)(layers.Concatenate()(pooled))
    return keras.Model(inp, layers.Dense(K, activation="softmax", dtype="float32")(x))


def fit_keras_text(Xtr, ytr, Xva, yva, params, seed, K, kind="bilstm", trial=None):
    import keras

    from .config import set_global_seed
    set_global_seed(seed)
    prep = params.get("prep", "standard")
    tr, va = _prep(Xtr, prep), _prep(Xva, prep)
    seq_len = params.get("sequence_length", 100)
    vec = _vectorizer(tr, seq_len)
    vocab = vec.get_vocabulary()
    dim = params.get("embedding_dim", 100)
    E, cov = None, 0.0
    if params.get("w2v", "skipgram") != "none":
        w2v = train_word2vec([t.split() for t in tr], dim, sg=int(params.get("w2v", "skipgram") == "skipgram"),
                             seed=seed)
        E, cov = _embedding_matrix(vocab, w2v, dim)
    if kind == "bilstm":
        model = build_bilstm(len(vocab), dim, E, params.get("lstm_units", 64), params.get("num_lstm_layers", 1),
                             params.get("dropout_rate", 0.3), K, params.get("pooling", "max"),
                             trainable=params.get("emb_trainable", True))
    else:
        model = build_cnn1d(len(vocab), dim, E, params.get("filters", 128), params.get("dropout_rate", 0.5), K)
    model.compile(keras.optimizers.Adam(params.get("learning_rate", 1e-3)), "sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    xtr = vec(np.array(tr, dtype=object)).numpy()
    xva = vec(np.array(va, dtype=object)).numpy()
    cbs = [keras.callbacks.EarlyStopping("val_loss", patience=params.get("patience", 4), restore_best_weights=True)]
    if trial is not None:
        cbs.append(_OptunaPrune(trial))
    t0 = time.time()
    h = model.fit(xtr, np.asarray(ytr), validation_data=(xva, np.asarray(yva)), epochs=params.get("max_epochs", 20),
                  batch_size=params.get("batch_size", 32), verbose=0, callbacks=cbs)
    m = KerasTextModel(model, vec, prep)
    m.history = h.history
    m.train_seconds = time.time() - t0
    m.w2v_coverage = cov
    return m


def _optuna_prune_cls():
    import keras

    class _P(keras.callbacks.Callback):
        def __init__(self, trial):
            super().__init__()
            self.trial = trial

        def on_epoch_end(self, epoch, logs=None):
            import optuna
            self.trial.report(-float(logs["val_loss"]), epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
    return _P


def _OptunaPrune(trial):
    return _optuna_prune_cls()(trial)


# ============================================================ transformer ======
@dataclass
class HFModel:
    model: Any
    tok: Any
    max_length: int
    head: int

    def encode(self, texts):
        return encode_head_tail(self.tok, _prep(texts, "minimal"), self.max_length, self.head)

    def logits(self, texts, batch: int = 32) -> np.ndarray:
        import torch
        dev = next(self.model.parameters()).device
        self.model.eval()
        out = []
        enc = self.encode(texts)
        with torch.no_grad():
            for s in range(0, len(texts), batch):
                ids = torch.as_tensor(enc["input_ids"][s:s + batch], device=dev)
                am = torch.as_tensor(enc["attention_mask"][s:s + batch], device=dev)
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    out.append(self.model(input_ids=ids, attention_mask=am).logits.float().cpu().numpy())
        return np.concatenate(out)

    def predict_proba(self, texts) -> np.ndarray:
        from .calibration import softmax
        return softmax(self.logits(texts))


def encode_head_tail(tok, texts, max_length: int, head: int):
    """Truncamiento cabeza+cola (Sun et al., 2019): primeros ``head`` y últimos tokens restantes."""
    ids, masks = [], []
    body = max_length - 2
    for t in texts:
        x = tok(t, add_special_tokens=False, truncation=False)["input_ids"]
        if len(x) > body:
            x = x[:head] + x[-(body - head):] if head < body else x[:body]
        x = [tok.cls_token_id] + x + [tok.sep_token_id]
        m = [1] * len(x) + [0] * (max_length - len(x))
        ids.append(x + [tok.pad_token_id] * (max_length - len(x)))
        masks.append(m)
    return {"input_ids": np.array(ids), "attention_mask": np.array(masks)}


def tiny_tokenizer(texts):
    """Tokenizer WordPiece entrenado en el train (solo perfil smoke: sin descargar modelos)."""
    from tokenizers import BertWordPieceTokenizer
    from transformers import PreTrainedTokenizerFast
    bt = BertWordPieceTokenizer(lowercase=True)
    bt.train_from_iterator(texts, vocab_size=2000, min_frequency=1)
    return PreTrainedTokenizerFast(tokenizer_object=bt._tokenizer, cls_token="[CLS]", sep_token="[SEP]",
                                   pad_token="[PAD]", unk_token="[UNK]", mask_token="[MASK]")


def fit_transformer(Xtr, ytr, Xva, yva, params, seed, K, log: Callable | None = None):
    """Fine-tuning con warmup + decaimiento lineal, LR decreciente por capa (LLRD), precisión mixta y early stopping
    por F1 macro de validación."""
    import torch
    from sklearn.metrics import f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DistilBertConfig,
        DistilBertForSequenceClassification,
    )

    from .config import set_global_seed
    set_global_seed(seed, deterministic=False)
    name, L = params["name"], params["max_length"]
    if name == "tiny-random":
        tok = tiny_tokenizer(_prep(Xtr, "minimal"))
        cfg = DistilBertConfig(vocab_size=tok.vocab_size, dim=32, hidden_dim=64, n_layers=1, n_heads=2,
                               max_position_embeddings=max(L, 64), num_labels=K)
        model = DistilBertForSequenceClassification(cfg)
    else:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=K)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    head = params.get("head", 128)
    tr = encode_head_tail(tok, _prep(Xtr, "minimal"), L, head)
    lr, decay = params["lr"], params.get("llrd", 0.9)
    groups = []
    n_layers = max([int(n.split("layer.")[1].split(".")[0]) for n, _ in model.named_parameters() if "layer." in n] or
                   [0]) + 1
    for n, p in model.named_parameters():
        if "layer." in n:
            depth = int(n.split("layer.")[1].split(".")[0])
            groups.append({"params": [p], "lr": lr * decay ** (n_layers - 1 - depth)})
        elif "embeddings" in n:
            groups.append({"params": [p], "lr": lr * decay ** n_layers})
        else:
            groups.append({"params": [p], "lr": lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    bs, epochs = params["batch"], params["epochs"]
    steps = epochs * int(np.ceil(len(Xtr) / bs))
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              max(0.0, (steps - s) / max(1, steps - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    y = torch.as_tensor(np.asarray(ytr))
    rng = np.random.default_rng(seed)
    hf = HFModel(model, tok, L, head)
    best, best_state, hist = -1.0, None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(Xtr))
        tot = 0.0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            ids = torch.as_tensor(tr["input_ids"][b], device=dev)
            am = torch.as_tensor(tr["attention_mask"][b], device=dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = model(input_ids=ids, attention_mask=am, labels=y[b].to(dev)).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach())
        f1 = f1_score(yva, hf.predict_proba(Xva).argmax(1), average="macro")
        hist.append({"epoch": ep, "loss": tot / max(1, len(perm) / bs), "val_f1": f1})
        if log:
            log(hist[-1])
        if f1 > best:
            best = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    hf.history = hist
    hf.train_seconds = time.time() - t0
    return hf


# ============================================================ registro ==========
FITTERS: dict[str, Callable] = {
    "majority": fit_majority, "tfidf_lr": fit_tfidf_lr, "nbsvm": fit_nbsvm, "tfidf_xgb": fit_xgb,
    "fasttext": fit_fasttext,
    "bilstm": lambda *a, **k: fit_keras_text(*a, kind="bilstm", **k),
    "cnn1d": lambda *a, **k: fit_keras_text(*a, kind="cnn1d", **k),
}


def fit_model(name, Xtr, ytr, Xva, yva, params, seed, K, **kw):
    return FITTERS[name](Xtr, ytr, Xva, yva, params, seed, K, **kw)


def space(name: str, trial, cap_epochs: int = 50) -> dict:
    """Espacios de búsqueda (mismo número de trials para todos los modelos ligeros)."""
    s = trial.suggest_categorical
    if name == "bilstm":   # rejilla obligatoria del enunciado (576 configuraciones) + pooling a comparar
        return dict(lstm_units=s("lstm_units", [32, 64, 128]), num_lstm_layers=s("num_lstm_layers", [1, 2]),
                    dropout_rate=s("dropout_rate", [0.3, 0.5]),
                    learning_rate=s("learning_rate", [0.0005, 0.001, 0.003]), batch_size=s("batch_size", [32, 64]),
                    sequence_length=s("sequence_length", [50, 100]),
                    max_epochs=min(cap_epochs, s("max_epochs", [20, 50])), embedding_dim=s("embedding_dim", [100, 200]),
                    pooling=s("pooling", ["last", "max", "attention"]))
    if name == "cnn1d":
        return dict(filters=s("filters", [64, 128]), dropout_rate=s("dropout_rate", [0.3, 0.5]),
                    learning_rate=s("learning_rate", [0.0005, 0.001, 0.003]), batch_size=s("batch_size", [32, 64]),
                    sequence_length=s("sequence_length", [100, 300, 500]), max_epochs=min(cap_epochs, 30),
                    embedding_dim=s("embedding_dim", [100, 200]))
    if name == "tfidf_xgb":
        return dict(max_depth=trial.suggest_int("max_depth", 3, 8), lr=trial.suggest_float("lr", 0.03, 0.3, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample=trial.suggest_float("colsample", 0.2, 0.8),
                    min_child_weight=trial.suggest_int("min_child_weight", 1, 6),
                    max_features=s("max_features", [20_000, 50_000]), ngram=s("ngram", [1, 2]))
    if name == "fasttext":
        return dict(lr=trial.suggest_float("lr", 0.05, 1.0, log=True), epoch=trial.suggest_int("epoch", 5, 50),
                    wordNgrams=s("wordNgrams", [1, 2, 3]), dim=s("dim", [50, 100, 200]), minn=s("minn", [0, 3]),
                    maxn=s("maxn", [0, 6]))
    if name == "tfidf_lr":
        return dict(C=trial.suggest_float("C", 0.1, 100, log=True), ngram=s("ngram", [1, 2]),
                    max_features=s("max_features", [20_000, 50_000]))
    if name == "nbsvm":
        return dict(C=trial.suggest_float("C", 0.1, 50, log=True), ngram=s("ngram", [1, 2]))
    return {}
