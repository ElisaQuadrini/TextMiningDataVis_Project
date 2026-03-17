# -*- coding: utf-8 -*-
"""
Text Mining Dashboard — SQuAD 2.0
EDA | BiDAF | DeBERTa
"""

import json
import re
import numpy as np
import pandas as pd
from collections import Counter

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from nltk.corpus import stopwords
from nltk.util import ngrams
import nltk
nltk.download("stopwords", quiet=True)

import io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

STOP = set(stopwords.words("english"))
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
WH_WORDS = ["what", "who", "when", "where", "why", "how", "which"]

# ─────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────

def load_squad_df(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        squad = json.load(f)
    rows = []
    for art in squad["data"]:
        title = art.get("title", "")
        for para in art["paragraphs"]:
            context = para["context"]
            for qa in para.get("qas", []):
                is_imp = bool(qa.get("is_impossible", False))
                answers = qa.get("answers", [])
                ans_text = answers[0]["text"] if answers else "NO_ANSWER"
                rows.append({
                    "id": qa.get("id", ""),
                    "title": title,
                    "context": context,
                    "question": qa.get("question", ""),
                    "is_impossible": is_imp,
                    "y": 0 if is_imp else 1,
                    "answer_text": ans_text,
                })
    df = pd.DataFrame(rows)

    def clean(s):
        s = str(s).strip()
        return re.sub(r"\s+", " ", s)
    df["question"] = df["question"].map(clean)
    df["context"]  = df["context"].map(clean)

    df["q_len_tok"] = df["question"].str.split().str.len()
    df["c_len_tok"] = df["context"].str.split().str.len()
    df["q_len_char"] = df["question"].str.len()
    df["c_len_char"] = df["context"].str.len()
    df["a_len_tok"] = df["answer_text"].str.split().str.len()
    df.loc[df["answer_text"] == "NO_ANSWER", "a_len_tok"] = 0

    def wh_cat(q):
        q = str(q).lower()
        for w in WH_WORDS:
            if re.search(rf"\b{w}\b", q):
                return w
        return "other"
    df["wh"] = df["question"].map(wh_cat)

    def overlap(q, c):
        qt = set(str(q).lower().split())
        ct = set(str(c).lower().split())
        return len(qt & ct) / len(qt) if qt else 0.0
    df["overlap"] = [overlap(q, c) for q, c in zip(df["question"], df["context"])]

    return df


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(str(text)) if t]


def top_tokens(texts, k=20):
    c = Counter()
    for t in texts:
        c.update([w for w in tokenize(t) if w not in STOP])
    return c.most_common(k)


def top_ngrams_list(texts, n=2, k=15):
    c = Counter()
    for t in texts:
        toks = [w for w in tokenize(t) if w not in STOP]
        c.update(list(ngrams(toks, n)))
    return c.most_common(k)


# ─────────────────────────────────────────────
# WORDCLOUD GENERATOR
# ─────────────────────────────────────────────

def generate_wordcloud_img(df, top_n=4):
    top_titles = df["title"].value_counts().head(top_n).index.tolist()
    df_sample  = df[df["title"].isin(top_titles)].copy()
    df_sample["text"] = df_sample["question"].astype(str) + " " + df_sample["context"].astype(str)

    COLORMAP_CYCLE = ["viridis", "plasma", "cividis", "magma"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flat

    for ax, title, cmap in zip(axes, top_titles, COLORMAP_CYCLE):
        texts = df_sample.loc[df_sample["title"] == title, "text"].tolist()
        words = []
        for t in texts:
            words += [w.lower() for w in TOKEN_RE.findall(t) if w.lower() not in STOP and len(w) > 2]
        if words:
            wc = WordCloud(width=700, height=420, background_color="white",
                           colormap=cmap, max_words=120).generate(" ".join(words))
            ax.imshow(wc, interpolation="bilinear")
            ax.set_title(title.replace("_", " "), fontsize=14, fontweight="bold", pad=10)
        ax.axis("off")

    plt.suptitle("Word Clouds by Topic (lemmas, stopwords removed)", fontsize=16, y=1.01)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
try:
    train_df = load_squad_df("train_sampled.json")
    DATA_LOADED = True
except Exception as e:
    train_df = pd.DataFrame()
    DATA_LOADED = False
    DATA_ERROR = str(e)


# ─────────────────────────────────────────────
# DEBERTA MODEL — lazy load on first inference
# ─────────────────────────────────────────────
import torch
import string
import collections

DEBERTA_PATH = "v2/deberta_squad_finetuned"
_deberta_model     = None
_deberta_tokenizer = None
_deberta_device    = None

def get_deberta():
    global _deberta_model, _deberta_tokenizer, _deberta_device
    if _deberta_model is None:
        from transformers import AutoTokenizer, DebertaV2ForQuestionAnswering
        _deberta_device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _deberta_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_PATH)
        _deberta_model     = DebertaV2ForQuestionAnswering.from_pretrained(
            DEBERTA_PATH, torch_dtype=torch.float32
        ).to(_deberta_device)
        _deberta_model.eval()
    return _deberta_model, _deberta_tokenizer, _deberta_device


def answer_question_deberta(question, context):
    model, tokenizer, device = get_deberta()
    inputs = tokenizer(
        question, context,
        max_length=384, truncation="only_second", stride=128,
        return_overflowing_tokens=True, return_offsets_mapping=True,
        padding="max_length", return_tensors="pt",
    )
    offset_mapping = inputs.pop("offset_mapping")
    inputs.pop("overflow_to_sample_mapping")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    best_score  = -float("inf")
    best_answer = ""
    for i in range(len(outputs.start_logits)):
        sl = outputs.start_logits[i]
        el = outputs.end_logits[i]
        null_score  = (sl[0] + el[0]).item()
        best_start  = int(sl[1:].argmax()) + 1
        best_end    = int(el[1:].argmax()) + 1
        span_score  = (sl[best_start] + el[best_end]).item()
        if span_score > best_score:
            best_score = span_score
            if null_score > span_score:
                best_answer = None
            else:
                if best_end < best_start:
                    best_end = best_start
                s = offset_mapping[i][best_start][0].item()
                e = offset_mapping[i][best_end][1].item()
                best_answer = context[s:e]
    return best_answer


# ─────────────────────────────────────────────
# BIDAF MODEL — lazy load on first inference
# ─────────────────────────────────────────────
BIDAF_PATH = "BiDAF_finale"   # path to the BiDAF checkpoint (e.g. ~/Desktop/BiDAF_finale)
GLOVE_PATH = "glove.6B.300d.txt"

_bidaf_model     = None
_bidaf_word2idx  = None
_bidaf_char2idx  = None
_bidaf_device    = None

# BiDAF constants (must match training config)
_EMBED_DIM      = 300
_CHAR_EMBED_DIM = 64
_CHAR_OUT_DIM   = 100
_CHAR_KERNEL    = 5
_MAX_WORD_LEN   = 16
_MAX_CONTEXT    = 300
_MAX_QUESTION   = 40
_HIDDEN_DIM     = 100
_DROPOUT        = 0.15
_NUM_HIGHWAY    = 2
_BEST_THRESHOLD = 0.80
_PAD_TOKEN      = "<PAD>"
_UNK_TOKEN      = "<UNK>"
_BIDAF_TOKEN_RE = re.compile(r"([.,!?;:\"'()\[\]{}])")


def _bidaf_tokenize(text):
    text = str(text).lower()
    text = _BIDAF_TOKEN_RE.sub(r" \1 ", text)
    return text.split()


def _encode_sequence(tokens, word2idx, max_len):
    unk = word2idx.get(_UNK_TOKEN, 1)
    pad = word2idx.get(_PAD_TOKEN, 0)
    ids = [word2idx.get(t, unk) for t in tokens[:max_len]]
    pad_len = max_len - len(ids)
    mask = [1] * len(ids) + [0] * pad_len
    ids  = ids + [pad] * pad_len
    return ids, mask


def _tokens_to_char_ids(tokens, char2idx, max_seq_len, max_word_len=_MAX_WORD_LEN):
    unk_c = char2idx.get("<UNK_C>", 1)
    pad_c = char2idx.get("<PAD_C>", 0)
    result = []
    for tok in tokens[:max_seq_len]:
        ids = [char2idx.get(c, unk_c) for c in tok[:max_word_len]]
        ids = ids + [pad_c] * (max_word_len - len(ids))
        result.append(ids)
    while len(result) < max_seq_len:
        result.append([pad_c] * max_word_len)
    return result


def _exact_match_feature(ctx_tokens, q_tokens, max_c_len):
    q_set = set(t.lower() for t in q_tokens)
    feats = [1.0 if t.lower() in q_set else 0.0 for t in ctx_tokens[:max_c_len]]
    feats += [0.0] * (max_c_len - len(feats))
    return feats


def _best_span(start_logits, end_logits, max_answer_len=15):
    T = start_logits.size(0)
    scores = start_logits.unsqueeze(1) + end_logits.unsqueeze(0)
    mask = torch.tril(
        torch.ones(T, T, dtype=torch.bool), diagonal=max_answer_len - 1
    ) & torch.triu(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~mask, -1e18)
    best = scores.argmax()
    return (best // T).item(), (best % T).item()


def get_bidaf():
    global _bidaf_model, _bidaf_word2idx, _bidaf_char2idx, _bidaf_device
    if _bidaf_model is not None:
        return _bidaf_model, _bidaf_word2idx, _bidaf_char2idx, _bidaf_device

    import torch.nn as nn
    import torch.nn.functional as F

    class CharCNNEncoder(nn.Module):
        def __init__(self, char_vocab_size, char_embed_dim, char_out_dim, kernel_size=5):
            super().__init__()
            self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
            self.conv = nn.Conv1d(char_embed_dim, char_out_dim, kernel_size, padding=kernel_size // 2)

        def forward(self, char_ids):
            B, L, W = char_ids.size()
            x = self.char_embedding(char_ids.view(B * L, W))
            x = x.permute(0, 2, 1)
            x = torch.relu(self.conv(x))
            x = x.max(dim=2).values
            return x.view(B, L, -1)

    class BiDAFQA(nn.Module):
        def __init__(self, vocab_size, char_vocab_size, embed_dim=300, char_embed_dim=64,
                     char_out_dim=100, char_kernel=5, hidden_dim=100, embeddings=None,
                     dropout=0.15, num_highway=2):
            super().__init__()
            H       = hidden_dim * 2
            inp_dim = embed_dim + char_out_dim + 1

            self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            if embeddings is not None:
                self.word_embedding.weight = nn.Parameter(torch.tensor(embeddings))

            self.char_encoder = CharCNNEncoder(char_vocab_size, char_embed_dim, char_out_dim, char_kernel)
            self.dropout      = nn.Dropout(dropout)
            self.highway_proj = nn.Linear(inp_dim, embed_dim)
            self.highway_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim * 2) for _ in range(num_highway)])

            lstm_kwargs = dict(input_size=embed_dim, hidden_size=hidden_dim,
                               num_layers=1, batch_first=True, bidirectional=True, dropout=0.0)
            self.ctx_lstm = nn.LSTM(**lstm_kwargs)
            self.q_lstm   = nn.LSTM(**lstm_kwargs)
            self.attn_w   = nn.Linear(H * 3, 1, bias=False)
            self.fusion_proj = nn.Linear(H * 4, H)

            self.model_lstm = nn.LSTM(input_size=H, hidden_size=hidden_dim,
                                      num_layers=2, batch_first=True,
                                      bidirectional=True, dropout=dropout)

            self.sa_q = nn.Linear(H, H, bias=False)
            self.sa_k = nn.Linear(H, H, bias=False)
            self.sa_v = nn.Linear(H, H, bias=False)
            self.sa_o = nn.Linear(H, H, bias=False)
            self.sa_norm = nn.LayerNorm(H)

            self.span_start = nn.Linear(H * 2, 1)
            self.span_end   = nn.Linear(H * 2, 1)
            self.ans_head   = nn.Sequential(
                nn.Linear(H, H // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(H // 2, 1)
            )

        def _highway(self, x):
            x = self.dropout(torch.relu(self.highway_proj(x)))
            for layer in self.highway_layers:
                out   = layer(x)
                h, g  = out.chunk(2, dim=-1)
                h     = torch.relu(h)
                g     = torch.sigmoid(g)
                x     = g * h + (1 - g) * x
            return x

        def _encode(self, word_ids, masks, char_ids, em=None, lstm=None):
            w  = self.word_embedding(word_ids)
            ch = self.char_encoder(char_ids)
            if em is not None:
                x = torch.cat([w, ch, em.unsqueeze(-1)], dim=-1)
            else:
                x = torch.cat([w, ch, torch.zeros(w.size(0), w.size(1), 1, device=w.device)], dim=-1)
            x  = self._highway(x)
            x  = self.dropout(x)
            out, _ = lstm(x)
            return self.dropout(out)

        def forward(self, ctx_ids, ctx_masks, ctx_chars, ctx_em, q_ids, q_masks, q_chars):
            ctx_enc = self._encode(ctx_ids, ctx_masks, ctx_chars, ctx_em, self.ctx_lstm)
            q_enc   = self._encode(q_ids,   q_masks,   q_chars,   None,   self.q_lstm)

            B, Lc, H = ctx_enc.size()
            Lq = q_enc.size(1)

            ctx_exp = ctx_enc.unsqueeze(2).expand(-1, -1, Lq, -1)
            q_exp   = q_enc.unsqueeze(1).expand(-1, Lc, -1, -1)
            sim     = self.attn_w(torch.cat([ctx_exp, q_exp, ctx_exp * q_exp], dim=-1)).squeeze(-1)

            q_mask_exp = q_masks.unsqueeze(1).expand(-1, Lc, -1).bool()
            sim_c2q    = sim.masked_fill(~q_mask_exp, -1e18)
            alpha      = torch.softmax(sim_c2q, dim=-1)
            c2q        = (alpha.unsqueeze(-1) * q_enc.unsqueeze(1)).sum(dim=2)

            ctx_mask_exp = ctx_masks.unsqueeze(2).expand(-1, -1, Lq).bool()
            sim_q2c      = sim.masked_fill(~ctx_mask_exp.transpose(1, 2), -1e18)
            beta         = torch.softmax(sim_q2c.max(dim=2).values, dim=-1)
            q2c          = (beta.unsqueeze(-1) * ctx_enc).sum(dim=1, keepdim=True).expand(-1, Lc, -1)

            fused     = torch.cat([ctx_enc, c2q, ctx_enc * c2q, ctx_enc * q2c], dim=-1)
            fused_proj = torch.relu(self.fusion_proj(fused))

            modeled, _ = self.model_lstm(self.dropout(fused_proj))
            modeled    = self.dropout(modeled)

            Q_ = self.sa_q(modeled)
            K_ = self.sa_k(modeled)
            V_ = self.sa_v(modeled)
            sa_scores = torch.bmm(Q_, K_.transpose(1, 2)) / (H ** 0.5)
            ctx_m_exp = ctx_masks.unsqueeze(1).expand(-1, Lc, -1).bool()
            sa_scores = sa_scores.masked_fill(~ctx_m_exp, -1e18)
            sa_out    = torch.bmm(torch.softmax(sa_scores, dim=-1), V_)
            sa_out    = self.sa_norm(self.sa_o(sa_out) + modeled)

            span_repr = torch.cat([fused_proj, sa_out], dim=-1)
            start_logits = self.span_start(span_repr).squeeze(-1)
            end_logits   = self.span_end(span_repr).squeeze(-1)

            ctx_mask_bool = ctx_masks.bool()
            start_logits = start_logits.masked_fill(~ctx_mask_bool, -1e18)
            end_logits   = end_logits.masked_fill(~ctx_mask_bool, -1e18)

            pooled = (sa_out * ctx_masks.unsqueeze(-1).float()).sum(1) / ctx_masks.float().sum(1, keepdim=True).clamp(min=1)
            ans_logits = self.ans_head(pooled).squeeze(-1)

            return {"start_logits": start_logits, "end_logits": end_logits, "answerable_logits": ans_logits}

    _bidaf_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(BIDAF_PATH, map_location=_bidaf_device, weights_only=False)

    _bidaf_word2idx = ckpt["word2idx"]
    _bidaf_char2idx = ckpt["char2idx"]
    vocab_size      = len(_bidaf_word2idx)
    char_vocab_size = len(_bidaf_char2idx)

    _bidaf_model = BiDAFQA(
        vocab_size=vocab_size, char_vocab_size=char_vocab_size,
        embed_dim=_EMBED_DIM, char_embed_dim=_CHAR_EMBED_DIM,
        char_out_dim=_CHAR_OUT_DIM, char_kernel=_CHAR_KERNEL,
        hidden_dim=_HIDDEN_DIM, dropout=_DROPOUT, num_highway=_NUM_HIGHWAY,
    ).to(_bidaf_device)
    _bidaf_model.load_state_dict(ckpt["model_state"])
    _bidaf_model.eval()

    return _bidaf_model, _bidaf_word2idx, _bidaf_char2idx, _bidaf_device


def answer_question_bidaf(question, context):
    model, word2idx, char2idx, device = get_bidaf()

    ctx_tokens = _bidaf_tokenize(context)[:_MAX_CONTEXT]
    q_tokens   = _bidaf_tokenize(question)[:_MAX_QUESTION]

    ctx_ids,  ctx_mask = _encode_sequence(ctx_tokens, word2idx, _MAX_CONTEXT)
    q_ids,    q_mask   = _encode_sequence(q_tokens,   word2idx, _MAX_QUESTION)
    ctx_chars          = _tokens_to_char_ids(ctx_tokens, char2idx, _MAX_CONTEXT)
    q_chars            = _tokens_to_char_ids(q_tokens,   char2idx, _MAX_QUESTION)
    ctx_em             = _exact_match_feature(ctx_tokens, q_tokens, _MAX_CONTEXT)

    def to_t(x, dtype):
        return torch.tensor(x, dtype=dtype).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            to_t(ctx_ids,   torch.long),
            to_t(ctx_mask,  torch.float),
            to_t(ctx_chars, torch.long),
            to_t(ctx_em,    torch.float),
            to_t(q_ids,     torch.long),
            to_t(q_mask,    torch.float),
            to_t(q_chars,   torch.long),
        )

    prob = torch.sigmoid(outputs["answerable_logits"]).item()
    if prob < _BEST_THRESHOLD:
        return None

    s, e = _best_span(outputs["start_logits"][0].cpu(), outputs["end_logits"][0].cpu())
    if s < 0 or e < s or s >= len(ctx_tokens):
        return None
    return " ".join(ctx_tokens[s:min(e + 1, len(ctx_tokens))])


# ─────────────────────────────────────────────
# MODEL METRICS
# ─────────────────────────────────────────────

DEBERTA_METRICS = {
    "precision_unanswerable": 0.91,
    "recall_unanswerable":    0.89,
    "f1_unanswerable":        0.90,
    "precision_answerable":   0.92,
    "recall_answerable":      0.93,
    "f1_answerable":          0.93,
    "accuracy":               0.92,
    "macro_f1":               0.91,
    "exact_match":            0.6767,
    "f1_span":                0.8006,
    "evaluated_examples":     9038,
}

BIDAF_METRICS = {
    "precision_unanswerable": 0.59,
    "recall_unanswerable":    0.92,
    "f1_unanswerable":        0.72,
    "precision_answerable":   0.91,
    "recall_answerable":      0.54,
    "f1_answerable":          0.68,
    "accuracy":               0.70,
    "macro_f1":               0.70,
    "exact_match":            0.4369,
    "f1_span":                0.5101,
    "evaluated_examples":     8990,
    "best_threshold":         0.80,
    "span_f1_answerable":     0.2140,
}


# ─────────────────────────────────────────────
# PLOTLY FIGURE BUILDERS
# ─────────────────────────────────────────────

COLORS = {
    "answerable":   "#2563EB",
    "unanswerable": "#EF4444",
    "neutral":      "#6366F1",
    "bg":           "white",
    "grid":         "#F3F4F6",
    "bidaf":        "#7C3AED",
    "deberta":      "#0891B2",
}

LAYOUT_BASE = dict(
    plot_bgcolor=COLORS["bg"],
    paper_bgcolor=COLORS["bg"],
    font=dict(family="Georgia, serif", size=12),
    margin=dict(l=50, r=20, t=50, b=50),
)


def fig_label_dist(df):
    counts = df["y"].value_counts().sort_index()
    labels = ["Unanswerable", "Answerable"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker_colors=["#F4A261", "#2563EB"],
        hole=0.45,
        textinfo="label+percent",
        textfont=dict(size=13),
    ))
    fig.update_layout(**LAYOUT_BASE, title="Label Distribution (Train)",
                      legend=dict(orientation="v", x=1.02, y=1, xanchor="left"))
    return fig


def fig_wh(df):
    counts = df.groupby(["wh", "y"]).size().unstack(fill_value=0)
    if 0 not in counts.columns: counts[0] = 0
    if 1 not in counts.columns: counts[1] = 0
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    labels = counts.index.tolist()
    fig = go.Figure([
        go.Bar(name="Unanswerable", x=labels, y=counts[0], marker_color="#F4A261"),
        go.Bar(name="Answerable",   x=labels, y=counts[1], marker_color=COLORS["answerable"]),
    ])
    fig.update_layout(**LAYOUT_BASE, barmode="stack",
                      title="Answerability by Question Type (WH words)",
                      xaxis_title="Question word", yaxis_title="Count",
                      yaxis=dict(gridcolor=COLORS["grid"]),
                      legend=dict(orientation="v", x=1.02, y=1, xanchor="left"))
    return fig


def fig_top_tokens(df, k=10):
    common = top_tokens(df["question"], k=k)
    words = [w for w, _ in common][::-1]
    freqs = [f for _, f in common][::-1]
    fig = go.Figure(go.Bar(x=freqs, y=words, orientation="h",
                           marker_color=COLORS["neutral"], marker_line_width=0))
    fig.update_layout(**LAYOUT_BASE, title=f"Top {k} Tokens in Questions (no stopwords)",
                      xaxis_title="Frequency", xaxis=dict(gridcolor=COLORS["grid"]),
                      height=max(300, k * 28))
    return fig


def fig_top_bigrams(df, k=10):
    common = top_ngrams_list(df["question"], n=2, k=k)
    labels = [" ".join(g) for g, _ in common][::-1]
    freqs  = [f for _, f in common][::-1]
    fig = go.Figure(go.Bar(x=freqs, y=labels, orientation="h",
                           marker_color="#0891B2", marker_line_width=0))
    fig.update_layout(**LAYOUT_BASE, title=f"Top {k} Bigrams in Questions (no stopwords)",
                      xaxis_title="Frequency", xaxis=dict(gridcolor=COLORS["grid"]),
                      height=max(300, k * 28))
    return fig


def fig_top_titles(df, top_n=10):
    top_t  = df["title"].value_counts().head(top_n).index
    tmp    = df[df["title"].isin(top_t)]
    counts = tmp.groupby(["title", "y"]).size().unstack(fill_value=0)
    if 0 not in counts.columns: counts[0] = 0
    if 1 not in counts.columns: counts[1] = 0
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    perc   = counts.div(counts.sum(axis=1), axis=0) * 100
    titles = [t.replace("_", " ") for t in perc.index.tolist()]
    fig = go.Figure([
        go.Bar(name="Unanswerable", x=perc[0].values, y=titles,
               orientation="h", marker_color="#F4A261"),
        go.Bar(name="Answerable",   x=perc[1].values, y=titles,
               orientation="h", marker_color=COLORS["answerable"]),
    ])
    fig.update_layout(**LAYOUT_BASE, barmode="stack",
                      title=f"Top {top_n} Topics: % Answerable vs Unanswerable",
                      xaxis_title="Percentage (%)", xaxis=dict(gridcolor=COLORS["grid"]),
                      legend=dict(orientation="v", x=1.02, y=1, xanchor="left"),
                      height=top_n * 42 + 80)
    return fig


def fig_length_hist(df, mode="question"):
    if mode == "question":
        x = df["q_len_tok"]; title = "Question Length (tokens)"; color = COLORS["answerable"]
    elif mode == "context":
        x = df["c_len_tok"]; title = "Context Length (tokens)"; color = COLORS["neutral"]
    else:
        x = df.loc[df["y"] == 1, "a_len_tok"]; title = "Answer Length — answerable only (tokens)"; color = "#0891B2"
    fig = go.Figure(go.Histogram(x=x, nbinsx=60, marker_color=color, opacity=0.85))
    fig.update_layout(**LAYOUT_BASE, title=title, xaxis_title="Tokens", yaxis_title="Frequency",
                      yaxis=dict(gridcolor=COLORS["grid"]))
    return fig


def fig_model_cls_report(m, model_name, accent_color):
    """Grouped bar: Precision / Recall / F1 per class."""
    categories = ["Unanswerable", "Answerable"]
    fig = go.Figure()
    for metric, vals, color in [
        ("Precision", [m["precision_unanswerable"], m["precision_answerable"]], "#6366F1"),
        ("Recall",    [m["recall_unanswerable"],    m["recall_answerable"]],    accent_color),
        ("F1",        [m["f1_unanswerable"],        m["f1_answerable"]],        "#0891B2"),
    ]:
        fig.add_trace(go.Bar(name=metric, x=categories, y=vals, marker_color=color))
    fig.update_layout(
        **LAYOUT_BASE, barmode="group",
        title=dict(text=f"Classification Report — {model_name}", y=0.97),
        yaxis=dict(range=[0, 1], gridcolor=COLORS["grid"], tickformat=".0%"),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )
    return fig


def fig_model_radar(m, model_name, accent_color):
    radar_cats = ["Precision<br>Unanswerable", "Recall<br>Unanswerable", "F1<br>Unanswerable",
                  "Precision<br>Answerable",   "Recall<br>Answerable",   "F1<br>Answerable"]
    radar_vals = [m["precision_unanswerable"], m["recall_unanswerable"], m["f1_unanswerable"],
                  m["precision_answerable"],   m["recall_answerable"],   m["f1_answerable"]]
    fig = go.Figure(go.Scatterpolar(
        r=radar_vals + [radar_vals[0]],
        theta=radar_cats + [radar_cats[0]],
        fill="toself",
        line_color=accent_color,
        fillcolor=f"rgba({int(accent_color[1:3],16)},{int(accent_color[3:5],16)},{int(accent_color[5:7],16)},0.15)",
        name=model_name,
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Per-Class Metrics Radar — {model_name}",
    )
    return fig


def fig_model_span(m, model_name):
    fig = go.Figure([
        go.Bar(
            x=["Exact Match", "F1 Span"],
            y=[m["exact_match"], m["f1_span"]],
            marker_color=["#6366F1", "#0891B2"],
            text=[f"{m['exact_match']:.1%}", f"{m['f1_span']:.1%}"],
            textposition="outside",
            width=0.35,
        )
    ])
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"QA Span Metrics — {model_name}  (n={m['evaluated_examples']:,})",
        yaxis=dict(range=[0, 1], gridcolor=COLORS["grid"], tickformat=".0%"),
    )
    return fig


# ─────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────

app = dash.Dash(
    suppress_callback_exceptions=True,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap",
    ]
)
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>SQuAD 2.0 — Text Mining Dashboard</title>
{%favicon%}
{%css%}
<style>
  body {
    background-color: #F0F2F8;
    font-family: "DM Sans", sans-serif;
  }

  /* ── Header ── */
  #header-bar {
    background: linear-gradient(135deg, #0B1D51 0%, #1A3A8F 60%, #2563EB 100%);
    padding: 0;
    border-bottom: 3px solid #3B82F6;
  }
  .header-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 36px;
  }
  .header-title-block { display: flex; align-items: center; gap: 16px; }
  .header-badge {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 22px;
    line-height: 1;
  }
  .header-title {
    font-family: "DM Serif Display", serif;
    font-size: 24px;
    color: #FFFFFF;
    letter-spacing: 0.3px;
    margin: 0;
  }
  .header-subtitle {
    font-size: 12px;
    color: rgba(255,255,255,0.55);
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin: 2px 0 0 0;
  }
  .header-pill {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 6px 16px;
    color: rgba(255,255,255,0.75);
    font-size: 12px;
    letter-spacing: 0.5px;
  }

  /* ── Nav bar ── */
  #nav-bar {
    background: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    padding: 0 36px;
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .nav-tab {
    padding: 14px 22px;
    font-size: 13px;
    font-weight: 500;
    color: #64748B;
    cursor: pointer;
    border-bottom: 3px solid transparent;
    transition: all 0.2s;
    letter-spacing: 0.3px;
    background: none;
    border-top: none;
    border-left: none;
    border-right: none;
    font-family: "DM Sans", sans-serif;
  }
  .nav-tab:hover { color: #1A3A8F; }

  /* ── KPI cards ── */
  .kpi-card {
    background: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 8px;
  }
  .kpi-value {
    font-family: "DM Serif Display", serif;
    font-size: 32px;
    line-height: 1;
    margin-bottom: 4px;
  }
  .kpi-sub {
    font-size: 12px;
    color: #94A3B8;
    margin: 0;
  }

  /* ── Chart cards ── */
  .chart-card {
    background: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    margin-bottom: 20px;
    overflow: hidden;
  }
  .chart-card-header {
    padding: 14px 20px 10px 20px;
    border-bottom: 1px solid #F1F5F9;
    font-size: 13px;
    font-weight: 600;
    color: #1E293B;
    letter-spacing: 0.2px;
  }
  .chart-card-body { padding: 4px 4px 0 4px; }

  /* ── Sidebar layout (shared by EDA, BiDAF, DeBERTa) ── */
  .model-layout {
    display: flex;
    gap: 24px;
    align-items: flex-start;
  }
  .model-sidebar {
    flex: 0 0 200px;
    background: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    padding: 16px 0;
    position: sticky;
    top: 20px;
  }
  .model-sidebar-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #94A3B8;
    padding: 0 16px 12px 16px;
    border-bottom: 1px solid #F1F5F9;
    margin-bottom: 8px;
  }
  .model-main { flex: 1; min-width: 0; }

  /* ── Sidebar items ── */
  .sidebar-item {
    display: block;
    padding: 9px 16px;
    font-size: 13px;
    color: #64748B;
    cursor: pointer;
    border-left: 3px solid transparent;
    transition: all 0.15s;
    background: none;
    border-top: none;
    border-right: none;
    border-bottom: none;
    text-align: left;
    width: 100%;
    font-family: "DM Sans", sans-serif;
  }
  .sidebar-item:hover { color: #1A3A8F; background: #F8FAFC; }
  .sidebar-item-active {
    color: #1A3A8F;
    font-weight: 600;
    border-left: 3px solid #2563EB;
    background: #EFF6FF;
  }

  /* ── Group selector (dropdowns, sliders) ── */
  .group-selector {
    background: #FFFFFF;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    padding: 12px 16px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .group-selector-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #94A3B8;
    white-space: nowrap;
  }

  /* ── Section label ── */
  .section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #94A3B8;
    margin: 24px 0 14px 0;
  }

  /* ── Model overview card ── */
  .overview-arch-table td {
    padding: 8px 0;
    font-size: 13px;
    font-family: "DM Sans", sans-serif;
    border-bottom: 1px solid #F1F5F9;
  }
  .overview-arch-table tr:last-child td { border-bottom: none; }
  .overview-arch-table .td-label { color: #64748B; width: 50%; }
  .overview-arch-table .td-val   { font-weight: 600; color: #1E293B; }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
'''

# ── Header ──
header = html.Div(
    id="header-bar",
    children=[
        html.Div(className="header-inner", children=[
            html.Div(className="header-title-block", children=[
                html.Div("🔍", className="header-badge"),
                html.Div([
                    html.H1("SQuAD 2.0  Text Mining", className="header-title"),
                    html.P("Interactive Exploratory Dashboard", className="header-subtitle"),
                ]),
            ]),
            html.Div("NLP · University Project", className="header-pill"),
        ])
    ]
)

# ── Navigation tabs ──
nav_bar = html.Div(
    id="nav-bar",
    children=[
        html.Button("📊  EDA",     id="nav-eda",     n_clicks=0, className="nav-tab",
                    style={"borderBottom": "3px solid #2563EB", "color": "#1A3A8F"}),
        html.Button("🧠  BiDAF",   id="nav-bidaf",   n_clicks=0, className="nav-tab"),
        html.Button("🤖  DeBERTa", id="nav-deberta", n_clicks=0, className="nav-tab"),
        dcc.Store(id="current-section", data="eda"),
        dcc.Store(id="eda-section",     data="overview"),
    ]
)

body_app = dbc.Container([
    html.Br(),
    html.Div(id="page-content"),
    html.Br(), html.Br(),
], fluid=True, style={"padding": "24px 32px"})

app.layout = html.Div(id="parent", children=[header, nav_bar, body_app])


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def graph_card(title_text, fig, height="360px"):
    return html.Div(className="chart-card", children=[
        html.Div(title_text, className="chart-card-header"),
        html.Div(className="chart-card-body", children=[
            dcc.Graph(figure=fig, style={"height": height},
                      config={"displayModeBar": False}),
        ]),
    ])


def kpi_card(label, value, sub=None, color="#0B1D51"):
    children = [
        html.P(label, className="kpi-label"),
        html.Div(f"{value}", className="kpi-value", style={"color": color}),
    ]
    if sub:
        children.append(html.P(sub, className="kpi-sub"))
    return html.Div(className="kpi-card", children=children)


def model_arch_table(rows):
    """Render a neat key-value architecture table inside a chart-card."""
    return html.Div(className="chart-card", children=[
        html.Div("Model Details", className="chart-card-header"),
        html.Div(style={"padding": "20px"}, children=[
            html.Table(
                className="overview-arch-table",
                style={"width": "100%", "borderCollapse": "collapse"},
                children=[
                    html.Tbody([
                        html.Tr([
                            html.Td(label, className="td-label"),
                            html.Td(val,   className="td-val"),
                        ]) for label, val in rows
                    ])
                ]
            )
        ])
    ])


def live_input_card(context_id, question_id, submit_id, answer_id, model_label, accent):
    return html.Div(className="chart-card", style={"padding": "0"}, children=[
        html.Div(f"Live Question Answering — {model_label}", className="chart-card-header"),
        html.Div(style={"padding": "24px"}, children=[
            dbc.Row([
                dbc.Col([
                    html.P("Context paragraph", style={"fontSize": "12px", "fontWeight": "600",
                        "letterSpacing": "0.8px", "textTransform": "uppercase",
                        "color": "#64748B", "marginBottom": "8px"}),
                    dcc.Textarea(
                        id=context_id,
                        value="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
                              "in Paris, France. It is named after the engineer Gustave Eiffel, whose "
                              "company designed and built the tower from 1887 to 1889. "
                              "The tower is 330 metres tall.",
                        style={"width": "100%", "height": "200px", "borderRadius": "8px",
                               "border": "1px solid #E2E8F0", "padding": "12px",
                               "fontFamily": "DM Sans, sans-serif", "fontSize": "13px",
                               "resize": "vertical"},
                    ),
                ], md=6),
                dbc.Col([
                    html.P("Question", style={"fontSize": "12px", "fontWeight": "600",
                        "letterSpacing": "0.8px", "textTransform": "uppercase",
                        "color": "#64748B", "marginBottom": "8px"}),
                    dcc.Textarea(
                        id=question_id,
                        value="How tall is the Eiffel Tower?",
                        style={"width": "100%", "height": "90px", "borderRadius": "8px",
                               "border": "1px solid #E2E8F0", "padding": "12px",
                               "fontFamily": "DM Sans, sans-serif", "fontSize": "13px",
                               "resize": "vertical"},
                    ),
                    html.Br(),
                    dbc.Button("Get Answer", id=submit_id, n_clicks=0,
                               style={"backgroundColor": accent, "border": "none",
                                      "borderRadius": "8px", "padding": "10px 28px",
                                      "fontFamily": "DM Sans, sans-serif", "fontWeight": "600",
                                      "fontSize": "13px", "cursor": "pointer", "color": "white"}),
                ], md=6),
            ]),
            html.Br(),
            dcc.Loading(id=f"{answer_id}-loading", type="circle", color=accent,
                        children=html.Div(id=answer_id)),
        ]),
    ])


def answer_display(answer, accent):
    if answer is None or answer == "":
        return html.Div([
            html.P("UNANSWERABLE", style={"fontSize": "11px", "fontWeight": "700",
                "letterSpacing": "1.2px", "textTransform": "uppercase",
                "color": "#F4A261", "marginBottom": "4px"}),
            html.P("The model could not find an answer in the provided context.",
                   style={"fontSize": "13px", "color": "#64748B"}),
        ], style={"background": "#FFF7ED", "borderRadius": "10px",
                  "border": "1px solid #FED7AA", "padding": "16px 20px"})

    return html.Div([
        html.P("ANSWER", style={"fontSize": "11px", "fontWeight": "700",
            "letterSpacing": "1.2px", "textTransform": "uppercase",
            "color": accent, "marginBottom": "4px"}),
        html.P(answer, style={"fontSize": "18px", "fontWeight": "600",
            "color": "#0B1D51", "fontFamily": "DM Serif Display, serif"}),
    ], style={"background": "#EFF6FF" if accent == "#2563EB" else "#F5F3FF",
              "borderRadius": "10px",
              "border": f"1px solid {'#BFDBFE' if accent == '#2563EB' else '#DDD6FE'}",
              "padding": "16px 20px"})


def model_error_display(model_path, exc):
    return html.Div([
        html.P("Model not loaded.", style={"color": "#EF4444", "fontWeight": "600"}),
        html.P(f"Make sure the model folder/file is at: {model_path}",
               style={"fontSize": "12px", "color": "#94A3B8"}),
        html.P(str(exc), style={"fontSize": "11px", "color": "#CBD5E1"}),
    ])


# ─────────────────────────────────────────────
# CALLBACKS — nav tab highlight
# ─────────────────────────────────────────────

@app.callback(
    Output("current-section", "data"),
    Output("nav-eda",     "style"),
    Output("nav-bidaf",   "style"),
    Output("nav-deberta", "style"),
    Input("nav-eda",     "n_clicks"),
    Input("nav-bidaf",   "n_clicks"),
    Input("nav-deberta", "n_clicks"),
    State("current-section", "data"),
)
def switch_tab(c_eda, c_bidaf, c_deberta, current):
    from dash import ctx
    active   = {"borderBottom": "3px solid #2563EB", "color": "#1A3A8F", "fontWeight": "600"}
    inactive = {"borderBottom": "3px solid transparent", "color": "#64748B"}

    triggered = ctx.triggered_id
    if triggered == "nav-eda":
        section = "eda"
    elif triggered == "nav-bidaf":
        section = "bidaf"
    elif triggered == "nav-deberta":
        section = "deberta"
    else:
        section = current or "eda"

    styles = {
        "eda":     [active, inactive, inactive],
        "bidaf":   [inactive, active, inactive],
        "deberta": [inactive, inactive, active],
    }
    s = styles[section]
    return section, s[0], s[1], s[2]


# ─────────────────────────────────────────────
# CALLBACKS — page content router
# ─────────────────────────────────────────────

@app.callback(
    Output("page-content", "children"),
    Input("current-section", "data"),
)
def render_page(section):

    # ── EDA ──────────────────────────────────────────────────────────────────
    if section == "eda":
        if not DATA_LOADED:
            return dbc.Alert(
                f"⚠️  Could not load train_sampled.json — make sure it is in the same folder as app.py.\n\nError: {DATA_ERROR}",
                color="danger",
            )

        sidebar_items = [
            ("overview",    "📊  Overview"),
            ("lengths",     "📏  Length Analysis"),
            ("wh",          "❓  WH-Word Analysis"),
            ("tokens",      "🔤  Tokens & Bigrams"),
            ("topics",      "📚  Top Topics"),
            ("wordclouds",  "☁️  Word Clouds"),
        ]

        sidebar = html.Div(className="model-sidebar", children=[
            html.Div("Sections", className="model-sidebar-title"),
            *[html.Button(label, id=f"sidebar-{key}", n_clicks=0, className="sidebar-item")
              for key, label in sidebar_items],
            dcc.Store(id="eda-section-local", data="overview"),
        ])

        main_content = html.Div(id="eda-main-content", className="model-main")
        return html.Div(className="model-layout", children=[sidebar, main_content])

    # ── BiDAF ─────────────────────────────────────────────────────────────────
    elif section == "bidaf":
        sidebar_items = [
            ("overview",      "🏗️  Overview"),
            ("tokenization",  "🔤  Tokenization"),
            ("metrics",       "📈  Metrics"),
            ("live",          "💬  Live Input"),
        ]
        sidebar = html.Div(className="model-sidebar", children=[
            html.Div("BiDAF", className="model-sidebar-title"),
            *[html.Button(label, id=f"bidaf-sidebar-{key}", n_clicks=0, className="sidebar-item")
              for key, label in sidebar_items],
            dcc.Store(id="bidaf-section-local", data="overview"),
        ])
        main_content = html.Div(id="bidaf-main-content", className="model-main")
        return html.Div(className="model-layout", children=[sidebar, main_content])

    # ── DeBERTa ───────────────────────────────────────────────────────────────
    elif section == "deberta":
        sidebar_items = [
            ("overview",      "🏗️  Overview"),
            ("tokenization",  "🔤  Tokenization"),
            ("metrics",       "📈  Metrics"),
            ("rag",           "🔍  RAG Pipeline"),
            ("llm",           "🧑‍⚖️  LLM Judge"),
            ("live",          "💬  Live Input"),
        ]
        sidebar = html.Div(className="model-sidebar", children=[
            html.Div("DeBERTa", className="model-sidebar-title"),
            *[html.Button(label, id=f"deberta-sidebar-{key}", n_clicks=0, className="sidebar-item")
              for key, label in sidebar_items],
            dcc.Store(id="deberta-section-local", data="overview"),
        ])
        main_content = html.Div(id="deberta-main-content", className="model-main")
        return html.Div(className="model-layout", children=[sidebar, main_content])


# ─────────────────────────────────────────────
# CALLBACKS — EDA sidebar
# ─────────────────────────────────────────────

@app.callback(
    Output("eda-section-local", "data"),
    Output("sidebar-overview",   "className"),
    Output("sidebar-lengths",    "className"),
    Output("sidebar-wh",         "className"),
    Output("sidebar-tokens",     "className"),
    Output("sidebar-topics",     "className"),
    Output("sidebar-wordclouds", "className"),
    [Input(f"sidebar-{k}", "n_clicks") for k in
     ["overview", "lengths", "wh", "tokens", "topics", "wordclouds"]],
    State("eda-section-local", "data"),
    prevent_initial_call=False,
)
def switch_eda_section(*args):
    from dash import ctx
    keys    = ["overview", "lengths", "wh", "tokens", "topics", "wordclouds"]
    current = args[-1] or "overview"
    triggered = ctx.triggered_id
    active_key = triggered.replace("sidebar-", "") if triggered and triggered.startswith("sidebar-") else current
    classes = ["sidebar-item-active" if k == active_key else "sidebar-item" for k in keys]
    return [active_key] + classes


@app.callback(
    Output("eda-main-content", "children"),
    Input("eda-section-local", "data"),
    prevent_initial_call=False,
)
def render_eda_section(section):
    if not DATA_LOADED:
        return dbc.Alert("Data not loaded.", color="danger")

    df = train_df
    n_total  = len(df)
    n_ans    = int(df["y"].sum())
    n_unans  = n_total - n_ans
    n_titles = df["title"].nunique()

    if section == "overview" or section is None:
        kpi_style = {"padding": "6px", "height": "100%"}
        card_style = {"height": "110px"}
        kpi_row = dbc.Row([
            dbc.Col(html.Div(className="kpi-card", style=card_style, children=[
                html.P("Total Questions", className="kpi-label"),
                html.Div(f"{n_total:,}", className="kpi-value", style={"color": "#0B1D51"}),
            ]), xs=12, sm=6, md=3, style=kpi_style),
            dbc.Col(html.Div(className="kpi-card", style=card_style, children=[
                html.P("Answerable", className="kpi-label"),
                html.Div(f"{n_ans:,}", className="kpi-value", style={"color": "#2563EB"}),
                html.P(f"{n_ans/n_total*100:.1f}%", className="kpi-sub"),
            ]), xs=12, sm=6, md=3, style=kpi_style),
            dbc.Col(html.Div(className="kpi-card", style=card_style, children=[
                html.P("Unanswerable", className="kpi-label"),
                html.Div(f"{n_unans:,}", className="kpi-value", style={"color": "#F4A261"}),
                html.P(f"{n_unans/n_total*100:.1f}%", className="kpi-sub"),
            ]), xs=12, sm=6, md=3, style=kpi_style),
            dbc.Col(html.Div(className="kpi-card", style=card_style, children=[
                html.P("Unique Topics", className="kpi-label"),
                html.Div(f"{n_titles:,}", className="kpi-value", style={"color": "#0B1D51"}),
            ]), xs=12, sm=6, md=3, style=kpi_style),
        ], style={"marginBottom": "24px"})
        row1 = dbc.Row([
            dbc.Col(graph_card("Label Distribution",       fig_label_dist(df)), md=5),
            dbc.Col(graph_card("Answerability by WH-Word", fig_wh(df)),         md=7),
        ])
        return html.Div([kpi_row, row1])

    elif section == "lengths":
        selector = html.Div(className="group-selector", children=[
            html.Span("Select distribution", className="group-selector-label"),
            dcc.Dropdown(
                id="length-mode",
                options=[
                    {"label": "Question length", "value": "question"},
                    {"label": "Context length",  "value": "context"},
                    {"label": "Answer length",   "value": "answer"},
                ],
                value="question", clearable=False,
                style={"minWidth": "220px", "fontFamily": "DM Sans, sans-serif"},
            ),
        ])
        return html.Div([
            selector,
            html.Div(id="length-chart-container",
                     children=[graph_card("Question Length (tokens)", fig_length_hist(df, "question"))]),
        ])

    elif section == "wh":
        return html.Div([graph_card("Answerability by WH-Word", fig_wh(df), height="420px")])

    elif section == "tokens":
        slider = html.Div(className="group-selector", children=[
            html.Span("Top N", className="group-selector-label"),
            html.Div(
                dcc.Slider(id="topn-slider", min=5, max=15, step=1, value=10,
                           marks={i: str(i) for i in range(5, 16)},
                           tooltip={"placement": "bottom", "always_visible": False}),
                style={"flex": "1", "paddingTop": "4px"},
            ),
        ])
        return html.Div([
            slider,
            html.Div(id="tokens-chart-container", children=[
                dbc.Row([
                    dbc.Col(graph_card("Top 10 Tokens in Questions",  fig_top_tokens(df, 10)),  md=6),
                    dbc.Col(graph_card("Top 10 Bigrams in Questions", fig_top_bigrams(df, 10)), md=6),
                ])
            ]),
        ])

    elif section == "topics":
        slider = html.Div(className="group-selector", children=[
            html.Span("Top N", className="group-selector-label"),
            html.Div(
                dcc.Slider(id="topn-topics-slider", min=5, max=15, step=1, value=10,
                           marks={i: str(i) for i in range(5, 16)},
                           tooltip={"placement": "bottom", "always_visible": False}),
                style={"flex": "1", "paddingTop": "4px"},
            ),
        ])
        return html.Div([
            slider,
            html.Div(id="topics-chart-container", children=[
                graph_card("Top 10 Topics — % Answerable vs Unanswerable",
                           fig_top_titles(df, 10), height=f"{10*42+80}px"),
            ]),
        ])

    elif section == "wordclouds":
        wc_img = generate_wordcloud_img(df)
        return html.Div(className="chart-card", children=[
            html.Div("Word Clouds by Topic (top 4 titles)", className="chart-card-header"),
            html.Div(
                html.Img(src=wc_img, style={"width": "100%", "borderRadius": "0 0 12px 12px"}),
                className="chart-card-body",
            ),
        ])

    return html.Div()


@app.callback(
    Output("length-chart-container", "children"),
    Input("length-mode", "value"),
    prevent_initial_call=True,
)
def update_length_chart(mode):
    label = {"question": "Question Length (tokens)",
             "context":  "Context Length (tokens)",
             "answer":   "Answer Length — answerable only (tokens)"}[mode]
    return graph_card(label, fig_length_hist(train_df, mode))


@app.callback(
    Output("tokens-chart-container", "children"),
    Input("topn-slider", "value"),
    prevent_initial_call=True,
)
def update_tokens_chart(k):
    k = k or 10
    h = f"{k*42+80}px"
    return dbc.Row([
        dbc.Col(graph_card(f"Top {k} Tokens in Questions",  fig_top_tokens(train_df, k),  height=h), md=6),
        dbc.Col(graph_card(f"Top {k} Bigrams in Questions", fig_top_bigrams(train_df, k), height=h), md=6),
    ])


@app.callback(
    Output("topics-chart-container", "children"),
    Input("topn-topics-slider", "value"),
    prevent_initial_call=True,
)
def update_topics_chart(k):
    k = k or 10
    h = f"{k*42+80}px"
    return graph_card(f"Top {k} Topics — % Answerable vs Unanswerable",
                      fig_top_titles(train_df, k), height=h)


# ─────────────────────────────────────────────
# CALLBACKS — BiDAF sidebar
# ─────────────────────────────────────────────

@app.callback(
    Output("bidaf-section-local", "data"),
    Output("bidaf-sidebar-overview",     "className"),
    Output("bidaf-sidebar-tokenization", "className"),
    Output("bidaf-sidebar-metrics",      "className"),
    Output("bidaf-sidebar-live",         "className"),
    [Input(f"bidaf-sidebar-{k}", "n_clicks") for k in ["overview", "tokenization", "metrics", "live"]],
    State("bidaf-section-local", "data"),
    prevent_initial_call=False,
)
def switch_bidaf_section(*args):
    from dash import ctx
    keys    = ["overview", "tokenization", "metrics", "live"]
    current = args[-1] or "overview"
    triggered = ctx.triggered_id
    active_key = triggered.replace("bidaf-sidebar-", "") if triggered and triggered.startswith("bidaf-sidebar-") else current
    classes = ["sidebar-item-active" if k == active_key else "sidebar-item" for k in keys]
    return [active_key] + classes


@app.callback(
    Output("bidaf-main-content", "children"),
    Input("bidaf-section-local", "data"),
    prevent_initial_call=False,
)
def render_bidaf_section(section):
    m       = BIDAF_METRICS
    accent  = "#7C3AED"

    if section == "overview" or section is None:
        kpi_row = dbc.Row([
            dbc.Col(kpi_card("Accuracy",    f"{m['accuracy']:.1%}",  "Answerability", color="#0B1D51"),  xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("Macro F1",    f"{m['macro_f1']:.1%}",  "Classification",color=accent),     xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("Exact Match", f"{m['exact_match']:.1%}","QA Span",       color="#6366F1"),  xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("F1 Span",     f"{m['f1_span']:.1%}",   "QA Span",       color="#0891B2"),  xs=12, sm=6, md=3, style={"padding": "6px"}),
        ], style={"marginBottom": "24px"})

        def mini_card(header, rows):
            return html.Div(className="chart-card", children=[
                html.Div(header, className="chart-card-header"),
                html.Div(style={"padding": "16px 20px"}, children=[
                    html.Table(
                        className="overview-arch-table",
                        style={"width": "100%", "borderCollapse": "collapse"},
                        children=[html.Tbody([
                            html.Tr([
                                html.Td(lbl, className="td-label"),
                                html.Td(val, className="td-val"),
                            ]) for lbl, val in rows
                        ])]
                    )
                ])
            ])

        desc_card = html.Div(className="chart-card", children=[
            html.Div("Architecture Overview", className="chart-card-header"),
            html.Div(style={"padding": "20px"}, children=[
                html.P([
                    html.Strong("BiDAF (Bidirectional Attention Flow)"), " is a classic extractive QA "
                    "model based on GloVe word embeddings and BiLSTM encoders. Each token's input "
                    "representation combines a 300-dim GloVe vector, a 100-dim character CNN output, "
                    "and a binary exact-match feature — totalling 401 dims, projected to 300 via a "
                    "highway network."
                ], style={"fontSize": "13px", "color": "#475569", "lineHeight": "1.7", "marginBottom": "12px"}),
                html.P([
                    "The BiDAF attention mechanism computes a similarity matrix between every context "
                    "token and question token, deriving C2Q (context-to-question) and Q2C attention. "
                    "A 2-layer modelling BiLSTM and scaled dot-product self-attention follow before the "
                    "span heads predict start/end positions. An answerability head classifies whether "
                    "any answer exists in the passage."
                ], style={"fontSize": "13px", "color": "#475569", "lineHeight": "1.7", "marginBottom": "12px"}),
                html.P([
                    html.Strong("Key limitation: "), "GloVe uses static word vectors — the same "
                    "embedding regardless of context. This is the primary bottleneck vs DeBERTa's "
                    "contextualised representations. Diagnostic Span F1 on answerable-only examples "
                    f"is just {m['span_f1_answerable']:.1%}, confirming span extraction as the weakest link."
                ], style={"fontSize": "13px", "color": "#475569", "lineHeight": "1.7"}),
            ])
        ])

        arch_card = mini_card("Model Architecture", [
            ("Model type",          "BiDAF + BiLSTM + CharCNN + Self-Attention"),
            ("Tokenizer",           "GloVe 6B.300d — 300-dim, frozen"),
            ("Char embeddings",     "CNN — 64-dim → 100-dim output"),
            ("Input dim per token", "300 + 100 + 1 (EM flag) = 401-dim"),
            ("BiLSTM hidden dim",   "100 per direction → H = 200"),
        ])

        train_card = mini_card("Training Setup", [
            ("Max context tokens",  "300  (re-centering around answer)"),
            ("Max question tokens", "40  (fixed cap)"),
            ("Batch size",          "32"),
            ("Optimizer",           "Adadelta"),
            ("Dropout",             "0.15"),
        ])

        perf_card = mini_card("Performance & Dataset", [
            ("Best Validation F1",  "49.46%"),
            ("Best Epoch",          "14"),
            ("Threshold",           "0.80  (tuned on validation set)"),
            ("Test set size",       f"{m['evaluated_examples']:,}"),
            ("Answerable ratio",    "58.2%  (5,229 / 8,990)"),
        ])


        return html.Div([
            kpi_row,
            desc_card,
            dbc.Row([
                dbc.Col(arch_card,  md=4),
                dbc.Col(train_card, md=4),
                dbc.Col(perf_card,  md=4),
            ], style={"marginTop": "4px"}),
        ])

    elif section == "tokenization":
        p_s = lambda txt, mb="12px": html.P(txt, style={"fontSize": "13px", "color": "#475569", "lineHeight": "1.7", "marginBottom": mb})
        code_b = lambda txt: html.Pre(txt, style={
            "background": "#F8FAFC", "border": "1px solid #E2E8F0", "borderRadius": "8px",
            "padding": "14px 16px", "fontSize": "12px", "fontFamily": "monospace",
            "overflowX": "auto", "marginBottom": "16px", "color": "#1E293B",
        })
        step_c = lambda title, content, color="#7C3AED": html.Div(style={
            "borderLeft": f"3px solid {color}", "paddingLeft": "16px", "marginBottom": "20px"
        }, children=[
            html.P(title, style={"fontSize": "12px", "fontWeight": "700", "letterSpacing": "0.8px",
                                  "textTransform": "uppercase", "color": color, "marginBottom": "8px"}),
            *content,
        ])
        return html.Div([
            html.Div(className="chart-card", children=[
                html.Div("BiDAF Tokenization Pipeline", className="chart-card-header"),
                html.Div(style={"padding": "24px"}, children=[

                    step_c("1 · Word-level tokenization", [
                        p_s("BiDAF uses a simple whitespace + punctuation tokenizer: text is lowercased, "
                            "punctuation is surrounded by spaces, and the result is split on whitespace. "
                            "This produces whole-word tokens — no subword splitting."),
                        code_b(
                            '"playing"       → ["playing"]\n'
                            '"unbelievable"  → ["unbelievable"]\n'
                            '"Eiffel,"       → ["eiffel", ","]\n'
                            '"New York"      → ["new", "york"]'
                        ),
                    ], color="#7C3AED"),

                    step_c("2 · GloVe vocabulary lookup", [
                        p_s("Each word token is mapped to an integer ID using a word2idx vocabulary built "
                            "from the training data (max 50,000 words). Words not in the vocabulary receive "
                            "the <UNK> index (1). Padding tokens use index 0 (all-zeros embedding)."),
                        p_s("GloVe 6B.300d pre-trained vectors are loaded: 85.9% of the 50,000 words are "
                            "found in GloVe. Unknown words get a small random initialisation from Uniform(−0.1, 0.1)."),
                        code_b(
                            "vocab size:      50,000 words\n"
                            "GloVe coverage:  42,965 / 50,000  (85.9%)\n"
                            "each word  →  300-dim vector"
                        ),
                    ], color="#7C3AED"),

                    step_c("3 · Character CNN encoding", [
                        p_s("Each token passes through a character-level CNN to capture morphology and handle "
                            "out-of-vocabulary words. Characters are embedded into 64-dim vectors; a 1D CNN "
                            "with 100 filters and kernel size 5 is applied; max-pooling over the character "
                            "dimension yields one 100-dim vector per token regardless of word length. "
                            "Tokens are truncated to MAX_WORD_LEN = 16 characters."),
                        code_b(
                            'token "playing"  (7 chars)\n'
                            "  → char embeddings:  (7, 64)\n"
                            "  → after conv1d:     (7, 100)\n"
                            "  → after max-pool:   (100,)   ← one vector per token"
                        ),
                    ], color="#6366F1"),

                    step_c("4 · Exact-match feature", [
                        p_s("A binary scalar (0 or 1) is appended to each context token's representation. "
                            "It is 1 if the lowercased context token appears anywhere in the question, "
                            "directly signalling lexical overlap between passage and question."),
                        code_b(
                            'question: "how tall is the eiffel tower"\n'
                            'context:  ["the", "eiffel", "tower", "is", "330", "metres", "tall"]\n'
                            'exact-match:  [ 1,    1,       1,     1,    0,      0,        1  ]'
                        ),
                    ], color="#0891B2"),

                    step_c("5 · Final input vector per token", [
                        p_s("The three signals are concatenated and projected through a 2-layer highway network:"),
                        code_b(
                            "GloVe word vector:   300-dim\n"
                            "CharCNN output:      100-dim\n"
                            "Exact-match flag:      1-dim\n"
                            "───────────────────────────\n"
                            "concatenated:        401-dim\n"
                            "highway projection → 300-dim"
                        ),
                    ], color="#0891B2"),

                    step_c("6 · Sequence caps & re-centering", [
                        p_s("Context is capped at MAX_CONTEXT = 300 tokens. When a passage exceeds 300 tokens, "
                            "the window is re-centered around the answer span midpoint. This contrasts with "
                            "DeBERTa's sliding window (doc_stride = 128): BiDAF cannot use doc_stride because "
                            "GloVe does not provide character-level offset mappings, making chunk-boundary "
                            "reconciliation non-trivial without native tokenizer support."),
                        p_s("Questions are capped at MAX_QUESTION = 40 tokens.", mb="0"),
                    ], color="#F59E0B"),

                ])
            ])
        ])

    elif section == "metrics":
        kpi_row = dbc.Row([
            dbc.Col(kpi_card("Accuracy",    f"{m['accuracy']:.1%}",  "Answerability", color="#0B1D51"),  xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("Macro F1",    f"{m['macro_f1']:.1%}",  "Classification",color=accent),     xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("Exact Match", f"{m['exact_match']:.1%}","QA Span",       color="#6366F1"),  xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("F1 Span",     f"{m['f1_span']:.1%}",   "QA Span (all)", color="#0891B2"),  xs=12, sm=6, md=3, style={"padding": "6px"}),
        ], style={"marginBottom": "24px"})
        row1 = dbc.Row([
            dbc.Col(graph_card("Classification Report", fig_model_cls_report(m, "BiDAF", accent)), md=7),
            dbc.Col(graph_card("Metrics Radar",         fig_model_radar(m, "BiDAF", accent)),      md=5),
        ])
        row2 = dbc.Row([
            dbc.Col(graph_card("QA Span Metrics", fig_model_span(m, "BiDAF")), md=6),
            dbc.Col(model_arch_table([
                ("Test examples",         f"{m['evaluated_examples']:,}"),
                ("Answerability thr.",    f"{m['best_threshold']:.2f} (tuned on val)"),
                ("Accuracy",             f"{m['accuracy']:.1%}"),
                ("Macro F1",             f"{m['macro_f1']:.1%}"),
                ("Precision (unans.)",   f"{m['precision_unanswerable']:.2f}"),
                ("Recall (unans.)",      f"{m['recall_unanswerable']:.2f}"),
                ("F1 (unans.)",          f"{m['f1_unanswerable']:.2f}"),
                ("Precision (ans.)",     f"{m['precision_answerable']:.2f}"),
                ("Recall (ans.)",        f"{m['recall_answerable']:.2f}"),
                ("F1 (ans.)",            f"{m['f1_answerable']:.2f}"),
                ("Exact Match (all)",    f"{m['exact_match']:.1%}"),
                ("F1 Span (all)",        f"{m['f1_span']:.1%}"),
                ("Span F1 (ans. only)",  f"{m['span_f1_answerable']:.1%}"),
            ]), md=6),
        ])
        return html.Div([kpi_row, row1, row2])

    elif section == "live":
        return live_input_card(
            context_id="bidaf-live-context",
            question_id="bidaf-live-question",
            submit_id="bidaf-live-submit",
            answer_id="bidaf-live-answer",
            model_label="BiDAF",
            accent=accent,
        )

    return html.Div()


# ─────────────────────────────────────────────
# CALLBACKS — DeBERTa sidebar
# ─────────────────────────────────────────────

@app.callback(
    Output("deberta-section-local", "data"),
    Output("deberta-sidebar-overview",     "className"),
    Output("deberta-sidebar-tokenization", "className"),
    Output("deberta-sidebar-metrics",      "className"),
    Output("deberta-sidebar-rag",          "className"),
    Output("deberta-sidebar-llm",          "className"),
    Output("deberta-sidebar-live",         "className"),
    [Input(f"deberta-sidebar-{k}", "n_clicks") for k in ["overview", "tokenization", "metrics", "rag", "llm", "live"]],
    State("deberta-section-local", "data"),
    prevent_initial_call=False,
)
def switch_deberta_section(*args):
    from dash import ctx
    keys    = ["overview", "tokenization", "metrics", "rag", "llm", "live"]
    current = args[-1] or "overview"
    triggered = ctx.triggered_id
    active_key = triggered.replace("deberta-sidebar-", "") if triggered and triggered.startswith("deberta-sidebar-") else current
    classes = ["sidebar-item-active" if k == active_key else "sidebar-item" for k in keys]
    return [active_key] + classes


@app.callback(
    Output("deberta-main-content", "children"),
    Input("deberta-section-local", "data"),
    prevent_initial_call=False,
)
def render_deberta_section(section):
    m      = DEBERTA_METRICS
    accent = "#0891B2"

    if section == "overview" or section is None:
        kpi_row = dbc.Row([
            dbc.Col(kpi_card("Accuracy",    f"{m['accuracy']:.1%}",  "Answerability", color="#0B1D51"),  xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("Macro F1",    f"{m['macro_f1']:.1%}",  "Classification",color=accent),     xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("Exact Match", f"{m['exact_match']:.1%}","QA Span",       color="#6366F1"),  xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("F1 Span",     f"{m['f1_span']:.1%}",   "QA Span",       color="#2563EB"),  xs=12, sm=6, md=3, style={"padding": "6px"}),
        ], style={"marginBottom": "24px"})

        def mini_card_d(header, rows):
            return html.Div(className="chart-card", children=[
                html.Div(header, className="chart-card-header"),
                html.Div(style={"padding": "16px 20px"}, children=[
                    html.Table(
                        className="overview-arch-table",
                        style={"width": "100%", "borderCollapse": "collapse"},
                        children=[html.Tbody([
                            html.Tr([
                                html.Td(lbl, className="td-label"),
                                html.Td(val, className="td-val"),
                            ]) for lbl, val in rows
                        ])]
                    )
                ])
            ])

        desc_card = html.Div(className="chart-card", children=[
            html.Div("Architecture Overview", className="chart-card-header"),
            html.Div(style={"padding": "20px"}, children=[
                html.P([
                    html.Strong("DeBERTa-v3-base"), " (Decoding-enhanced BERT with Disentangled Attention) "
                    "is a transformer-based pre-trained language model fine-tuned for extractive QA on SQuAD 2.0. "
                    "It uses a SentencePiece tokenizer with a ~128k vocabulary, splitting text into subword units "
                    "that handle any word — even unknown ones — by decomposing them into smaller known pieces."
                ], style={"fontSize": "13px", "color": "#475569", "lineHeight": "1.7", "marginBottom": "12px"}),
                html.P([
                    "The model processes question–context pairs with a sliding window of 384 tokens "
                    "and a stride of 128, allowing full-document coverage without information loss. "
                    "Unanswerable questions are handled by pointing the predicted start/end positions "
                    "to the [CLS] token (position 0)."
                ], style={"fontSize": "13px", "color": "#475569", "lineHeight": "1.7", "marginBottom": "12px"}),
                html.P([
                    html.Strong("Key advantage: "), "Contextualised subword embeddings — each token's "
                    "representation is informed by the full surrounding sequence via multi-head "
                    "disentangled attention (content + position modelled separately). This directly "
                    "addresses BiDAF's static-embedding bottleneck, yielding substantially higher "
                    f"F1 Span ({m['f1_span']:.1%} vs BiDAF's {BIDAF_METRICS['f1_span']:.1%})."
                ], style={"fontSize": "13px", "color": "#475569", "lineHeight": "1.7"}),
            ])
        ])

        arch_card = mini_card_d("Model Architecture", [
            ("Model type",          "DeBERTa-v3-base (transformer)"),
            ("Tokenizer",           "SentencePiece — ~128k subword vocab"),
            ("Char embeddings",     "Native subword decomposition"),
            ("Input dim per token", "768-dim  (hidden size)"),
            ("Attention type",      "Disentangled — content + position"),
        ])

        train_card = mini_card_d("Training Setup", [
            ("Max context tokens",  "384  (sliding window, stride 128)"),
            ("Max question tokens", "Dynamic  (within 384 total)"),
            ("Batch size",          "32"),
            ("Optimizer",           "AdamW + linear warmup 20%"),
            ("Dropout",             "~0.1  (DeBERTa default)"),
        ])

        perf_card = mini_card_d("Performance & Dataset", [
            ("Best Validation F1",  "92.9%"),
            ("Best Epoch",          "3  (early stopping, patience=2)"),
            ("Threshold",           "CLS-based  (no tuning needed)"),
            ("Test set size",       f"{m['evaluated_examples']:,}"),
            ("Answerable ratio",    "58.2%  (5,240 / 9,038)"),
        ])

        return html.Div([
            kpi_row,
            desc_card,
            dbc.Row([
                dbc.Col(arch_card,  md=4),
                dbc.Col(train_card, md=4),
                dbc.Col(perf_card,  md=4),
            ], style={"marginTop": "4px"}),
        ])

    elif section == "tokenization":
        p_d = lambda txt, mb="12px": html.P(txt, style={"fontSize": "13px", "color": "#475569", "lineHeight": "1.7", "marginBottom": mb})
        code_d = lambda txt: html.Pre(txt, style={
            "background": "#F8FAFC", "border": "1px solid #E2E8F0", "borderRadius": "8px",
            "padding": "14px 16px", "fontSize": "12px", "fontFamily": "monospace",
            "overflowX": "auto", "marginBottom": "16px", "color": "#1E293B",
        })
        step_d = lambda title, content, color="#0891B2": html.Div(style={
            "borderLeft": f"3px solid {color}", "paddingLeft": "16px", "marginBottom": "20px"
        }, children=[
            html.P(title, style={"fontSize": "12px", "fontWeight": "700", "letterSpacing": "0.8px",
                                  "textTransform": "uppercase", "color": color, "marginBottom": "8px"}),
            *content,
        ])
        return html.Div([
            html.Div(className="chart-card", children=[
                html.Div("DeBERTa Tokenization Pipeline", className="chart-card-header"),
                html.Div(style={"padding": "24px"}, children=[

                    step_d("1 · SentencePiece subword tokenizer", [
                        p_d("DeBERTa v3 uses a SentencePiece tokenizer with a vocabulary of ~128,000 subword "
                            "tokens. Unlike BiDAF's word-level tokenizer, SentencePiece splits text into "
                            "subword units — prefixing continuation tokens with ## — so every word can be "
                            "represented even if it was never seen during pre-training."),
                        code_d(
                            '"playing"       → ["play", "##ing"]\n'
                            '"unbelievable"  → ["un", "##believe", "##able"]\n'
                            '"Eiffel"        → ["Ei", "##ff", "##el"]\n'
                            '"New York"      → ["New", "York"]'
                        ),
                    ], color="#0891B2"),

                    step_d("2 · Special tokens structure", [
                        p_d("Three special tokens structure every input sequence:"),
                        code_d(
                            "[CLS]  position 0 — classification token.\n"
                            "       For unanswerable questions the model predicts start=0 and end=0 here.\n\n"
                            "[SEP]  separator between question and context.\n\n"
                            "[PAD]  padding tokens to reach MAX_LENGTH = 384.\n\n"
                            "Full input:  [CLS] question tokens [SEP] context tokens [SEP] [PAD]..."
                        ),
                    ], color="#0891B2"),

                    step_d("3 · Offset mapping", [
                        p_d("return_offsets_mapping=True gives each token a (start_char, end_char) pair "
                            "pointing to its position in the original string. This is essential for "
                            "converting the model's token-level start/end predictions back into the "
                            "actual answer text."),
                        code_d(
                            'Text:    "Gustave Eiffel"\n'
                            'Tokens:  ["Gust", "##ave", " Ei", "##ff", "##el"]\n'
                            'Offsets: [(0,4),  (4,7),  (8,10), (10,12), (12,14)]'
                        ),
                        p_d("sequence_ids() marks each token: None = special, 0 = question, 1 = context. "
                            "This lets the code locate answer boundaries only within the context portion."),
                    ], color="#2563EB"),

                    step_d("4 · Sliding window (doc_stride = 128)", [
                        p_d("Contexts longer than MAX_LENGTH = 384 tokens are split into overlapping "
                            "chunks with a stride of 128. Each chunk overlaps the previous by 128 tokens, "
                            "ensuring answers near chunk boundaries are never missed."),
                        code_d(
                            "Chunk 1:  tokens   0 → 383\n"
                            "Chunk 2:  tokens 256 → 383+256\n"
                            "Chunk 3:  tokens 512 → 383+512\n\n"
                            "train examples: 70,807  →  71,433 chunks  (+626 extra chunks)"
                        ),
                        p_d("If the answer falls outside a chunk's context window, that chunk is treated "
                            "as unanswerable (start=0, end=0 → [CLS] position)."),
                    ], color="#2563EB"),

                    step_d("5 · Answer span alignment", [
                        p_d("SQuAD 2.0 provides answers as character offsets. For each chunk the tokenizer "
                            "finds the first context token whose offset starts at or after answer_start, "
                            "and the last token whose offset ends at or before answer_end. "
                            "If the answer cannot be aligned to this chunk it is marked unanswerable."),
                        code_d(
                            "answer_start = 83  (char offset in original context)\n"
                            "answer_end   = 97\n\n"
                            "→ find token t_s where offsets[t_s][0] >= 83\n"
                            "→ find token t_e where offsets[t_e][1] <= 97\n"
                            "→ start_position = t_s,  end_position = t_e"
                        ),
                    ], color="#6366F1"),

                    step_d("6 · HuggingFace Dataset tensors", [
                        p_d("The tokenized output is converted to a HuggingFace Dataset returning 5 "
                            "PyTorch tensor columns per chunk:", mb="8px"),
                        code_d(
                            "input_ids        (384,)  — subword token IDs\n"
                            "attention_mask   (384,)  — 1=real token, 0=padding\n"
                            "token_type_ids   (384,)  — 0=question, 1=context\n"
                            "start_positions  ()      — answer start token index\n"
                            "end_positions    ()      — answer end token index"
                        ),
                    ], color="#6366F1"),

                ])
            ])
        ])

    elif section == "metrics":
        kpi_row = dbc.Row([
            dbc.Col(kpi_card("Accuracy",    f"{m['accuracy']:.1%}",  "Answerability", color="#0B1D51"),  xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("Macro F1",    f"{m['macro_f1']:.1%}",  "Classification",color=accent),     xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("Exact Match", f"{m['exact_match']:.1%}","QA Span",       color="#6366F1"),  xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("F1 Span",     f"{m['f1_span']:.1%}",   "QA Span",       color="#2563EB"),  xs=12, sm=6, md=3, style={"padding": "6px"}),
        ], style={"marginBottom": "24px"})
        row1 = dbc.Row([
            dbc.Col(graph_card("Classification Report", fig_model_cls_report(m, "DeBERTa", accent)), md=7),
            dbc.Col(graph_card("Metrics Radar",         fig_model_radar(m, "DeBERTa", accent)),      md=5),
        ])
        row2 = dbc.Row([
            dbc.Col(graph_card("QA Span Metrics", fig_model_span(m, "DeBERTa")), md=6),
            dbc.Col(model_arch_table([
                ("Test examples",        f"{m['evaluated_examples']:,}"),
                ("Accuracy",            f"{m['accuracy']:.1%}"),
                ("Macro F1",            f"{m['macro_f1']:.1%}"),
                ("Precision (unans.)",  f"{m['precision_unanswerable']:.2f}"),
                ("Recall (unans.)",     f"{m['recall_unanswerable']:.2f}"),
                ("F1 (unans.)",         f"{m['f1_unanswerable']:.2f}"),
                ("Precision (ans.)",    f"{m['precision_answerable']:.2f}"),
                ("Recall (ans.)",       f"{m['recall_answerable']:.2f}"),
                ("F1 (ans.)",           f"{m['f1_answerable']:.2f}"),
                ("Exact Match (all)",   f"{m['exact_match']:.1%}"),
                ("F1 Span (all)",       f"{m['f1_span']:.1%}"),
            ]), md=6),
        ])
        return html.Div([kpi_row, row1, row2])

    elif section == "rag":
        p_r = lambda txt, mb="12px": html.P(txt, style={"fontSize": "13px", "color": "#475569", "lineHeight": "1.7", "marginBottom": mb})
        code_r = lambda txt: html.Pre(txt, style={
            "background": "#F8FAFC", "border": "1px solid #E2E8F0", "borderRadius": "8px",
            "padding": "14px 16px", "fontSize": "12px", "fontFamily": "monospace",
            "overflowX": "auto", "marginBottom": "16px", "color": "#1E293B",
        })
        step_r = lambda title, content, color="#0891B2": html.Div(style={
            "borderLeft": f"3px solid {color}", "paddingLeft": "16px", "marginBottom": "20px"
        }, children=[
            html.P(title, style={"fontSize": "12px", "fontWeight": "700", "letterSpacing": "0.8px",
                                  "textTransform": "uppercase", "color": color, "marginBottom": "8px"}),
            *content,
        ])

        rag_kpis = dbc.Row([
            dbc.Col(kpi_card("Context index", "16,629", "unique passages", color="#0B1D51"), xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("Retriever",  "Hybrid", "semantic + lexical", color=accent),   xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("α weight",   "0.70",   "semantic priority",  color="#6366F1"), xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("Top-k (eval)", "5",    "retrieved contexts", color="#F59E0B"), xs=12, sm=6, md=3, style={"padding": "6px"}),
        ], style={"marginBottom": "24px"})

        return html.Div([
            rag_kpis,
            html.Div(className="chart-card", children=[
                html.Div("RAG Pipeline — Architecture", className="chart-card-header"),
                html.Div(style={"padding": "24px"}, children=[

                    step_r("Architecture overview", [
                        p_r("The system follows a Retrieval-Augmented Generation (RAG) architecture for "
                            "extractive QA. Instead of providing the gold context directly, the pipeline "
                            "first retrieves the most relevant passages from a corpus index of 16,629 unique "
                            "contexts (deduplicated across train/val/test), then passes them to DeBERTa for "
                            "answer extraction."),
                    ], color="#0891B2"),

                    step_r("1 · Dense semantic retrieval", [
                        p_r("Contexts are encoded into 384-dim dense vectors using the pretrained sentence "
                            "encoder multi-qa-MiniLM-L6-cos-v1. Both contexts and questions are enriched "
                            "with their Wikipedia title (e.g. 'Eiffel Tower. The tower is…') to help the "
                            "encoder understand the topic."),
                        p_r("Semantic similarity is measured via cosine similarity between the query "
                            "embedding and all context embeddings. This captures meaning and paraphrases "
                            "even when exact keywords differ."),
                        code_r(
                            "embedder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')\n"
                            "ctx_embeddings = embedder.encode(contexts)   # shape: (16629, 384)\n"
                            "sem_scores = cosine_similarity(q_emb, ctx_embeddings)"
                        ),
                    ], color="#0891B2"),

                    step_r("2 · Lexical retrieval (TF-IDF)", [
                        p_r("A TF-IDF index is built over unigrams and bigrams (max 50,000 features). "
                            "TF-IDF weights rare, informative terms more heavily, capturing exact keyword "
                            "matches that semantic retrieval might miss (e.g. proper nouns, technical terms)."),
                        code_r(
                            "vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))\n"
                            "tfidf_matrix = vectorizer.fit_transform(contexts)  # (16629, 50000)\n"
                            "lex_scores = cosine_similarity(q_tfidf, tfidf_matrix)"
                        ),
                    ], color="#6366F1"),

                    step_r("3 · Hybrid scoring", [
                        p_r("The two scores are min-max normalised and combined with a weighted sum. "
                            "α = 0.7 prioritises semantic similarity while retaining 30% weight for "
                            "lexical precision. The top-k contexts by combined score are returned."),
                        code_r(
                            "score = α · s_semantic + (1 − α) · s_lexical\n"
                            "      = 0.7 · s_semantic + 0.3 · s_lexical"
                        ),
                    ], color="#6366F1"),

                    step_r("4 · Reader stage (DeBERTa)", [
                        p_r("Each retrieved context is passed to DeBERTa as [CLS] question [SEP] context [SEP]. "
                            "The model outputs start_logits and end_logits for every token. The final answer "
                            "is the span maximising start_logit[i] + end_logit[j] across all retrieved "
                            "contexts, further weighted by the retrieval score (weight = 0.1)."),
                        code_r(
                            "final_score = span_score + 0.1 × retrieval_score\n\n"
                            "if null_score > span_score:\n"
                            "    answer = ''   # unanswerable\n"
                            "else:\n"
                            "    answer = context[start_char:end_char]"
                        ),
                    ], color="#2563EB"),

                    step_r("5 · Gold context vs RAG — performance gap", [
                        p_r("DeBERTa with the gold context (direct evaluation) substantially outperforms "
                            "the RAG pipeline. Any retrieval error propagates directly to the reader, "
                            "lowering EM and F1. The retriever was limited to k = 5 for computational "
                            "reasons, which reduced the probability of including the truly relevant context. "
                            "This gap is expected and does not imply retrieval augmentation is ineffective — "
                            "larger k and a stronger retriever would narrow it considerably.", mb="0"),
                    ], color="#F59E0B"),

                ])
            ]),
            html.Br(),
            # ── Live retrieval + DeBERTa answer ──────────────────────────
            html.Div(className="chart-card", style={"padding": "0"}, children=[
                html.Div("Live RAG — Retrieve & Answer", className="chart-card-header"),
                html.Div(style={"padding": "24px"}, children=[
                    html.P("Type any question. The retriever returns the top-10 most relevant contexts "
                           "from the index, then DeBERTa extracts the answer from the best one.",
                           style={"fontSize": "13px", "color": "#64748B", "marginBottom": "16px"}),
                    html.P("Question", style={"fontSize": "12px", "fontWeight": "600",
                        "letterSpacing": "0.8px", "textTransform": "uppercase",
                        "color": "#64748B", "marginBottom": "8px"}),
                    dcc.Textarea(
                        id="rag-live-question",
                        value="How did the National Archives make its documents available online?",
                        style={"width": "100%", "height": "80px", "borderRadius": "8px",
                               "border": "1px solid #E2E8F0", "padding": "12px",
                               "fontFamily": "DM Sans, sans-serif", "fontSize": "13px",
                               "resize": "vertical"},
                    ),
                    html.Br(),
                    dbc.Button("Retrieve & Answer", id="rag-live-submit", n_clicks=0,
                               style={"backgroundColor": accent, "border": "none",
                                      "borderRadius": "8px", "padding": "10px 28px",
                                      "fontFamily": "DM Sans, sans-serif", "fontWeight": "600",
                                      "fontSize": "13px", "cursor": "pointer", "color": "white"}),
                    html.Br(), html.Br(),
                    dcc.Loading(id="rag-live-loading", type="circle", color=accent,
                                children=html.Div(id="rag-live-output")),
                ]),
            ]),
        ])

    elif section == "llm":
        p_l = lambda txt, mb="12px": html.P(txt, style={"fontSize": "13px", "color": "#475569", "lineHeight": "1.7", "marginBottom": mb})
        code_l = lambda txt: html.Pre(txt, style={
            "background": "#F8FAFC", "border": "1px solid #E2E8F0", "borderRadius": "8px",
            "padding": "14px 16px", "fontSize": "12px", "fontFamily": "monospace",
            "overflowX": "auto", "marginBottom": "16px", "color": "#1E293B",
        })
        step_l = lambda title, content, color="#6366F1": html.Div(style={
            "borderLeft": f"3px solid {color}", "paddingLeft": "16px", "marginBottom": "20px"
        }, children=[
            html.P(title, style={"fontSize": "12px", "fontWeight": "700", "letterSpacing": "0.8px",
                                  "textTransform": "uppercase", "color": color, "marginBottom": "8px"}),
            *content,
        ])

        llm_kpis = dbc.Row([
            dbc.Col(kpi_card("Sample size",  "800",   "answerable examples", color="#0B1D51"), xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("LLM judge",    "LLaMA", "3.1-8b-instant / Groq", color="#6366F1"), xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("Auto EM",      "65.4%", "on 800-sample subset",  color=accent),   xs=12, sm=6, md=3, style={"padding": "6px"}),
            dbc.Col(kpi_card("LLM errors",   "175",   "EM=0 flagged by LLM",   color="#F4A261"), xs=12, sm=6, md=3, style={"padding": "6px"}),
        ], style={"marginBottom": "24px"})

        return html.Div([
            llm_kpis,
            html.Div(className="chart-card", children=[
                html.Div("LLM-Based Evaluation — LLaMA 3.1 Judge via Groq", className="chart-card-header"),
                html.Div(style={"padding": "24px"}, children=[

                    step_l("Motivation", [
                        p_l("Standard automatic metrics — Exact Match and F1 — rely on lexical overlap and "
                            "can fail to capture semantic equivalence in borderline cases. A prediction like "
                            "'research project' vs gold 'a research project' gets EM=0 despite being "
                            "semantically identical. An external LLM judge provides a more nuanced, "
                            "meaning-aware second opinion."),
                    ], color="#6366F1"),

                    step_l("Evaluation setup", [
                        p_l("800 answerable test examples were randomly sampled. DeBERTa was run on all of "
                            "them using the gold context (direct evaluation, no RAG). Each prediction was "
                            "then sent to LLaMA 3.1-8b-instant via the Groq API, which assigned EM and F1 "
                            "scores and provided a brief reasoning for each."),
                        code_l(
                            "model:             llama-3.1-8b-instant  (Groq API)\n"
                            "temperature:       0.0  (deterministic)\n"
                            "sample size:       800 answerable examples\n"
                            "retry logic:       3 attempts + local fallback on JSON parse error\n"
                            "checkpoint every:  10 samples  (resumable)"
                        ),
                    ], color="#6366F1"),

                    step_l("Prompt structure", [
                        p_l("The LLM received question, context, predicted answer, and gold answer, and was "
                            "asked to return a JSON object with EM (0 or 1), F1 (0.0–1.0) and a brief "
                            "explanation. Markdown backticks are stripped before JSON parsing; any invalid "
                            "response falls back to local EM/F1 computation."),
                        code_l(
                            'You are an expert judge evaluating a QA system.\n\n'
                            'QUESTION:         {question}\n'
                            'CONTEXT:          {context}\n'
                            'PREDICTED ANSWER: {prediction}\n'
                            'GOLD ANSWER:      {gold_answer}\n\n'
                            'Return ONLY a valid JSON:\n'
                            '{ "em": 0|1, "f1": float, "reasoning": "..." }'
                        ),
                    ], color="#2563EB"),

                    step_l("Key findings — LLM vs automatic metrics", [
                        p_l("The LLM judge is more semantically flexible than EM: several predictions "
                            "marked incorrect by EM were accepted by the LLM because they preserve the "
                            "essential meaning despite minor lexical differences:"),
                        code_l(
                            'Gold: "a research project"   Pred: "research project"   → article variation\n'
                            'Gold: "a steel plate"         Pred: "steel plate"        → article variation\n'
                            'Gold: "the United States"     Pred: "United States"      → article variation\n\n'
                            "LLM EM=1 for all three above; standard EM=0"
                        ),
                        p_l("At the same time, the LLM is not uniformly lenient — it still flags genuinely "
                            "wrong predictions where both EM and F1 remain low:"),
                        code_l(
                            'Gold: "The Battle of Osan"    Pred: "Battle of Taejon"  → genuinely wrong\n'
                            'Gold: "minister"              Pred: "minister and schoolmaster"\n'
                            "                              → partial: LLM assigns intermediate F1, EM=0"
                        ),
                        p_l("175 out of 800 predictions were flagged as EM=0 by the LLM. This confirms "
                            "that automatic metrics may be too rigid in borderline cases, while the LLM "
                            "provides a more nuanced assessment of answer quality.", mb="0"),
                    ], color="#2563EB"),

                ])
            ]),
            html.Br(),
            # ── Comparison table — cell 58 ────────────────────────────────
            html.Div(className="chart-card", children=[
                html.Div(
                    "Notebook Output — LLM vs DeBERTa Comparison  "
                    "(cases where deberta_em = False, first 20)",
                    className="chart-card-header"
                ),
                html.Div(style={"padding": "0", "overflowX": "auto"}, children=[
                    html.Table(style={
                        "width": "100%", "borderCollapse": "collapse",
                        "fontFamily": "monospace", "fontSize": "12px",
                    }, children=[
                        html.Thead(html.Tr([
                            html.Th(col, style={
                                "padding": "10px 14px", "textAlign": "left" if i < 2 else "center",
                                "background": "#0F172A", "color": "#94A3B8",
                                "fontWeight": "600", "letterSpacing": "0.5px",
                                "borderBottom": "2px solid #1E293B", "whiteSpace": "nowrap",
                            }) for i, col in enumerate([
                                "#", "gold_answer", "deberta_pred",
                                "llm_f1", "deberta_f1", "llm_em", "deberta_em"
                            ])
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td(cell, style={
                                    "padding": "8px 14px",
                                    "borderBottom": "1px solid #F1F5F9",
                                    "textAlign": "left" if j < 2 else "center",
                                    "color": (
                                        "#EF4444" if (j == 6 and cell == "False")
                                        else "#F59E0B" if (j == 5 and cell in ("0", 0))
                                        else "#22C55E" if (j == 5 and cell in ("1", 1))
                                        else "#1E293B"
                                    ),
                                    "fontWeight": "500" if j in (3, 4, 5, 6) else "400",
                                    "maxWidth": "220px" if j in (1, 2) else "auto",
                                    "overflow": "hidden", "textOverflow": "ellipsis",
                                    "whiteSpace": "nowrap",
                                }) for j, cell in enumerate(row)
                            ], style={"background": "#FFFFFF" if i % 2 == 0 else "#F8FAFC"})
                            for i, row in enumerate([
                                ("7",  "Time magazine's 'Man of the Year'",          "Man of the Year",                           "0.5",  "0.800", "0", "False"),
                                ("11", "40 million bacterial cells",                  "40 million",                                "1.0",  "0.667", "1", "False"),
                                ("14", "Battle of Taejon",                            "The Battle of Osan",                        "0.0",  "0.571", "0", "False"),
                                ("15", "24th Infantry Division",                      "24th",                                      "0.8",  "0.500", "1", "False"),
                                ("19", "a research project",                          "research project",                          "1.0",  "0.800", "1", "False"),
                                ("22", "three percent of the world's Jewish pop.",    "three percent",                             "0.0",  "0.200", "0", "False"),
                                ("26", "guide the people",                            "Allah Must assign someone similar...",       "0.0",  "0.222", "0", "False"),
                                ("27", "characters like faces and heels",             "faces and heels",                           "0.5",  "0.750", "0", "False"),
                                ("28", "a steel plate",                               "steel plate",                               "1.0",  "0.800", "1", "False"),
                                ("29", "Panhandle or Inside Passage",                 "the Panhandle or Inside Passage",           "1.0",  "0.889", "1", "False"),
                                ("30", "about 160 million years ago",                 "160 million years ago",                     "1.0",  "0.889", "1", "False"),
                                ("35", "In 1896",                                     "1896",                                      "1.0",  "0.667", "1", "False"),
                                ("36", "minister and schoolmaster",                   "minister",                                  "0.5",  "0.500", "0", "False"),
                                ("40", "Brick Gothic architecture",                   "Brick",                                     "0.5",  "0.500", "0", "False"),
                                ("41", "charter",                                     "lessened the penalties for possession...",   "0.0",  "0.000", "0", "False"),
                                ("46", "maintenance of a high and stable body temp.", "for the maintenance of a high and...",      "0.8",  "0.889", "0", "False"),
                                ("47", "once a year.",                                "once a year",                               "1.0",  "1.000", "1", "False"),
                                ("48", "such waste",                                  "waste",                                     "0.5",  "0.667", "0", "False"),
                                ("49", "using a single-lens microscope of his design","single-lens microscope",                    "0.5",  "0.400", "0", "False"),
                                ("51", "United States",                               "the United States",                         "1.0",  "0.800", "1", "False"),
                            ])
                        ]),
                    ]),
                ]),
            ]),
        ])

    elif section == "live":
        return live_input_card(
            context_id="deberta-live-context",
            question_id="deberta-live-question",
            submit_id="deberta-live-submit",
            answer_id="deberta-live-answer",
            model_label="DeBERTa",
            accent=accent,
        )

    return html.Div()


# ─────────────────────────────────────────────
# CALLBACKS — Live inference
# ─────────────────────────────────────────────

@app.callback(
    Output("bidaf-live-answer", "children"),
    Input("bidaf-live-submit", "n_clicks"),
    State("bidaf-live-context",  "value"),
    State("bidaf-live-question", "value"),
    prevent_initial_call=True,
)
def run_bidaf_inference(n_clicks, context, question):
    accent = "#7C3AED"
    if not context or not question:
        return html.P("Please fill in both the context and the question.",
                      style={"color": "#EF4444", "fontSize": "13px"})
    try:
        answer = answer_question_bidaf(question.strip(), context.strip())
    except Exception as ex:
        return model_error_display(BIDAF_PATH, ex)
    return answer_display(answer, accent)


@app.callback(
    Output("deberta-live-answer", "children"),
    Input("deberta-live-submit", "n_clicks"),
    State("deberta-live-context",  "value"),
    State("deberta-live-question", "value"),
    prevent_initial_call=True,
)
def run_deberta_inference(n_clicks, context, question):
    accent = "#0891B2"
    if not context or not question:
        return html.P("Please fill in both the context and the question.",
                      style={"color": "#EF4444", "fontSize": "13px"})
    try:
        answer = answer_question_deberta(question.strip(), context.strip())
    except Exception as ex:
        return model_error_display(DEBERTA_PATH, ex)
    return answer_display(answer, accent)



# ─────────────────────────────────────────────
# CALLBACKS — RAG live retrieval + answer
# ─────────────────────────────────────────────

@app.callback(
    Output("rag-live-output", "children"),
    Input("rag-live-submit", "n_clicks"),
    State("rag-live-question", "value"),
    prevent_initial_call=True,
)
def run_rag_retrieval(n_clicks, question):
    if not question or not question.strip():
        return html.P("Please enter a question.", style={"color": "#EF4444", "fontSize": "13px"})
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer as _Tfidf
        from sklearn.metrics.pairwise import cosine_similarity as _cos

        if not DATA_LOADED:
            return html.P("Dataset not loaded — cannot build retrieval index.",
                          style={"color": "#EF4444", "fontSize": "13px"})

        # Build TF-IDF index from deduplicated train contexts
        dedup    = train_df.drop_duplicates(subset="context")
        all_ctx  = dedup["context"].tolist()
        all_ttl  = dedup["title"].tolist()
        enriched = [f"{t.replace('_',' ')}. {c}" for t, c in zip(all_ttl, all_ctx)]

        vect   = _Tfidf(max_features=30000, ngram_range=(1, 2))
        matrix = vect.fit_transform(enriched)
        q_vec  = vect.transform([f"question: {question.strip()}"])
        scores = _cos(q_vec, matrix).flatten()
        top10  = scores.argsort()[-10:][::-1]

        # Context rows
        ctx_rows = []
        for rank, idx in enumerate(top10, 1):
            score   = float(scores[idx])
            snippet = all_ctx[idx][:250].replace("\n", " ")
            ctx_rows.append(html.Div(style={
                "borderLeft": "3px solid #0891B2", "paddingLeft": "14px", "marginBottom": "14px",
            }, children=[
                html.Div([
                    html.Span(f"[{rank}]", style={"fontWeight": "700", "color": "#0891B2",
                                                   "fontFamily": "monospace", "marginRight": "8px"}),
                    html.Span(f"Score: {score:.4f}", style={"fontSize": "11px", "color": "#94A3B8",
                                                             "fontFamily": "monospace"}),
                ], style={"marginBottom": "5px"}),
                html.P(snippet + ("…" if len(all_ctx[idx]) > 250 else ""),
                       style={"fontSize": "12px", "color": "#475569", "lineHeight": "1.6", "margin": "0"}),
            ]))

        # DeBERTa answer from top-1 context
        best_ctx = all_ctx[top10[0]]
        try:
            answer = answer_question_deberta(question.strip(), best_ctx)
        except Exception:
            answer = None

        answer_box = answer_display(answer, "#0891B2")

        return html.Div([
            html.P([html.Strong("Question: "), question.strip()],
                   style={"fontSize": "13px", "color": "#1E293B", "marginBottom": "16px",
                          "fontStyle": "italic"}),
            html.P("Top 10 retrieved contexts  (TF-IDF — live demo):",
                   style={"fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.8px",
                          "textTransform": "uppercase", "color": "#94A3B8", "marginBottom": "12px"}),
            *ctx_rows,
            html.Hr(style={"border": "none", "borderTop": "1px solid #E2E8F0", "margin": "20px 0"}),
            html.P("DeBERTa answer  (from top-1 context):",
                   style={"fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.8px",
                          "textTransform": "uppercase", "color": "#94A3B8", "marginBottom": "12px"}),
            answer_box,
        ], style={"background": "#F8FAFC", "borderRadius": "10px",
                  "border": "1px solid #E2E8F0", "padding": "20px"})

    except Exception as ex:
        return html.P(f"Retrieval error: {ex}", style={"color": "#EF4444", "fontSize": "13px"})


if __name__ == "__main__":
    app.run(debug=True)
