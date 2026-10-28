"""
SMS Shield - Intelligent SMS Spam Detection
-------------------------------------------
A college ML microproject.

Pipeline : TF-IDF Vectorizer  ->  Multinomial Naive Bayes
Dataset  : UCI SMS Spam Collection (downloaded automatically)
Frontend : Streamlit + custom CSS

Run locally with:  streamlit run app.py
"""

import csv
import html
import io
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# --------------------------------------------------------------------------
st.set_page_config(page_title="SMS Shield", page_icon="🛡️", layout="wide")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
# Primary source: the official UCI Machine Learning Repository (zip file).
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# Backup source: the same dataset as a plain tab-separated file on GitHub.
GITHUB_TSV_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
    "master/data/sms.tsv"
)

DISPLAY_NAMES = {"ham": "LEGITIMATE", "spam": "SPAM"}

COLOR_SPAM = "#F43F5E"
COLOR_HAM = "#34D399"
COLOR_PURPLE = "#8B5CF6"
COLOR_CYAN = "#22D3EE"
COLOR_TEXT = "#E2E8F0"

SAMPLE_MESSAGES = {
    "🎁 Prize scam": (
        "WINNER!! You have been selected to receive a £900 prize reward. "
        "Call 09061701461 now to claim. Claim code KL341. Valid 12 hours only."
    ),
    "💳 Free offer": (
        "Free entry in a weekly competition to win FA Cup tickets! "
        "Text FA to 87121 to receive entry question. T&C apply."
    ),
    "👋 Friendly chat": "Hey, are we still meeting at the library at 5 pm today?",
    "🍕 Daily life": "Can you please pick up some milk and bread on your way home?",
}

# --------------------------------------------------------------------------
# Custom CSS (dark navy background, purple + cyan accents)
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, p, label, li, span, div {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

.stApp {
    background: radial-gradient(1200px 600px at 10% -10%, #1b1f4b 0%, transparent 60%),
                radial-gradient(900px 500px at 100% 0%, #123a55 0%, transparent 55%),
                #0B1020;
    color: #E2E8F0;
}

header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* ---------- Hero header ---------- */
.hero { text-align: center; padding: 1.2rem 0 1.6rem 0; }
.hero-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #22D3EE;
    background: rgba(34, 211, 238, 0.10);
    border: 1px solid rgba(34, 211, 238, 0.35);
    margin-bottom: 0.9rem;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(90deg, #8B5CF6 0%, #22D3EE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle { font-size: 1.25rem; font-weight: 500; color: #CBD5E1; margin: 0.4rem 0 0.9rem 0; }
.hero-text { max-width: 720px; margin: 0 auto; color: #94A3B8; font-size: 1rem; line-height: 1.65; }

/* ---------- Cards ---------- */
.card {
    background: linear-gradient(180deg, rgba(30, 36, 77, 0.85) 0%, rgba(19, 26, 51, 0.9) 100%);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 18px;
    padding: 1.3rem 1.4rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    margin-bottom: 1rem;
    height: 100%;
}
.card h4 { margin: 0 0 0.5rem 0; font-size: 1.05rem; font-weight: 700; color: #F1F5F9; }
.card p, .card li { color: #A9B6CB; font-size: 0.95rem; line-height: 1.6; }
.card ul { padding-left: 1.2rem; margin: 0.3rem 0 0 0; }

.section-title { font-size: 1.5rem; font-weight: 700; color: #F8FAFC; margin: 0.3rem 0 0.2rem 0; }
.section-sub { color: #94A3B8; margin-bottom: 1rem; font-size: 0.95rem; }

/* ---------- Metric cards ---------- */
.metric-card {
    background: linear-gradient(180deg, rgba(30, 36, 77, 0.85) 0%, rgba(19, 26, 51, 0.9) 100%);
    border: 1px solid rgba(34, 211, 238, 0.25);
    border-radius: 18px;
    padding: 1.1rem 1rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    margin-bottom: 1rem;
}
.metric-label { color: #94A3B8; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }
.metric-value {
    font-size: 2.1rem; font-weight: 800; margin-top: 0.2rem;
    background: linear-gradient(90deg, #8B5CF6, #22D3EE);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}

/* ---------- Result card ---------- */
.result-card { border-radius: 18px; padding: 1.4rem 1.5rem; margin: 1rem 0; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35); }
.result-spam { background: linear-gradient(135deg, rgba(244, 63, 94, 0.20), rgba(139, 92, 246, 0.12)); border: 1px solid rgba(244, 63, 94, 0.6); }
.result-ham  { background: linear-gradient(135deg, rgba(52, 211, 153, 0.20), rgba(34, 211, 238, 0.10)); border: 1px solid rgba(52, 211, 153, 0.6); }
.result-label { font-size: 0.8rem; letter-spacing: 0.1em; font-weight: 600; color: #94A3B8; text-transform: uppercase; }
.result-verdict { font-size: 2rem; font-weight: 800; margin: 0.1rem 0 0.4rem 0; }
.verdict-spam { color: #FB7185; }
.verdict-ham { color: #34D399; }
.result-text { color: #CBD5E1; font-size: 0.97rem; line-height: 1.6; }
.chip {
    display: inline-block; padding: 0.15rem 0.7rem; margin: 0.2rem 0.3rem 0 0;
    border-radius: 999px; font-size: 0.85rem; font-weight: 600;
    background: rgba(139, 92, 246, 0.18); border: 1px solid rgba(139, 92, 246, 0.5); color: #DDD6FE;
}
.disclaimer {
    margin-top: 0.8rem; padding: 0.7rem 0.9rem; border-radius: 12px;
    background: rgba(250, 204, 21, 0.08); border: 1px solid rgba(250, 204, 21, 0.3);
    color: #FDE68A; font-size: 0.85rem;
}

/* ---------- Workflow steps ---------- */
.step-card {
    background: linear-gradient(180deg, rgba(30, 36, 77, 0.85) 0%, rgba(19, 26, 51, 0.9) 100%);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 18px; padding: 1.1rem 0.9rem; text-align: center; height: 100%;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35); margin-bottom: 1rem;
}
.step-num {
    display: inline-block; width: 2rem; height: 2rem; line-height: 2rem; border-radius: 50%;
    font-weight: 700; color: #0B1020; background: linear-gradient(135deg, #8B5CF6, #22D3EE); margin-bottom: 0.5rem;
}
.step-title { font-weight: 700; color: #F1F5F9; font-size: 0.98rem; margin-bottom: 0.3rem; }
.step-text { color: #A9B6CB; font-size: 0.85rem; line-height: 1.5; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid rgba(148, 163, 184, 0.2); flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 12px 12px 0 0; padding: 0.6rem 1.1rem;
    color: #94A3B8; font-weight: 600;
}
.stTabs [aria-selected="true"] { color: #22D3EE !important; background: rgba(34, 211, 238, 0.08); }
.stTabs [data-baseweb="tab-highlight"] { background: linear-gradient(90deg, #8B5CF6, #22D3EE); }

/* ---------- Inputs and buttons ---------- */
.stTextArea textarea {
    background: rgba(15, 23, 42, 0.85) !important; color: #F1F5F9 !important;
    border: 1px solid rgba(139, 92, 246, 0.4) !important; border-radius: 14px !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus { border-color: #22D3EE !important; box-shadow: 0 0 0 1px #22D3EE !important; }

.stButton > button, .stDownloadButton > button {
    border-radius: 12px; font-weight: 600; width: 100%;
    background: rgba(30, 36, 77, 0.9); color: #E2E8F0;
    border: 1px solid rgba(139, 92, 246, 0.45);
}
.stButton > button:hover { border-color: #22D3EE; color: #22D3EE; }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, #8B5CF6 0%, #22D3EE 100%);
    color: #0B1020; border: none; font-weight: 700;
    box-shadow: 0 8px 22px rgba(139, 92, 246, 0.35);
}
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    color: #0B1020; filter: brightness(1.08);
}

.footer { text-align: center; color: #64748B; font-size: 0.85rem; margin-top: 2.5rem; }

/* ---------- Mobile ---------- */
@media (max-width: 768px) {
    .hero-title { font-size: 2.3rem; }
    .hero-subtitle { font-size: 1.05rem; }
    .metric-value { font-size: 1.7rem; }
    .result-verdict { font-size: 1.6rem; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
"""


def render_html(markup):
    """Show an HTML string. Newlines are removed so Markdown never turns it into a code block."""
    flat = " ".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 1. Dataset loading and cleaning
# --------------------------------------------------------------------------
def download_bytes(url):
    """Download a URL and return its raw bytes (raises an error on failure)."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (SMS-Shield student project)"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def parse_tab_separated(raw_bytes):
    """Convert the raw dataset file (label <TAB> message, no header) into a DataFrame."""
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")

    return pd.read_csv(
        io.StringIO(text),
        sep="\t",
        header=None,
        names=["label", "message"],
        quoting=csv.QUOTE_NONE,  # messages contain quote characters
        on_bad_lines="skip",     # skip malformed rows instead of crashing
        dtype=str,
    )


def read_uci_zip(zip_bytes):
    """Open the UCI zip file and read the 'SMSSpamCollection' data file inside it."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        data_files = [n for n in archive.namelist() if "readme" not in n.lower() and not n.endswith("/")]
        if not data_files:
            raise ValueError("The downloaded zip file does not contain a data file.")
        preferred = [n for n in data_files if n.endswith("SMSSpamCollection")]
        chosen = preferred[0] if preferred else data_files[0]
        return parse_tab_separated(archive.read(chosen))


def clean_dataset(df):
    """Validate and clean the data. Returns the clean DataFrame and a summary dictionary."""
    rows_before = len(df)

    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].str.strip()

    # Keep only valid labels and non-empty messages
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"] != ""]
    rows_after_validation = len(df)

    # Remove exact duplicate messages so the same SMS cannot be in both train and test sets
    df = df.drop_duplicates(subset="message").reset_index(drop=True)

    summary = {
        "rows_raw": rows_before,
        "invalid_removed": rows_before - rows_after_validation,
        "duplicates_removed": rows_after_validation - len(df),
        "rows_clean": len(df),
    }

    if len(df) < 1000 or df["label"].nunique() < 2:
        raise ValueError("The dataset looks incomplete or invalid after cleaning.")
    return df, summary


@st.cache_data(show_spinner="Downloading the SMS Spam Collection dataset...")
def load_dataset():
    """Download (UCI first, GitHub backup), clean and return the dataset. Cached after the first run."""
    errors = []
    sources = [
        ("UCI Machine Learning Repository", UCI_ZIP_URL, True),
        ("GitHub mirror", GITHUB_TSV_URL, False),
    ]
    for name, url, is_zip in sources:
        try:
            raw = download_bytes(url)
            df = read_uci_zip(raw) if is_zip else parse_tab_separated(raw)
            clean_df, summary = clean_dataset(df)
            summary["source"] = name
            return clean_df, summary
        except Exception as error:  # try the next source
            errors.append(f"{name}: {error}")

    raise RuntimeError("Could not load the dataset from any source.\n" + "\n".join(errors))


# --------------------------------------------------------------------------
# 2. Model training and evaluation (cached - runs only once)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the model (first run only)...")
def train_model():
    """Train the TF-IDF + Naive Bayes pipeline and evaluate it on the held-out test set."""
    df, summary = load_dataset()

    messages = df["message"]
    labels = df["label"]

    # 80% training / 20% testing, reproducible and stratified
    x_train, x_test, y_train, y_test = train_test_split(
        messages, labels, test_size=0.20, random_state=42, stratify=labels
    )

    # The pipeline learns the vocabulary ONLY from training data (no test data leakage)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
            ("nb", MultinomialNB(alpha=0.1)),
        ]
    )
    pipeline.fit(x_train, y_train)

    # Evaluate on unseen test messages. 'spam' is treated as the positive class.
    y_pred = pipeline.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="spam"),
        "recall": recall_score(y_test, y_pred, pos_label="spam"),
        "f1": f1_score(y_test, y_pred, pos_label="spam"),
    }
    matrix = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "matrix": matrix,
        "df": df,
        "summary": summary,
        "n_train": len(x_train),
        "n_test": len(x_test),
    }


# --------------------------------------------------------------------------
# 3. Prediction logic
# --------------------------------------------------------------------------
def predict_message(pipeline, message):
    """Classify one SMS. Returns (label, spam_probability, ham_probability)."""
    classes = list(pipeline.classes_)
    probabilities = pipeline.predict_proba([message])[0]
    spam_prob = float(probabilities[classes.index("spam")])
    ham_prob = float(probabilities[classes.index("ham")])
    label = "spam" if spam_prob >= ham_prob else "ham"
    return label, spam_prob, ham_prob


def influential_words(pipeline, message, predicted_label, top_n=5):
    """Find the words in the message that pushed the model towards its decision."""
    tfidf = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["nb"]

    vector = tfidf.transform([message])
    word_positions = vector.nonzero()[1]
    if len(word_positions) == 0:
        return []

    classes = list(model.classes_)
    spam_log_prob = model.feature_log_prob_[classes.index("spam"), word_positions]
    ham_log_prob = model.feature_log_prob_[classes.index("ham"), word_positions]
    spam_lean = spam_log_prob - ham_log_prob  # positive = word is more typical of spam

    words = tfidf.get_feature_names_out()[word_positions]
    order = np.argsort(spam_lean)
    if predicted_label == "spam":
        order = order[::-1]

    selected = []
    for position in order[:top_n]:
        points_to_spam = spam_lean[position] > 0
        if (predicted_label == "spam") == points_to_spam:
            selected.append(words[position])
    return selected


# --------------------------------------------------------------------------
# 4. Charts
# --------------------------------------------------------------------------
def style_axes(fig, ax):
    """Apply the dark theme to a Matplotlib chart."""
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.tick_params(colors=COLOR_TEXT)
    ax.xaxis.label.set_color(COLOR_TEXT)
    ax.yaxis.label.set_color(COLOR_TEXT)
    ax.title.set_color(COLOR_TEXT)
    for side in ax.spines.values():
        side.set_color("#334155")


def confusion_matrix_figure(matrix):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    cmap = LinearSegmentedColormap.from_list("shield", ["#1E2A52", COLOR_PURPLE])
    labels = ["LEGITIMATE", "SPAM"]
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap=cmap, cbar=False, linewidths=2, linecolor="#0B1020",
        xticklabels=labels, yticklabels=labels, ax=ax,
        annot_kws={"size": 18, "weight": "bold", "color": "white"},
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion Matrix (test set)")
    style_axes(fig, ax)
    fig.tight_layout()
    return fig


def class_distribution_figure(n_ham, n_spam):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    bars = ax.bar(["LEGITIMATE", "SPAM"], [n_ham, n_spam], color=[COLOR_HAM, COLOR_SPAM], width=0.55)
    for bar, value in zip(bars, [n_ham, n_spam]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom",
                color=COLOR_TEXT, fontweight="bold")
    ax.set_ylabel("Number of messages")
    ax.set_xlabel("Message class")
    ax.set_title("Spam vs Legitimate Messages")
    ax.set_ylim(0, max(n_ham, n_spam) * 1.15)
    ax.grid(axis="y", alpha=0.15)
    style_axes(fig, ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 5. Page sections
# --------------------------------------------------------------------------
def render_header():
    render_html(
        """
        <div class="hero">
            <div class="hero-badge">🛡️ MACHINE LEARNING PROJECT</div>
            <h1 class="hero-title">SMS Shield</h1>
            <div class="hero-subtitle">Intelligent SMS Spam Detection</div>
            <div class="hero-text">
                Spam messages waste time and can trick people into sharing money or personal details.
                SMS Shield uses a machine learning model trained on thousands of real text messages to
                decide whether a message is <b>SPAM</b> or <b>LEGITIMATE</b>. Paste a message below and try it.
            </div>
        </div>
        """
    )


# Callbacks run BEFORE the page is redrawn, so they can safely change the text box value.
def use_sample(text):
    st.session_state["sms_input"] = text
    st.session_state["analyzed"] = False


def clear_input():
    st.session_state["sms_input"] = ""
    st.session_state["analyzed"] = False


def render_prediction_section(pipeline):
    render_html('<div class="section-title">Analyze a Message</div>'
                '<div class="section-sub">Type or paste an SMS, or start with one of the samples.</div>')

    st.markdown("**Try a sample message**")
    sample_columns = st.columns(len(SAMPLE_MESSAGES))
    for column, (title, text) in zip(sample_columns, SAMPLE_MESSAGES.items()):
        column.button(title, key=f"sample_{title}", on_click=use_sample, args=(text,))

    st.text_area(
        "SMS message",
        key="sms_input",
        height=150,
        placeholder="Type or paste an SMS message here...",
        label_visibility="collapsed",
    )

    button_columns = st.columns([3, 1])
    analyze_clicked = button_columns[0].button("🔍 Analyze Message", type="primary", key="analyze")
    button_columns[1].button("Clear / Try another", key="clear", on_click=clear_input)

    if analyze_clicked:
        st.session_state["analyzed"] = True

    if not st.session_state.get("analyzed"):
        return

    message = st.session_state.get("sms_input", "").strip()
    if not message:
        st.warning("Please enter an SMS message before clicking Analyze.")
        return

    label, spam_prob, ham_prob = predict_message(pipeline, message)
    words = influential_words(pipeline, message, label)
    confidence = spam_prob if label == "spam" else ham_prob

    if label == "spam":
        css_class, verdict_class = "result-spam", "verdict-spam"
        explanation = ("The model found wording that is common in spam messages "
                       "(for example offers, prizes, urgency or requests to call or text a number).")
        word_title = "Words that pushed the decision towards spam:"
    else:
        css_class, verdict_class = "result-ham", "verdict-ham"
        explanation = ("The wording looks like an ordinary personal or informational message, "
                       "similar to the legitimate messages the model saw during training.")
        word_title = "Words that pushed the decision towards legitimate:"

    chips = "".join(f'<span class="chip">{html.escape(str(word))}</span>' for word in words)
    if words:
        word_block = f'<div class="result-text" style="margin-top:0.6rem"><b>{word_title}</b><br>{chips}</div>'
    else:
        word_block = ('<div class="result-text" style="margin-top:0.6rem">None of the message words were '
                      'strongly linked to either class, so the model relied mostly on overall class frequency.</div>')

    render_html(
        f"""
        <div class="result-card {css_class}">
            <div class="result-label">Prediction</div>
            <div class="result-verdict {verdict_class}">{DISPLAY_NAMES[label]}</div>
            <div class="result-text">Model confidence: <b>{confidence * 100:.1f}%</b>
                (spam probability {spam_prob * 100:.1f}% | legitimate probability {ham_prob * 100:.1f}%)</div>
            <div class="result-text" style="margin-top:0.5rem">{explanation}</div>
            {word_block}
            <div class="disclaimer">⚠️ This result is generated by a machine learning model.
                It is a statistical estimate, not a certainty, and may occasionally be wrong.</div>
        </div>
        """
    )
    st.progress(spam_prob, text=f"Spam probability: {spam_prob * 100:.1f}%")


def metric_card(label, value):
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>'


def render_performance_section(result):
    metrics = result["metrics"]
    matrix = result["matrix"]

    render_html(
        f'<div class="section-title">Model Performance</div>'
        f'<div class="section-sub">Measured on {result["n_test"]:,} test messages the model never saw during training '
        f'(trained on {result["n_train"]:,} messages). Precision, recall and F1 are calculated for the SPAM class.</div>'
    )

    columns = st.columns(4)
    cards = [("Accuracy", metrics["accuracy"]), ("Precision", metrics["precision"]),
             ("Recall", metrics["recall"]), ("F1-Score", metrics["f1"])]
    for column, (name, value) in zip(columns, cards):
        column.markdown(metric_card(name, f"{value * 100:.2f}%"), unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        st.pyplot(confusion_matrix_figure(matrix))
    with right:
        true_neg, false_pos, false_neg, true_pos = matrix.ravel()
        render_html(
            f"""
            <div class="card">
                <h4>Reading the confusion matrix</h4>
                <ul>
                    <li><b>{true_neg:,}</b> legitimate messages correctly kept as LEGITIMATE (true negatives)</li>
                    <li><b>{false_pos:,}</b> legitimate messages wrongly marked as SPAM (false positives)</li>
                    <li><b>{false_neg:,}</b> spam messages that slipped through as LEGITIMATE (false negatives)</li>
                    <li><b>{true_pos:,}</b> spam messages correctly caught as SPAM (true positives)</li>
                </ul>
            </div>
            """
        )

    first, second = st.columns(2)
    with first:
        render_html(
            """
            <div class="card"><h4>Accuracy</h4>
            <p>Out of all test messages, the share that were classified correctly. Because most messages are
            legitimate, accuracy alone can look good even for a weak spam detector.</p></div>
            <div class="card"><h4>Precision</h4>
            <p>Of all messages the model called SPAM, the share that really were spam. High precision means
            few genuine messages are wrongly blocked.</p></div>
            """
        )
    with second:
        render_html(
            """
            <div class="card"><h4>Recall</h4>
            <p>Of all the real spam messages, the share the model managed to catch. High recall means
            little spam reaches the user.</p></div>
            <div class="card"><h4>F1-Score</h4>
            <p>A single number that balances precision and recall (their harmonic mean). It is high only when
            both are high.</p></div>
            """
        )


def render_dataset_section(result):
    df = result["df"]
    summary = result["summary"]
    n_total = len(df)
    n_spam = int((df["label"] == "spam").sum())
    n_ham = n_total - n_spam

    render_html(
        f'<div class="section-title">Dataset Overview</div>'
        f'<div class="section-sub">UCI SMS Spam Collection, loaded automatically from the {html.escape(summary["source"])}.</div>'
    )

    columns = st.columns(3)
    columns[0].markdown(metric_card("Total Messages", f"{n_total:,}"), unsafe_allow_html=True)
    columns[1].markdown(metric_card("Spam Messages", f"{n_spam:,}"), unsafe_allow_html=True)
    columns[2].markdown(metric_card("Legitimate Messages", f"{n_ham:,}"), unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.pyplot(class_distribution_figure(n_ham, n_spam))
    with right:
        render_html(
            f"""
            <div class="card">
                <h4>Class distribution</h4>
                <p>About <b>{n_spam / n_total * 100:.1f}%</b> of the messages are spam and
                <b>{n_ham / n_total * 100:.1f}%</b> are legitimate. The classes are imbalanced, which is why
                precision, recall and F1-score matter more than accuracy alone. The train/test split is
                stratified so both sets keep this same ratio.</p>
                <h4>Data cleaning summary</h4>
                <ul>
                    <li>Raw rows read: <b>{summary["rows_raw"]:,}</b></li>
                    <li>Invalid or empty rows removed: <b>{summary["invalid_removed"]:,}</b></li>
                    <li>Duplicate messages removed: <b>{summary["duplicates_removed"]:,}</b></li>
                    <li>Rows used: <b>{summary["rows_clean"]:,}</b></li>
                </ul>
            </div>
            """
        )

    with st.expander("Preview a few messages from the dataset"):
        preview = df.sample(10, random_state=1).copy()
        preview["label"] = preview["label"].map(DISPLAY_NAMES)
        preview.columns = ["Class", "Message"]
        st.dataframe(preview, hide_index=True)


def render_how_it_works_section():
    render_html('<div class="section-title">How It Works</div>'
                '<div class="section-sub">From raw text messages to a prediction in five steps.</div>')

    steps = [
        ("1", "Dataset", "5,000+ labelled SMS messages (spam or ham) are downloaded."),
        ("2", "Text Vectorization", "TF-IDF turns every message into numbers."),
        ("3", "Model Training", "Naive Bayes learns which words belong to spam."),
        ("4", "Model Evaluation", "The model is tested on unseen messages."),
        ("5", "SMS Prediction", "A new message is converted and classified."),
    ]
    columns = st.columns(5)
    for column, (number, title, text) in zip(columns, steps):
        column.markdown(
            f'<div class="step-card"><div class="step-num">{number}</div>'
            f'<div class="step-title">{title}</div><div class="step-text">{text}</div></div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns(2)
    with left:
        render_html(
            """
            <div class="card">
                <h4>What is TF-IDF?</h4>
                <p>Computers cannot read words, so we convert each message into numbers. <b>TF-IDF</b> gives every
                word a score. <b>TF</b> (term frequency) rewards words that appear often in a message.
                <b>IDF</b> (inverse document frequency) lowers the score of words that appear in almost every
                message, such as "the" or "you", because they say little. Rare but meaningful words like
                "prize" or "claim" get high scores.</p>
            </div>
            """
        )
    with right:
        render_html(
            """
            <div class="card">
                <h4>What is Multinomial Naive Bayes?</h4>
                <p>It is a simple probability-based classifier. During training it learns how likely each word is
                in spam and in legitimate messages. For a new message it combines the word probabilities and
                chooses the class with the higher probability. It is called "naive" because it treats every word as
                independent of the others, which is not fully true but works very well for text.</p>
            </div>
            """
        )


def render_about_section():
    render_html('<div class="section-title">About the Project</div>'
                '<div class="section-sub">SMS Shield - SMS Spam Detection Using Machine Learning</div>')

    first, second = st.columns(2)
    with first:
        render_html(
            """
            <div class="card"><h4>Problem statement</h4>
            <p>Unwanted and fraudulent SMS messages annoy users and can lead to financial loss and identity theft.
            Writing manual rules to catch every spam message is impractical because spammers keep changing their
            wording. A system that learns from examples is needed.</p></div>
            <div class="card"><h4>Objectives</h4><ul>
                <li>Build a model that classifies an SMS as spam or legitimate.</li>
                <li>Evaluate it honestly using accuracy, precision, recall, F1-score and a confusion matrix.</li>
                <li>Provide an easy-to-use web interface and publish it online.</li>
            </ul></div>
            <div class="card"><h4>Technologies used</h4><ul>
                <li>Python, Pandas, NumPy</li>
                <li>Scikit-learn (TF-IDF, Multinomial Naive Bayes, Pipeline)</li>
                <li>Matplotlib and Seaborn for charts</li>
                <li>Streamlit for the web app, GitHub and Streamlit Community Cloud for hosting</li>
            </ul></div>
            """
        )
    with second:
        render_html(
            """
            <div class="card"><h4>Advantages</h4><ul>
                <li>Fast to train and fast to predict.</li>
                <li>Simple, easy to understand and explain.</li>
                <li>Works well on small datasets and text data.</li>
                <li>Free to run and deploy; no database or API keys needed.</li>
            </ul></div>
            <div class="card"><h4>Limitations</h4><ul>
                <li>The dataset is relatively small and mostly English.</li>
                <li>Bag-of-words ignores deeper meaning and sentence context.</li>
                <li>New spam styles that use unseen words can be missed.</li>
                <li>Predictions are probabilistic and can be wrong.</li>
            </ul></div>
            <div class="card"><h4>Future enhancements</h4><ul>
                <li>Compare other models such as Logistic Regression, SVM or deep learning.</li>
                <li>Train on larger and multilingual datasets.</li>
                <li>Add user feedback to keep improving the model.</li>
                <li>Create a mobile app or SMS-filter integration.</li>
            </ul></div>
            """
        )


# --------------------------------------------------------------------------
# 6. Main program
# --------------------------------------------------------------------------
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_header()

    # Load the dataset and train the model (cached, so this happens only once)
    try:
        result = train_model()
    except Exception as error:
        st.error("SMS Shield could not load the dataset or train the model.")
        st.code(str(error))
        st.info("Please check your internet connection and try again.")
        if st.button("Retry"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        st.stop()

    tab_predict, tab_performance, tab_dataset, tab_how, tab_about = st.tabs(
        ["🔍 Analyze SMS", "📊 Model Performance", "🗂️ Dataset", "⚙️ How It Works", "📘 About"]
    )
    with tab_predict:
        render_prediction_section(result["pipeline"])
    with tab_performance:
        render_performance_section(result)
    with tab_dataset:
        render_dataset_section(result)
    with tab_how:
        render_how_it_works_section()
    with tab_about:
        render_about_section()

    render_html('<div class="footer">SMS Shield · B.Tech CSE ML Microproject · Built with Python, '
                'scikit-learn and Streamlit</div>')


main()
