import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
from groq import Groq


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SocialSense AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
/* ---------- Main background ---------- */
.stApp {
    background: #0b0d12;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #151820;
    border-right: 1px solid #272b36;
}

.sidebar-title {
    font-size: 25px;
    font-weight: 800;
    margin-bottom: 4px;
}

.sidebar-subtitle {
    color: #9ca3af;
    font-size: 14px;
    line-height: 1.6;
}

.sidebar-box {
    background: #1b1f29;
    border: 1px solid #2b303c;
    border-radius: 14px;
    padding: 16px;
    margin-top: 18px;
}

.sidebar-item {
    color: #d1d5db;
    font-size: 14px;
    margin: 9px 0;
}

/* ---------- Hero ---------- */
.hero {
    padding: 8px 0 20px 0;
}

.hero h1 {
    font-size: clamp(34px, 5vw, 58px);
    line-height: 1.05;
    font-weight: 850;
    letter-spacing: -2px;
    margin-bottom: 12px;
    background: linear-gradient(90deg, #8b7cff, #e85aa9, #ff9f43);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #9ca3af;
    font-size: 17px;
    max-width: 760px;
    line-height: 1.6;
}

/* ---------- Section headings ---------- */
.section-title {
    font-size: 24px;
    font-weight: 750;
    margin: 26px 0 12px 0;
}

.section-caption {
    color: #8f96a3;
    font-size: 14px;
    margin-bottom: 14px;
}

/* ---------- Input area ---------- */
.input-card {
    background: #14171f;
    border: 1px solid #292e39;
    border-radius: 16px;
    padding: 18px;
    margin-top: 10px;
}

/* ---------- Metric cards ---------- */
.metric-card {
    background: linear-gradient(145deg, #171a23, #11141b);
    border: 1px solid #2a2f3b;
    border-radius: 16px;
    padding: 20px;
    min-height: 142px;
    transition: 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: #7064ff;
}

.metric-label {
    color: #9ca3af;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 12px;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 8px;
}

.metric-confidence {
    color: #aeb4c0;
    font-size: 13px;
}

/* ---------- Status pills ---------- */
.pill {
    display: inline-block;
    padding: 5px 11px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
}

.positive {
    background: #143b2a;
    color: #6ee7a7;
}

.negative {
    background: #421d23;
    color: #ff929d;
}

.joy {
    background: #453a16;
    color: #ffe27a;
}

.anger {
    background: #461c20;
    color: #ff929d;
}

.optimism {
    background: #183b2b;
    color: #72e6a6;
}

.sadness {
    background: #192f49;
    color: #8bc5ff;
}

.neutral {
    background: #292d36;
    color: #c5cad3;
}

/* ---------- Aspect cards ---------- */
.aspect-card {
    background: #151922;
    border: 1px solid #2a2f3a;
    border-radius: 12px;
    padding: 13px 16px;
    margin: 8px 0;
}

.aspect-name {
    font-weight: 700;
}

.aspect-meta {
    color: #9ca3af;
    font-size: 13px;
}

/* ---------- AI insight ---------- */
.ai-card {
    background: linear-gradient(145deg, #15263b, #142033);
    border: 1px solid #29425f;
    border-radius: 16px;
    padding: 20px;
    color: #d9e8ff;
    line-height: 1.7;
}

/* ---------- Info / empty cards ---------- */
.info-card {
    background: #151922;
    border: 1px solid #292f3a;
    border-radius: 14px;
    padding: 16px;
    color: #b8bec9;
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 10px;
    border: 1px solid #5750d9;
    background: #171923;
    color: #f3f4f6;
    font-weight: 650;
    min-height: 42px;
    transition: 0.2s ease;
}

.stButton > button:hover {
    border-color: #9b91ff;
    background: #24203b;
    transform: translateY(-1px);
}

/* Primary analyze button */
button[kind="primary"] {
    background: linear-gradient(90deg, #6257e8, #a44fba) !important;
    border: none !important;
    color: white !important;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #292e38;
}

.stTabs [data-baseweb="tab"] {
    padding: 12px 18px;
    font-weight: 650;
}

.stTabs [aria-selected="true"] {
    color: #e86aaa !important;
}

/* ---------- Dataframe ---------- */
div[data-testid="stDataFrame"] {
    border: 1px solid #2a2f3a;
    border-radius: 12px;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: #666d79;
    font-size: 12px;
    padding-top: 35px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# API KEY
# ============================================================
import os
import streamlit as st
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
    except Exception:
        GROQ_API_KEY = None

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    # Prefer the current .keras model, but support the older .h5 file too.
    if os.path.exists("models/bilstm_sentiment_model.keras"):
        sentiment_model_path = "models/bilstm_sentiment_model.keras"
    else:
        sentiment_model_path = "models/bilstm_sentiment_model.h5"

    sentiment_model = tf.keras.models.load_model(sentiment_model_path)

    with open("models/tokenizer.pkl", "rb") as f:
        sentiment_tokenizer = pickle.load(f)

    emo_model = tf.keras.models.load_model(
        "models/emotion_bilstm_model.keras"
    )

    with open("models/emo_tokenizer.pkl", "rb") as f:
        emo_tokenizer = pickle.load(f)

    with open("models/kmeans_model.pkl", "rb") as f:
        kmeans_model = pickle.load(f)

    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    nlp = spacy.load("en_core_web_sm")

    return (
        sentiment_model,
        sentiment_tokenizer,
        emo_model,
        emo_tokenizer,
        kmeans_model,
        tfidf_vectorizer,
        nlp
    )


(
    sentiment_model,
    sentiment_tokenizer,
    emo_model,
    emo_tokenizer,
    kmeans_model,
    tfidf_vectorizer,
    nlp
) = load_models()


MAX_LEN = 50

emotion_names = [
    "anger",
    "joy",
    "optimism",
    "sadness"
]

topic_labels = {
    0: "Celebrations & Special Occasions",
    1: "Evening / Emotional",
    2: "Work Routine",
    3: "Miscellaneous",
    4: "#FollowFriday Culture",
    5: "Music & Entertainment"
}


# ============================================================
# PREPROCESSING
# ============================================================
def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# SENTIMENT
# ============================================================
def predict_sentiment(text):
    cleaned = clean_tweet(text)

    seq = sentiment_tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    pred = float(sentiment_model.predict(padded, verbose=0)[0][0])

    label = "Positive" if pred >= 0.5 else "Negative"
    confidence = pred if pred >= 0.5 else 1 - pred

    return label, float(confidence)


# ============================================================
# EMOTION
# ============================================================
def predict_emotion(text):
    cleaned = clean_tweet(text)

    seq = emo_tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    pred = emo_model.predict(padded, verbose=0)[0]

    idx = int(np.argmax(pred))

    return emotion_names[idx], float(pred[idx])


# ============================================================
# TOPIC
# ============================================================
def predict_topic(text):
    cleaned = clean_tweet(text)

    vec = tfidf_vectorizer.transform([cleaned])
    topic_id = int(kmeans_model.predict(vec)[0])

    return topic_labels.get(topic_id, f"Topic {topic_id}")


# ============================================================
# ASPECT EXTRACTION
# ============================================================
def get_local_context(doc, start, end, window=4):
    context_start = max(0, start - window)
    context_end = min(len(doc), end + window)

    for i in range(context_start, context_end):
        token = doc[i]

        if token.text.lower() in [
            "but",
            "however",
            "although",
            "though"
        ]:
            if i < start:
                context_start = i + 1
            elif i >= end:
                context_end = i
                break

    return doc[context_start:context_end].text


def get_aspect_sentiments(text):
    doc = nlp(text)
    results = []

    for chunk in doc.noun_chunks:
        aspect = " ".join(
            t.text
            for t in chunk
            if not t.is_stop and not t.is_punct
        ).strip()

        if not aspect or len(aspect) <= 1:
            continue

        local_context = get_local_context(
            doc,
            chunk.start,
            chunk.end,
            window=4
        )

        label, confidence = predict_sentiment(local_context)

        results.append({
            "aspect": aspect,
            "sentiment": label,
            "confidence": confidence
        })

    return results


# ============================================================
# GROQ AI INSIGHT
# ============================================================
def generate_ai_insight(sentiment, emotion, topic, aspects):
    aspect_summary = (
        ", ".join(
            f"{a['aspect']} ({a['sentiment']}, {a['confidence']:.0%})"
            for a in aspects
        )
        if aspects
        else "none detected"
    )

    prompt = f"""
You are a social media analytics assistant.

Explain these model results in 2-3 simple sentences.

Sentiment: {sentiment}
Emotion: {emotion}
Topic: {topic}
Aspects: {aspect_summary}

Rules:
- Use ONLY the information provided.
- Do not invent statistics, facts, causes, or user intentions.
- Do not treat the topic label as proof of the user's situation.
- Do not change the model predictions.
- Keep the explanation factual and concise.
"""

    if groq_client is None:
        return (
            "AI insight is unavailable because the Groq API key is not configured. "
            "The model predictions above are still available."
        )

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error(f"Debug - AI insight error: {type(e).__name__}: {str(e)}")
        return (
            "AI insight is unavailable right now. "
            "The analysis above is still valid."
        )


# ============================================================
# HELPER FUNCTIONS FOR UI
# ============================================================
def sentiment_class(label):
    return "positive" if label == "Positive" else "negative"


def emotion_class(label):
    return label.lower()


def show_metric_card(title, value, confidence=None, css_class="neutral"):
    confidence_html = (
        f'<div class="metric-confidence">'
        f'<span class="pill {css_class}">{confidence:.0%} confidence</span>'
        f'</div>'
        if confidence is not None
        else ""
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            {confidence_html}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">📊 SocialSense AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'AI-powered social media intelligence platform'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-box">
            <b>What it analyzes</b>
            <div class="sidebar-item">✓ Sentiment</div>
            <div class="sidebar-item">✓ Emotion</div>
            <div class="sidebar-item">✓ Topics</div>
            <div class="sidebar-item">✓ Aspect sentiment</div>
            <div class="sidebar-item">✓ AI-generated insights</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-box">
            <b>Technology</b>
            <div class="sidebar-item">BiLSTM Deep Learning</div>
            <div class="sidebar-item">TF-IDF + K-Means</div>
            <div class="sidebar-item">spaCy NLP</div>
            <div class="sidebar-item">Groq LLM</div>
            <div class="sidebar-item">Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("B.Tech CSE · Generative AI")


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>AI-Powered Social Media Analytics</h1>
        <p>
            Understand what people feel, discuss, and say through
            sentiment, emotion, topic, and aspect-level analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================
tab1, tab2 = st.tabs([
    "🔎 Single Post",
    "📁 CSV Analysis"
])


# ============================================================
# SINGLE POST
# ============================================================
with tab1:

    st.markdown(
        '<div class="section-title">Analyze a social media post</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Enter a post below or try one of the examples.'
        '</div>',
        unsafe_allow_html=True
    )

    example_posts = {
        "📷 Mixed Review":
            "the camera is amazing but the battery life is terrible",

        "🎉 Celebration":
            "happy birthday to my best friend! love you so much",

        "😤 Complaint":
            "so frustrated with this app, it keeps crashing"
    }

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    c1, c2, c3 = st.columns(3)

    if c1.button("📷 Mixed Review", use_container_width=True):
        st.session_state.user_input = example_posts["📷 Mixed Review"]

    if c2.button("🎉 Celebration", use_container_width=True):
        st.session_state.user_input = example_posts["🎉 Celebration"]

    if c3.button("😤 Complaint", use_container_width=True):
        st.session_state.user_input = example_posts["😤 Complaint"]

    user_input = st.text_area(
        "Social media post",
        key="user_input",
        height=130,
        placeholder="Example: The camera is amazing but the battery life is terrible..."
    )

    analyze_clicked = st.button(
        "✨ Analyze Post",
        type="primary",
        use_container_width=True
    )

    if analyze_clicked:

        if not user_input.strip():
            st.warning("Please enter a social media post first.")

        else:
            with st.spinner("Running sentiment, emotion, topic and aspect analysis..."):

                sentiment_label, sentiment_conf = predict_sentiment(user_input)
                emotion_label, emotion_conf = predict_emotion(user_input)
                topic_label = predict_topic(user_input)
                aspects = get_aspect_sentiments(user_input)

                ai_insight = generate_ai_insight(
                    sentiment_label,
                    emotion_label,
                    topic_label,
                    aspects
                )

            st.markdown(
                '<div class="section-title">Analysis Results</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                show_metric_card(
                    "Sentiment",
                    sentiment_label,
                    sentiment_conf,
                    sentiment_class(sentiment_label)
                )

            with col2:
                show_metric_card(
                    "Emotion",
                    emotion_label.capitalize(),
                    emotion_conf,
                    emotion_class(emotion_label)
                )

            with col3:
                show_metric_card(
                    "Topic",
                    topic_label
                )

            # Aspect analysis
            st.markdown(
                '<div class="section-title">Aspect-Level Analysis</div>',
                unsafe_allow_html=True
            )

            if aspects:
                for a in aspects:
                    css = sentiment_class(a["sentiment"])

                    st.markdown(
                        f"""
                        <div class="aspect-card">
                            <span class="aspect-name">{a['aspect']}</span>
                            &nbsp;→&nbsp;
                            <span class="pill {css}">
                                {a['sentiment']}
                            </span>
                            <span class="aspect-meta">
                                &nbsp; {a['confidence']:.0%} confidence
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    '<div class="info-card">'
                    'No clear aspect was detected in this post.'
                    '</div>',
                    unsafe_allow_html=True
                )

            # AI insight
            st.markdown(
                '<div class="section-title">🤖 AI-Generated Insight</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="ai-card">{ai_insight}</div>',
                unsafe_allow_html=True
            )


# ============================================================
# CSV ANALYSIS
# ============================================================
with tab2:

    st.markdown(
        '<div class="section-title">Analyze a CSV dataset</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Upload a CSV containing social media posts and select the text column.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv"
    )

    if uploaded_file is not None:

        try:
            df_upload = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(
                f"Could not read this CSV file. "
                f"Please check the format. ({type(e).__name__})"
            )
            st.stop()

        if df_upload.empty:
            st.warning("The uploaded CSV is empty.")
            st.stop()

        st.markdown("### Dataset Preview")
        st.dataframe(
            df_upload.head(10),
            use_container_width=True,
            hide_index=True
        )

        text_column = st.selectbox(
            "Select the column containing post text",
            df_upload.columns
        )

        if st.button(
            "✨ Analyze CSV",
            type="primary",
            use_container_width=True,
            key="csv_analyze"
        ):

            valid_df = df_upload[
                df_upload[text_column].notna()
                & (
                    df_upload[text_column]
                    .astype(str)
                    .str.strip()
                    != ""
                )
            ].copy()

            skipped = len(df_upload) - len(valid_df)

            if valid_df.empty:
                st.warning("No valid text was found in the selected column.")
                st.stop()

            if skipped > 0:
                st.info(f"Skipped {skipped} empty row(s).")

            with st.spinner(
                f"Analyzing {len(valid_df)} social media posts..."
            ):

                results = []
                progress_bar = st.progress(0)

                for i, text in enumerate(
                    valid_df[text_column].astype(str)
                ):
                    try:
                        sentiment_label, sentiment_conf = predict_sentiment(text)
                        emotion_label, emotion_conf = predict_emotion(text)
                        topic_label = predict_topic(text)

                        results.append({
                            "text": text,
                            "sentiment": sentiment_label,
                            "sentiment_confidence": sentiment_conf,
                            "emotion": emotion_label,
                            "emotion_confidence": emotion_conf,
                            "topic": topic_label
                        })

                    except Exception:
                        results.append({
                            "text": text,
                            "sentiment": "Error",
                            "sentiment_confidence": 0,
                            "emotion": "Error",
                            "emotion_confidence": 0,
                            "topic": "Error"
                        })

                    progress_bar.progress(
                        (i + 1) / len(valid_df)
                    )

                progress_bar.empty()
                results_df = pd.DataFrame(results)

            st.success(f"Successfully analyzed {len(results_df)} posts!")

            # --------------------------------------------------------
            # Summary
            # --------------------------------------------------------
            st.markdown(
                '<div class="section-title">Dataset Summary</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            positive_pct = (
                results_df["sentiment"] == "Positive"
            ).mean()

            top_emotion = results_df["emotion"].mode()[0]
            top_topic = results_df["topic"].mode()[0]

            with col1:
                show_metric_card(
                    "Positive Sentiment",
                    f"{positive_pct:.0%}"
                )

            with col2:
                show_metric_card(
                    "Most Common Emotion",
                    top_emotion.capitalize()
                )

            with col3:
                show_metric_card(
                    "Most Common Topic",
                    top_topic
                )

            # --------------------------------------------------------
            # Charts
            # --------------------------------------------------------
            st.markdown(
                '<div class="section-title">Distributions</div>',
                unsafe_allow_html=True
            )

            chart1, chart2 = st.columns(2)

            with chart1:
                st.markdown("**Sentiment Distribution**")
                st.bar_chart(
                    results_df["sentiment"].value_counts()
                )

            with chart2:
                st.markdown("**Emotion Distribution**")
                st.bar_chart(
                    results_df["emotion"].value_counts()
                )

            st.markdown("**Topic Distribution**")
            st.bar_chart(
                results_df["topic"].value_counts()
            )

            # --------------------------------------------------------
            # Groq batch insight
            # --------------------------------------------------------
            sentiment_stats = (
                results_df["sentiment"]
                .value_counts(normalize=True)
                .round(3)
                .to_dict()
            )

            emotion_stats = (
                results_df["emotion"]
                .value_counts(normalize=True)
                .round(3)
                .to_dict()
            )

            top_topics = (
                results_df["topic"]
                .value_counts()
                .head(3)
                .index
                .tolist()
            )

            batch_prompt = f"""
You are a social media analytics assistant.

Analyze ONLY these model-generated results:

Sentiment distribution:
{sentiment_stats}

Emotion distribution:
{emotion_stats}

Top topics:
{top_topics}

Give:
1. Three short factual findings.
2. One concise overall summary.

Rules:
- Do not invent statistics.
- Do not invent causes or user intentions.
- Use only the supplied results.
- Keep each finding under 20 words.
"""

            st.markdown(
                '<div class="section-title">🤖 AI-Generated Batch Insight</div>',
                unsafe_allow_html=True
            )

            if groq_client is None:
                st.warning(
                    "Groq API key is not configured. "
                    "The charts and analysis are still available."
                )
            else:
                try:
                    batch_response = groq_client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[
                            {
                                "role": "user",
                                "content": batch_prompt
                            }
                        ]
                    )

                    st.markdown(
                        f'<div class="ai-card">'
                        f'{batch_response.choices[0].message.content}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                except Exception:
                    st.warning(
                        "AI batch insight is unavailable right now. "
                        "The charts and results are still valid."
                    )

            # --------------------------------------------------------
            # Results table
            # --------------------------------------------------------
            st.markdown(
                '<div class="section-title">Detailed Results</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

            # --------------------------------------------------------
            # Download
            # --------------------------------------------------------
            csv_output = results_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Results as CSV",
                data=csv_output,
                file_name="social_media_analysis_results.csv",
                mime="text/csv",
                use_container_width=True
            )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        SocialSense AI · Sentiment · Emotion · Topics · Aspect Analysis · AI Insights
    </div>
    """,
    unsafe_allow_html=True
)
