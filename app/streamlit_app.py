import streamlit as st

from inference import load_artifacts, predict_sentiment


st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at top left, rgba(180, 90, 255, 0.22), transparent 30%),
        radial-gradient(circle at top right, rgba(255, 80, 180, 0.18), transparent 28%),
        linear-gradient(135deg, #0b0615 0%, #140a24 45%, #1b1033 100%);
    color: #f5edff;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 900px;
}

.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.35rem;
    color: #ffffff;
}

.sub-text {
    color: #d7c8f3;
    margin-bottom: 1.5rem;
    font-size: 1rem;
}

.glass-card {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.14);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-radius: 24px;
    padding: 1.4rem 1.4rem 1.2rem 1.4rem;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.28);
    margin-bottom: 1rem;
}

.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.07) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 16px !important;
}

.stSelectbox label,
.stTextArea label {
    color: #eadcff !important;
    font-weight: 600 !important;
}

.stButton > button {
    width: 100%;
    border: 0;
    border-radius: 16px;
    padding: 0.85rem 1rem;
    font-weight: 700;
    color: white;
    background: linear-gradient(90deg, #b84dff 0%, #ff4db8 100%);
    box-shadow: 0 8px 24px rgba(216, 73, 255, 0.35);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 30px rgba(216, 73, 255, 0.45);
}

.result-box {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 20px;
    padding: 1rem 1rem 0.75rem 1rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.result-label {
    font-size: 0.9rem;
    color: #d6c6f6;
    margin-bottom: 0.35rem;
}

.result-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: white;
}

.source-badge {
    display: inline-block;
    margin-top: 0.65rem;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background: rgba(184, 77, 255, 0.16);
    color: #f2ddff;
    border: 1px solid rgba(255, 255, 255, 0.12);
    font-size: 0.85rem;
    font-weight: 600;
}

.score-title {
    margin-top: 0.25rem;
    margin-bottom: 1rem;
    color: #ffffff;
    font-size: 1.15rem;
    font-weight: 700;
}

.stat-card {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 20px;
    padding: 1rem;
    min-height: 130px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.18);
}

.stat-label {
    color: #d6c6f6;
    font-size: 0.9rem;
    margin-bottom: 0.35rem;
}

.stat-value {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.6rem;
}

.stat-bar-wrap {
    width: 100%;
    height: 10px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.stat-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #b84dff 0%, #ff4db8 100%);
}

.stat-note {
    margin-top: 0.55rem;
    color: #f0e6ff;
    font-size: 0.85rem;
}

[data-testid="stAlert"] {
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
}
</style>
""", unsafe_allow_html=True)


def render_stat_card(label: str, value: float):
    percentage = max(0.0, min(value, 1.0)) * 100
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{percentage:.0f}%</div>
            <div class="stat-bar-wrap">
                <div class="stat-bar-fill" style="width: {percentage:.2f}%;"></div>
            </div>
            <div class="stat-note">{value:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown('<div class="main-title">Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Analyze reviews with a hybrid pipeline: rules, machine learning, and Groq fallback.</div>',
    unsafe_allow_html=True
)

with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    mode = st.selectbox(
        "Choose prediction mode",
        [
            "Auto (Rules + ML + Groq)",
            "ML only",
            "Groq only"
        ]
    )

    mode_map = {
        "Auto (Rules + ML + Groq)": "auto",
        "ML only": "ml",
        "Groq only": "groq"
    }

    user_text = st.text_area(
        "Enter text here",
        placeholder="Example: The product is okay, nothing special"
    )

    analyze = st.button("Analyze Sentiment")

    st.markdown('</div>', unsafe_allow_html=True)

if analyze:
    if user_text.strip():
        model, vectorizer = load_artifacts()
        result = predict_sentiment(
            user_text,
            model,
            vectorizer,
            mode=mode_map[mode]
        )

        prediction = result["prediction"]
        source = result["source"]
        scores = result["scores"]

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">Prediction</div>
                <div class="result-value">{prediction}</div>
                <div class="source-badge">Decision source: {source}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        negative_score = float(scores.get("negative", 0.0))
        neutral_score = float(scores.get("neutral", 0.0))
        positive_score = float(scores.get("positive", 0.0))

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="score-title">Confidence Distribution</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            render_stat_card("Negative", negative_score)
        with col2:
            render_stat_card("Neutral", neutral_score)
        with col3:
            render_stat_card("Positive", positive_score)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("Please enter some text first.")