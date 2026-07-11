import streamlit as st
import requests

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon=None,
    layout="centered",
)

# ── Minimal styling (only lightweight overrides) ─────────────
st.markdown("""
<style>
    .stButton > button {
        background-color: #1a1a1a;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 0.55rem 1.75rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #333333;
        color: #ffffff;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ── Backend URL ──────────────────────────────────────────────
API_URL = "http://13.51.70.238:8001/predict"

# ── Header ───────────────────────────────────────────────────
st.title("Sentiment Analysis")
st.caption("Analyse the sentiment of movie reviews using a fine-tuned DistilBERT model.")
st.write("")

# ── Input ────────────────────────────────────────────────────
text_input = st.text_area(
    label="Review text",
    placeholder="Enter a movie review here...",
    height=140,
    label_visibility="collapsed",
)

analyse_clicked = st.button("Analyse")

# ── Prediction ───────────────────────────────────────────────
if analyse_clicked:
    if not text_input or not text_input.strip():
        st.error("Please enter some text before analysing.")
    else:
        with st.spinner("Analysing..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"text": text_input.strip()},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                predicted = data["predicted_result"]
                confidence = data["confidence"]
                probs = data["class_probabilities"]

                pos_pct = probs.get("Positive", 0)
                neg_pct = probs.get("Negative", 0)

                st.write("")
                st.divider()
                st.write("")

                # Prediction and confidence
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Prediction", value=predicted)
                with col2:
                    st.metric(label="Confidence", value=f"{confidence:.1%}")

                st.write("")

                # Probability bars
                st.caption("POSITIVE")
                st.progress(pos_pct, text=f"{pos_pct:.1%}")

                st.caption("NEGATIVE")
                st.progress(neg_pct, text=f"{neg_pct:.1%}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the backend. "
                    "Make sure the FastAPI server is running on port 8000."
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"Server returned an error: {e}")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ── Footer ───────────────────────────────────────────────────
st.write("")
st.divider()
st.caption("Built by Prabhsimrat Singh")
