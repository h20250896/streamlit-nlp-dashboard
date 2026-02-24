import streamlit as st
import joblib
import numpy as np
import re
from scipy.sparse import hstack
import os
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Review Helpfulness Analyzer",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File paths - Compatible with Streamlit Cloud
# Place model files in the same directory as this script
MODEL_PATH = Path(__file__).parent
MODEL_FILE = MODEL_PATH / "helpfulness_model.pkl"
VECTORIZER_FILE = MODEL_PATH / "helpfulness_vectorizer.pkl"
MODEL_PATH = Path(r"C:\Users\ankit\Downloads")
MODEL_FILE = MODEL_PATH / "helpfulness_model.pkl"
VECTORIZER_FILE = MODEL_PATH / "helpfulness_vectorizer.pkl"

# ============================================================
# LOAD MODEL WITH ERROR HANDLING
# ============================================================

@st.cache_resource
def load_model():
    """Load model and vectorizer with error handling."""
    try:
        if not MODEL_FILE.exists() or not VECTORIZER_FILE.exists():
            st.error(f"❌ Files not found in {MODEL_PATH}")
            return None, None
        
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, vectorizer = load_model()

# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):
    """Clean and preprocess text."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_helpfulness(review_text, rating):
    """Predict review helpfulness."""
    if not review_text.strip():
        return None, None, "Empty review"
    
    try:
        cleaned = preprocess_text(review_text)
        tfidf_feat = vectorizer.transform([cleaned])
        review_length = len(cleaned.split())
        additional_feat = np.array([[review_length, rating]])
        features = hstack([tfidf_feat, additional_feat])
        
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0]
        confidence = prob[prediction]
        
        return prediction, confidence, None
    except Exception as e:
        return None, None, f"Prediction error: {str(e)}"

# ============================================================
# HEADER SECTION
# ============================================================

col_title, col_icon = st.columns([0.9, 0.1])
with col_title:
    st.title("⭐ Review Helpfulness Prediction System")
with col_icon:
    st.success("🟢 Model Loaded")

st.markdown("""
---
**Predict whether a customer review will be considered helpful by other users.**

### 📊 Business Applications:
- 📈 Rank useful reviews at the top
- 🎯 Improve customer decision-making
- 🤝 Increase platform trust and engagement
""")

if model is None:
    st.error("⚠️ Model failed to load. Please check the file paths.")
    st.stop()

# ============================================================
# MAIN INPUT SECTION
# ============================================================

st.markdown("---")
st.subheader("🔍 Analysis Dashboard")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    review_text = st.text_area(
        "📝 Enter Customer Review",
        height=150,
        placeholder="Example: This product exceeded my expectations. The quality is excellent and delivery was fast..."
    )

with col2:
    rating = st.slider("⭐ Product Rating", 1, 5, 4)
    review_length = len(review_text.split()) if review_text else 0
    st.metric("Word Count", review_length)

with col3:
    st.write("**Settings**")
    confidence_threshold = st.slider("Min Confidence", 0.5, 1.0, 0.7)
    st.write("")

analyze_button = st.button("🚀 Analyze Helpfulness", use_container_width=True)

# ============================================================
# PREDICTION & RESULTS
# ============================================================

if analyze_button:
    prediction, confidence, error = predict_helpfulness(review_text, rating)
    
    if error:
        st.error(f"❌ {error}")
    else:
        st.markdown("---")
        
        col_result, col_stats = st.columns([1, 1])
        
        with col_result:
            if prediction == 1:
                st.success("✅ HELPFUL REVIEW", icon="✅")
                st.write(f"**Confidence:** {confidence:.1%}")
                st.progress(confidence)
            else:
                st.error("⚠️ NOT HELPFUL REVIEW", icon="⚠️")
                st.write(f"**Confidence:** {confidence:.1%}")
                st.progress(confidence)
        
        with col_stats:
            st.write("**Review Statistics:**")
            st.write(f"- Word Count: {review_length}")
            st.write(f"- Rating: {rating}/5 ⭐")
            st.write(f"- Text Length: {len(review_text)} chars")
        
        st.markdown("---")
        st.subheader("💡 Business Insights")
        
        if prediction == 1:
            st.info("""
            **Positive Indicators Detected:**
            - ✓ Detailed and informative content
            - ✓ Likely to be upvoted by users
            - ✓ Recommended for top-tier display
            - ✓ High engagement potential
            """)
        else:
            st.warning("""
            **Areas for Improvement:**
            - ⚠ Review may be too brief or generic
            - ⚠ Limited informational value
            - ⚠ Consider adding more details
            - ⚠ Not suitable for top ranking
            """)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📋 Model Information")
st.sidebar.markdown("""
**Algorithm Details:**
- Model Type: Logistic Regression
- Training Data: Amazon Fine Food Reviews
- Total Features: 5,002
  - TF-IDF: 5,000
  - Review Length: 1
  - Rating Score: 1

**Objective:**
Predict if a review will receive helpful votes from users
""")

st.sidebar.divider()
st.sidebar.header("🧪 Quick Test Examples")

examples = [
    ("This product is excellent and arrived quickly. The quality exceeded my expectations and I would recommend it to anyone looking for this type of item.", 5),
    ("Good product", 4),
    ("Terrible packaging and stopped working after one week. Waste of money.", 1),
    ("Exactly as described on the website perfect fit and great quality", 5),
]

for idx, (text, score) in enumerate(examples):
    if st.sidebar.button(f"📌 Test {idx + 1}: {text[:40]}...", key=f"test_{idx}"):
        pred, conf, _ = predict_helpfulness(text, score)
        if pred == 1:
            st.sidebar.success(f"✅ HELPFUL (Confidence: {conf:.1%})")
        else:
            st.sidebar.warning(f"⚠️ NOT HELPFUL (Confidence: {conf:.1%})")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("🏢 NLP Industry Project | Review Helpfulness Prediction System v1.0")
st.caption("Last Updated: 2024 | Built with Streamlit")