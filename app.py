from pathlib import Path
import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.joblib"

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
)

st.title("📩 SMS Spam Detector")
st.write("Enter a message to predict whether it is spam.")

if not MODEL_PATH.exists():
    st.error("Model file missing. Run train.py first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
message = st.text_area("Message", placeholder = "Type a message here..."
                       , height = 150,)

if st.button("Check message"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0]

        if prediction == "spam":
            st.error("Prediction: spam")
        else:
            st.success("Prediction: Not spam")

st.caption("Learning project trained on SMS examples. "
           "Predictions can be incorrect. ")
