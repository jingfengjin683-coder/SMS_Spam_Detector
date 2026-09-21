from pathlib import Path
import joblib
import streamlit as st
import pandas as pd

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
def explain_prediction(model, message):
    vectoriser = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    message_vector = vectoriser.transform([message])
    feature_names = vectoriser.get_feature_names_out()
    weights = classifier.coef_[0]
    direction = 1 if classifier.classes_[1] == "spam" else -1
    rows = []
    for index, value in zip(message_vector.indices, message_vector.data):
        contribution = value * weights[index] * direction
        rows.append({"Word or phrase": feature_names[index],
                     "Contribution": float(contribution)})
    return pd.DataFrame(rows, columns=["Word or phrase", "Contribution"])
if st.button("Check message"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0]

        if prediction == "spam":
            st.error("Prediction: spam")
        else:
            st.success("Prediction: Not spam")
        explanation = explain_prediction(model, message)
        classifier = model.named_steps["classifier"]
        direction = 1 if classifier.classes_[1] == "spam" else -1
        starting_offset = float(classifier.intercept_[0]) * direction
        spam_total = explanation.loc[explanation["Contribution"] > 0, "Contribution"].sum()
        ham_total = explanation.loc[explanation["Contribution"] < 0, "Contribution"].sum()
        final_score = starting_offset + spam_total + ham_total
        st.write(f"Starting offset: {starting_offset:.4f}")
        st.write(f"All contributions toward spam: {spam_total:.4f}")
        st.write(f"All contributions toward ham: {ham_total:.4f}")
        st.write(f"Final score: {final_score:.4f}")

        st.subheader("Which word or phrase influenced this prediction?")

        if explanation.empty:
            st.info("No words or phrases matched the model's vocabulary.")
        else:
            toward_spam = explanation[explanation["Contribution"] > 0].nlargest(5, "Contribution")
            toward_ham = explanation[explanation["Contribution"] < 0].nsmallest(5, "Contribution")

            st.write("Top influences toward spam")
            if toward_spam.empty:
                st.write("None found")
            else:
                st.dataframe(toward_spam, hide_index = True)
            st.write("Top influences toward ham")
            if toward_ham.empty:
                st.write("None found")
            else:
                st.dataframe(toward_ham, hide_index = True)

st.caption("Learning project trained on SMS examples. "
           "Predictions can be incorrect. ")
