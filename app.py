import streamlit as st
import pickle
import sqlite3

# ---------------- PAGE CONFIG (FIRST LINE) ----------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
with open(r"C:\Users\DELL\Desktop\Sarfraz (Code_File)\PW Data Science\Project\fake-news-main\model.pkl", "rb") as f:
    model = pickle.load(f)

with open(r"C:\Users\DELL\Desktop\Sarfraz (Code_File)\PW Data Science\Project\fake-news-main\vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("feedback.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    predicted_label TEXT,
    actual_label TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# ---------------- UI ----------------
st.title("📰 Fake News Detection System")
st.write("Enter a news article below to check whether it is **Real or Fake**.")

news_text = st.text_area("✍️ Enter News Article", height=200)

check_btn = st.button("🔍 Check News")
submit_btn = st.button("📩 Submit Feedback")

# ---------------- PREDICTION ----------------
if check_btn:
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        text_vector = vectorizer.transform([news_text])
        prediction = model.predict(text_vector)[0]   # ← int (0 or 1)

        # Save to session
        st.session_state["news_text"] = news_text

        # 0 = Fake, 1 = Real
        if prediction == 0:
            st.error("🚨 This news is **FAKE**")
            st.session_state["prediction_label"] = "Fake"
        else:
            st.success("✅ This news is **REAL**")
            st.session_state["prediction_label"] = "Real"

        st.session_state["predicted"] = True

# ---------------- FEEDBACK ----------------
if "predicted" in st.session_state:
    st.markdown("---")
    st.subheader("🗳️ Feedback (Optional)")

    actual_label = st.radio(
        "Is the prediction correct?",
        ("Real", "Fake")
    )

    if submit_btn:
        cursor.execute(
            "INSERT INTO feedback (text, predicted_label, actual_label) VALUES (?, ?, ?)",
            (
                st.session_state["news_text"],
                st.session_state["prediction_label"],
                actual_label
            )
        )
        conn.commit()
        st.success("🙏 Thank you! Feedback saved.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Developer Sarfraz Khan | Fake News Detection Project")
