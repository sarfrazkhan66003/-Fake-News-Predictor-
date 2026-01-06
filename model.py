import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load datasets
df_true = pd.read_csv(r"C:\Users\DELL\Desktop\Sarfraz (Code_File)\PW Data Science\Project\fake-news-main\data\dataset_true.csv")
df_fake = pd.read_csv(r"C:\Users\DELL\Desktop\Sarfraz (Code_File)\PW Data Science\Project\fake-news-main\data\dataset_fake.csv")

# Auto-detect text column
def detect_text_column(df):
    for col in df.columns:
        if col.lower() in ["text", "content", "news", "article", "body", "headline", "title"]:
            return col
    return df.select_dtypes(include="object").columns[0]

df_true["text"] = df_true[detect_text_column(df_true)]
df_fake["text"] = df_fake[detect_text_column(df_fake)]

df_true["label"] = 1
df_fake["label"] = 0

df = pd.concat([df_true, df_fake])
df = df[["text", "label"]].dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Train-test split
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ TF-IDF (FIT HOGA YAHAN)
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_vec = vectorizer.fit_transform(X_train)  # 🔥 FIT HERE
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
print("Accuracy:", accuracy_score(y_test, model.predict(X_test_vec)))

# ✅ SAVE PROPERLY FITTED OBJECTS
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("✅ vectorizer.pkl & model.pkl SAVED CORRECTLY")
