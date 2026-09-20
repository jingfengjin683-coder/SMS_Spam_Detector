from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# Load the data and organize the data in a pandas dataframe
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"

with DATA_PATH.open(encoding = "utf-8") as file:
    rows = [line.rstrip("\r\n").split("\t", 1) for line in file
            if line.strip()
            ]

df = pd.DataFrame(rows, columns = ["label", "message"])
df = df.dropna(subset = ["label", "message"])
df = df.drop_duplicates(subset = ["message"])
df = df.reset_index(drop = True)

#Split the data into training and testing data
x = df["message"]
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.2,
                                                    random_state = 42,
                                                    stratify = y,)

#Training the model
models = {"Naives Bayes": Pipeline([("tfidf", TfidfVectorizer()), ("classifier", MultinomialNB())]),
         "Logistic regression": Pipeline([("tfidf", TfidfVectorizer()), ("classifier", LogisticRegression(
             max_iter = 1000, class_weight = "balanced"
         ))]),
          "Logistic regression + word pairs": Pipeline([("tfidf", TfidfVectorizer(ngram_range = (1, 2))), ("classifier", LogisticRegression
          (max_iter = 1000, class_weight = "balanced"))])}
scoring = {
    "precision": make_scorer(precision_score, pos_label = "spam", zero_division = 0),
    "recall": make_scorer(recall_score, pos_label="spam", zero_division = 0),
    "f1": make_scorer(f1_score, pos_label = "spam", zero_division = 0),
}
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
for name, candidate in models.items():
    scores = cross_validate(candidate, x_train, y_train, cv = folds, scoring = scoring)
    print(f"\n{name}")
    print(f"Spam precision: {scores['test_precision'].mean():.3f}")
    print(f"Spam recall:    {scores['test_recall'].mean():.3f}")
    print(f"Spam F1:        {scores['test_f1'].mean():.3f}")
model = models["Logistic regression + word pairs"]
model.fit(x_train, y_train)

#Evaluation of the model
predictions = model.predict(x_test)
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

matrix = confusion_matrix(y_test, predictions, labels = ["ham", "spam"])

print("\nConfusion matrix:")
print(pd.DataFrame(matrix, index = ["Actual ham", "Actual spam"],
                   columns = ["Predicted ham", "Predicted spam"],))

results = pd.DataFrame({
    "message": x_test,
    "actual": y_test,
    "predicted": predictions,
})

mistakes = results[results["actual"] != results["predicted"]]

MODEL_PATH = BASE_DIR / "spam_model.joblib"
joblib.dump(model, MODEL_PATH)
print(f"\nSaved model to {MODEL_PATH}")
