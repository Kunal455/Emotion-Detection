import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

print("Loading data...")
try:
    df = pd.read_csv("go_train.csv")
    print(f"Loaded {len(df)} rows.")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)


df = df[df['example_very_unclear'] == False]


emotion_cols = df.columns[9:]

valid_emotions = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

actual_cols = [c for c in df.columns if c in valid_emotions]
if not actual_cols:
    
     actual_cols = df.columns[9:]

print(f"Using emotion columns: {list(actual_cols)}")


df['label'] = df[actual_cols].idxmax(axis=1)

X = df['text'].fillna('')
y = df['label']

print("Training LogisticRegression...")
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1))
])

pipe.fit(X, y)

print("Saving model pipeline in some minutes...")
with open("go_emotion_model.pkl", "wb") as f:
    pickle.dump(pipe, f)
print("Done. Model saved to go_emotion_model.pkl")