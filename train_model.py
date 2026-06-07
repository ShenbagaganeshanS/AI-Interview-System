import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib


data = {
    "similarity": [0.9, 0.75, 0.6, 0.4, 0.2],
    "voice_score": [3, 3, 2, 2, 1],  # HIGH=3, MED=2, LOW=1
    "emotion_score": [3, 3, 2, 1, 1], # happy/neutral=3, surprise=2, others=1
    "final_score": [90, 80, 65, 50, 30]
}

df = pd.DataFrame(data)
print(df)

X = df[["similarity", "voice_score", "emotion_score"]]
y = df["final_score"]

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, "score_model.pkl")

print("Model trained and saved!")