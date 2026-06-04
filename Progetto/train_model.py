import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Carica dataset (metti il tuo CSV qui)
df = pd.read_csv("burnout.csv")

# Rimozioni come nel tuo progetto
df = df.drop(columns=['commits_per_day', 'age', 'experience_years', 'screen_time', 'stress_level'])
df = df.dropna()

# Feature e target
X = df[[
    "daily_work_hours",
    "sleep_hours",
    "caffeine_intake",
    "bugs_per_day",
    "meetings_per_day",
    "exercise_hours"
]]

y = df["burnout_level"]

# Encoding target
mapping = {'Low': 0, 'Medium': 1, 'High': 2}
y = y.map(mapping)

# Pipeline (scaler + modello)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(solver="saga",random_state=42, max_iter=1000)) #modello allenato con gli iperparametri migliori trovati dalla grid search
])


# Training
model.fit(X, y)

# Salvataggio
with open("model.bin", "wb") as f:
    pickle.dump(model, f)

print("Modello salvato in model.bin")