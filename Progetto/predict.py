from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Carica il modello una sola volta all'avvio
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.bin")

with open(model_path, "rb") as f:
    model = pickle.load(f)

print(f"✅ Modello caricato da: {model_path}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    try:
        inputs = [
            float(request.args["daily_work_hours"]),
            float(request.args["sleep_hours"]),
            float(request.args["caffeine_intake"]),
            float(request.args["bugs_per_day"]),
            float(request.args["meetings_per_day"]),
            float(request.args["exercise_hours"])
        ]
        print(f"Input ricevuti: {inputs}")

        output = model.predict([inputs])[0]
        print(f"Output modello: {output}")

        mapping = {0: "Low", 1: "Medium", 2: "High"}
        response = mapping.get(int(output), "Errore")
        return render_template("predict.html", resp=response)
    except Exception as e:
        return f"Errore: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
