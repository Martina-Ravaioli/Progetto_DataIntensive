from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Predizione
@app.route("/predict")
def predict():
    try:
        # Lettura input dal form
        inputs = [
            float(request.args["daily_work_hours"]),
            float(request.args["sleep_hours"]),
            float(request.args["caffeine_intake"]),
            float(request.args["bugs_per_day"]),
            float(request.args["meetings_per_day"]),
            float(request.args["exercise_hours"])
        ]

        # Caricamento modello
        with app.open_resource("model.bin", "rb") as f:
            model = pickle.load(f)

        # Predizione
        output = model.predict([inputs])[0]

        # Mapping classi
        mapping = {
            0: "Low",
            1: "Medium",
            2: "High"
        }

        response = mapping.get(output, "Errore")

        return render_template("predict.html", resp=response)

    except Exception as e:
        return f"Errore: {e}"

# Avvio del server (IMPORTANTE)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
