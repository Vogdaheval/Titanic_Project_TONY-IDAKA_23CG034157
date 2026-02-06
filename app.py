from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained Random Forest model
model = joblib.load("model/titanic_survival_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input from form
        pclass = int(request.form["pclass"])
        sex = int(request.form["sex"])
        age = float(request.form["age"])
        fare = float(request.form["fare"])
        embarked = int(request.form["embarked"])

        # Create input array for model
        input_data = np.array([[pclass, sex, age, fare, embarked]])

        # Predict
        prediction = model.predict(input_data)[0]

        result = "Survived" if prediction == 1 else "Did Not Survive"
        return render_template("index.html", prediction_text=f"Prediction: {result}")
    
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
