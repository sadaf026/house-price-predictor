import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained pipeline
model = joblib.load("models/house_pipeline.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        age = int(request.form["age"])
        location = request.form["location"]
        property_type = request.form["property_type"]

        # Create dataframe
        data = {
            "Area": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Age": age,
            "Location": location,
            "Property_Type": property_type
        }

        df = pd.DataFrame([data])

        # Predict
        prediction = model.predict(df)[0]

        # Return result + keep form values
        return render_template(
            "index.html",
            prediction=round(prediction, 2),
            area=area,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            age=age,
            location=location,
            property_type=property_type
        )

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)