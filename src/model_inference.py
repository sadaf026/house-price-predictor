import joblib
import pandas as pd

model = joblib.load("models/best_model.pkl")

sample_data = pd.DataFrame({
    "Area": [2000],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "Age": [5]
})

sample_data = pd.get_dummies(sample_data)

prediction = model.predict(sample_data)

print(f"Predicted House Price: ₹{prediction[0]}")