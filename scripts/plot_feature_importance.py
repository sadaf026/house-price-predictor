import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load pipeline
model = joblib.load("models/house_pipeline.pkl")

# Get model
final_model = model.named_steps["regressor"]

# Get feature names from preprocessor
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# Get importance
importances = final_model.feature_importances_

# Create dataframe
feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(10)

# Plot
plt.figure(figsize=(10,5))
plt.barh(feat_df["Feature"], feat_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()

# Save image
plt.savefig("static/feature_importance.png")

print("✅ Feature importance chart saved in /static folder")