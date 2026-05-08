import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/house_prices.csv")

X = df.drop("Price", axis=1)
y = df["Price"]

X = pd.get_dummies(X, drop_first=True)

model = RandomForestRegressor()
model.fit(X, y)

importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

print(feature_df.head())

plt.figure(figsize=(8,5))
plt.barh(
    feature_df["Feature"][:10],
    feature_df["Importance"][:10]
)

plt.title("Top Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()