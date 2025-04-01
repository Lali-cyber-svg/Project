import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# 🔹 Load the dataset
file_path = r"C:\Users\rajia\OneDrive\Desktop\smartbridge\smartbridge\traffic_volume.csv" # Ensure this file exists in the same directory
try:
    data = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: File '{file_path}' not found!")
    exit()

# 🔹 Handle missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

# 🔹 Extract date and time features
data[["day", "month", "year"]] = data["date"].str.split("-", expand=True)
data[["hours", "minutes", "seconds"]] = data["Time"].str.split(":", expand=True)

# Convert to numeric
data["day"] = data["day"].astype(int)
data["month"] = data["month"].astype(int)
data["year"] = data["year"].astype(int)
data["hours"] = data["hours"].astype(int)
data["minutes"] = data["minutes"].astype(int)
data["seconds"] = data["seconds"].astype(int)

# 🔹 Drop original date and time columns
data.drop(columns=['date', 'Time'], axis=1, inplace=True)

# 🔹 Separate features (X) and target (y)
y = data['traffic_volume']
X = data.drop(columns=['traffic_volume'], axis=1)

# 🔹 Identify categorical columns
categorical_cols = ['weather', 'holiday']

# 🔹 Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), X.columns.difference(categorical_cols))
    ])

# 🔹 Define and train the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the model
model.fit(X_train, y_train)

# Save the trained model
pickle.dump(model, open("model.pkl", "wb"))
print("✅ Model saved as 'model.pkl'")

# Save the preprocessor (encoder)
pickle.dump(preprocessor, open("encoder.pk1", "wb"))
print("✅ Preprocessor saved as 'encoder.pk1'")

# 🔹 Evaluate model performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"📊 Training Score: {train_score:.2f}")
print(f"📊 Test Score: {test_score:.2f}")
