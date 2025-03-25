# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# App title
st.title("💧 Water Potability Prediction App")
st.write("Enter water quality parameters to predict potability.")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    df.fillna(df.median(), inplace=True)
    return df

# Load and prepare the data
df = load_data()
X = df.drop("Potability", axis=1)
y = df["Potability"]

# Split and scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
    model.fit(X_train, y_train)
    return model

# Train Random Forest model
model = train_model()

# Input fields for user parameters
pH = st.number_input("pH Level (0-14)", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness (mg/L)", min_value=0.0, max_value=500.0, value=150.0)
solids = st.number_input("Solids (ppm)", min_value=0.0, max_value=50000.0, value=10000.0)
chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, max_value=15.0, value=5.0)
sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, max_value=500.0, value=200.0)
conductivity = st.number_input("Conductivity (µS/cm)", min_value=0.0, max_value=1000.0, value=500.0)
organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, max_value=30.0, value=10.0)
trihalomethanes = st.number_input("Trihalomethanes (µg/L)", min_value=0.0, max_value=150.0, value=50.0)
turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=10.0, value=4.0)

# Create a DataFrame from user input
data = {
    "pH": [pH],
    "Hardness": [hardness],
    "Solids": [solids],
    "Chloramines": [chloramines],
    "Sulfate": [sulfate],
    "Conductivity": [conductivity],
    "Organic_carbon": [organic_carbon],
    "Trihalomethanes": [trihalomethanes],
    "Turbidity": [turbidity],
}
input_df = pd.DataFrame(data)

# Scale user input to match model training
input_scaled = scaler.transform(input_df)

# Prediction button
if st.button("Predict Potability"):
    prediction = model.predict(input_scaled)
    result = "✅ Potable" if prediction[0] == 1 else "❌ Not Potable"
    st.success(f"Prediction: {result}")

# Show model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")
