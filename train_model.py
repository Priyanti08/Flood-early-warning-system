import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

# Load dataset
def train_model():
    print("Loading dataset...")
    df = pd.read_csv('teesta_rangit_flood_dataset.csv')

    features = [
        'water_level_cm', 'rainfall_active', 'temperature_c',
        'humidity_pct', 'flow_rate_lpm', 'turbidity_ntu',
        'soil_moisture_raw', 'pressure_hpa',
        'delta_water', 'delta_flow'
    ]

    X = df[features]
    y = df['label']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    print("\nModel Performance:")
    print(classification_report(y_test, model.predict(X_test)))

    # Save model and scaler
    joblib.dump(model, 'flood_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model saved!")

def predict(values):
    # Load model
    model = joblib.load('flood_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Calculate delta values (rate of change)
    features = values + [0, 0]  # delta_water, delta_flow placeholder

    # Scale and predict
    scaled = scaler.transform([features])
    prediction = model.predict(scaled)[0]

    labels = {
        0: ('NORMAL', 'green'),
        1: ('WATCH', 'yellow'),
        2: ('WARNING', 'orange'),
        3: ('FLOOD', 'red')
    }

    return prediction, labels[prediction]

if __name__ == '__main__':
    train_model()
