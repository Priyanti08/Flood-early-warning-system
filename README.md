# Flood Early Warning System — Teesta & Rangit Rivers

A low-cost IoT-based flood early warning prototype for the Teesta and Rangit river basins in Sikkim / West Bengal, India. The system integrates seven environmental sensors on an Arduino Uno, streams data to a Raspberry Pi 4, classifies flood risk in real time with a Random Forest model, and serves a live web dashboard with water-level forecasts aligned to Central Water Commission (CWC) threshold stages.

Built for under ₹9000 in sensing hardware, the project targets community-level monitoring of upstream Himalayan reaches not covered by national gauge networks. It was motivated by the October 2023 South Lhonak GLOF, which destroyed the Teesta-III dam and caused over 55 deaths with only ~18 minutes of downstream warning.

## Features

- Seven-sensor environmental stack: ultrasonic water level, rainfall, temperature/humidity, flow rate, soil moisture, barometric pressure, and turbidity
- Random Forest risk classifier (4 classes: Normal, Watch, Warning, Flood) trained on a physics-informed synthetic dataset
- Real-time Flask dashboard with live readings, rise-speed estimation, and 1/3/6-hour level forecasts
- CWC-aligned thresholds (Warning 65 cm, Danger 89 cm, Extreme 120 cm at Domohani) with threshold ETA estimates
- Graduated physical alerts: LEDs, buzzer, and a servo-actuated model dam gate driven over a bidirectional serial link

## Hardware

| Sensor | Measured quantity | Arduino pin |
|---|---|---|
| HC-SR04 ultrasonic | Water level (cm) | D9 / D10 |
| DHT11 | Temperature, humidity | D2 |
| Rain plate | Rainfall (binary) | D7 |
| YF-S201 | Flow rate (L/min) | D3 (interrupt) |
| Capacitive soil probe | Soil moisture (raw) | A0 |
| BMP280 | Pressure (hPa) | A4 / A5 (I²C) |
| Turbidity probe | Turbidity (NTU) | A1 |
| SG90 servo | Model dam gate | D11 |

Plus three LEDs, a piezo buzzer, an Arduino Uno, and a Raspberry Pi 4 (4 GB).

## Repository structure

```
flood-early-warning-system/
├── README.md          This file
├── app.py             Flask dashboard + real-time inference loop
├── train_model.py     Trains the Random Forest, writes model.joblib
├── simulator.py       Generates the physics-informed synthetic dataset
├── forecast.py        Rise-speed estimation + level forecasting + threshold ETAs
├── model.joblib       Trained Random Forest model
├── dataset.csv        6,000-row training dataset
└── paper.pdf          Full research paper
```

## Setup

On the Raspberry Pi (Python 3.11+):

```bash
pip install flask pyserial joblib scikit-learn numpy pandas
```

Upload the Arduino sketch to the Uno, connect it over USB, then run:

```bash
python app.py
```

Open `http://raspberrypi.local:5000` from any device on the same network.

To regenerate the dataset and retrain from scratch:

```bash
python simulator.py     # writes dataset.csv
python train_model.py   # writes model.joblib + scaler
```

## Dataset

The dataset contains 6,000 records across four CWC-aligned classes. Class distribution: Normal 3,302; Watch 1,499; Warning 767; Flood 432. It is parameterised against 115 years of IMD Sikkim rainfall data, 61 years of Gangtok station records, and published NHPC Rangit discharge series. The near-perfect classifier accuracy on this data reflects the internal consistency of the simulator and is **not** a claim of real-world field skill; validation against real CWC gauge observations is the next step.

## Limitations

This is a laboratory prototype. The model is trained on simulated data, forecasting uses linear extrapolation (which under-predicts accelerating GLOF rises), sensing is single-node, and all components are consumer-grade and bench-tested indoors. See the paper for the full limitations discussion and the roadmap to field deployment.

## License

MIT License — free to use, modify, and distribute with attribution.
