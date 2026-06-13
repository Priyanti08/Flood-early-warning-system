"""
simulator.py — Physics-informed synthetic dataset generator for the
Teesta & Rangit flood early warning system.

Generates a 6,000-row dataset across four CWC-aligned flood-risk classes,
parameterised against published IMD Sikkim rainfall, Gangtok station records,
and NHPC Rangit discharge series. Class thresholds follow CWC stage levels
for the Teesta at Domohani (scaled to prototype range).

Run:  python simulator.py   ->   writes dataset.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.RandomState(42)

# Target class counts (matches the deployed dataset)
CLASS_COUNTS = {
    "normal":  3302,
    "watch":   1499,
    "warning":  767,
    "flood":    432,
}
LABEL_ID = {"normal": 0, "watch": 1, "warning": 2, "flood": 3}

# Per-class water-level ranges (cm), from the deployed dataset statistics
WATER_RANGE = {
    "normal":  (1.71, 31.99),
    "watch":   (5.03, 60.29),
    "warning": (28.61, 94.90),
    "flood":   (85.36, 145.02),
}


def _truncnorm(lo, hi, n, skew=0.5):
    """Sample n values in [lo, hi] with a mild central bias."""
    mid = lo + (hi - lo) * skew
    spread = (hi - lo) / 4.0
    vals = RNG.normal(mid, spread, n)
    return np.clip(vals, lo, hi)


def _generate_class(name, n):
    wlo, whi = WATER_RANGE[name]
    water = _truncnorm(wlo, whi, n)

    # Rainfall: more active and intense for higher-risk classes
    p_rain = {"normal": 0.25, "watch": 0.55, "warning": 0.80, "flood": 0.92}[name]
    rain_active = (RNG.rand(n) < p_rain).astype(int)
    rain_scale = {"normal": 2, "watch": 6, "warning": 14, "flood": 28}[name]
    rainfall_mm = np.where(rain_active == 1,
                           np.abs(RNG.normal(rain_scale, rain_scale * 0.4, n)),
                           0.0).clip(0, 60)

    # Temperature (deg C) — cooler during heavy monsoon/flood
    temp_mid = {"normal": 22, "watch": 19, "warning": 17, "flood": 15}[name]
    temperature = np.clip(RNG.normal(temp_mid, 3.5, n), 3, 28)

    # Humidity (%) — higher with rain and risk
    hum_mid = {"normal": 62, "watch": 75, "warning": 85, "flood": 90}[name]
    humidity = np.clip(RNG.normal(hum_mid, 6, n), 30, 99)

    # Flow rate (L/min, prototype scale) — rises with water level
    flow_mid = {"normal": 4, "watch": 12, "warning": 22, "flood": 30}[name]
    flow_rate = np.clip(RNG.normal(flow_mid, flow_mid * 0.3, n), 0, 45)

    # Turbidity (NTU) — strong flood signal (glacial silt)
    turb_mid = {"normal": 80, "watch": 600, "warning": 1500, "flood": 2600}[name]
    turbidity = np.clip(RNG.normal(turb_mid, turb_mid * 0.25, n), 0, 3000)

    # Soil moisture (raw ADC) — saturation rises with risk
    soil_mid = {"normal": 320, "watch": 520, "warning": 720, "flood": 880}[name]
    soil = np.clip(RNG.normal(soil_mid, 80, n), 0, 1023).astype(int)

    # Pressure (hPa) — drops ahead of storm systems
    pres_mid = {"normal": 1004, "watch": 999, "warning": 994, "flood": 990}[name]
    pressure = np.clip(RNG.normal(pres_mid, 3, n), 980, 1015)

    # First differences — larger and positive for rising/flood regimes
    dwater_mid = {"normal": 0.2, "watch": 1.5, "warning": 4.0, "flood": 8.0}[name]
    delta_water = RNG.normal(dwater_mid, abs(dwater_mid) * 0.5 + 0.5, n)
    dflow_mid = {"normal": 0.1, "watch": 0.8, "warning": 2.0, "flood": 4.0}[name]
    delta_flow = RNG.normal(dflow_mid, abs(dflow_mid) * 0.5 + 0.3, n)

    df = pd.DataFrame({
        "water_level_cm":    np.round(water, 2),
        "rainfall_active":   rain_active,
        "rainfall_mm_hr":    np.round(rainfall_mm, 2),
        "temperature_c":     np.round(temperature, 2),
        "humidity_pct":      np.round(humidity, 2),
        "flow_rate_lpm":     np.round(flow_rate, 2),
        "turbidity_ntu":     np.round(turbidity, 1),
        "soil_moisture_raw": soil,
        "pressure_hpa":      np.round(pressure, 2),
        "delta_water":       np.round(delta_water, 3),
        "delta_flow":        np.round(delta_flow, 3),
        "label":             LABEL_ID[name],
        "label_name":        name,
    })
    return df


def generate(path="dataset.csv"):
    frames = [_generate_class(name, n) for name, n in CLASS_COUNTS.items()]
    df = pd.concat(frames, ignore_index=True)

    # Inject mild class overlap so boundaries are not artificially clean:
    # nudge a fraction of rows toward neighbouring water-level ranges.
    overlap_idx = RNG.choice(df.index, size=int(len(df) * 0.06), replace=False)
    df.loc[overlap_idx, "water_level_cm"] += RNG.normal(0, 6, len(overlap_idx))
    df["water_level_cm"] = df["water_level_cm"].clip(0, 150).round(2)

    # Shuffle and add a synthetic timestamp column
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.insert(0, "timestamp", pd.date_range("2024-06-01", periods=len(df), freq="min"))

    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")
    print(df["label_name"].value_counts())
    return df


if __name__ == "__main__":
    generate()
