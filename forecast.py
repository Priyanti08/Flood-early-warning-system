"""
forecast.py — Rise-speed estimation, water-level forecasting, and CWC
threshold ETA calculation for the flood early warning system.

These functions are imported by app.py. The rise speed is estimated by
linear regression over a sliding window of recent water-level samples,
with outlier rejection. Forecasts are linear extrapolations of the current
rise rate; see the paper (Section on forecasting limitations) for why this
under-predicts accelerating GLOF rises and should be replaced with an
LSTM model before field deployment.
"""

# CWC stage thresholds for the Teesta at Domohani (prototype scale, cm)
WARNING_LEVEL = 65
DANGER_LEVEL = 89
EXTREME_LEVEL = 120


def calculate_rise_speed(water_history):
    """
    Estimate rise speed (cm/hr) by linear regression over the most recent
    samples in water_history (a sequence of 1 Hz water-level readings).
    Outliers more than 20 cm from the window median are rejected.
    """
    if len(water_history) < 6:
        return 0.0

    n = min(10, len(water_history))
    vals = list(water_history)[-n:]

    sorted_vals = sorted(vals)
    median = sorted_vals[len(sorted_vals) // 2]
    filtered = [v for v in vals if abs(v - median) < 20]
    if len(filtered) < 3:
        return 0.0

    x = list(range(len(filtered)))
    mx = sum(x) / len(x)
    my = sum(filtered) / len(filtered)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, filtered))
    den = sum((xi - mx) ** 2 for xi in x)
    if den == 0:
        return 0.0

    # slope is cm per sample (1 s); convert to cm/hr
    return round((num / den) * 3600, 2)


def calculate_forecast(current, rise_speed):
    """Linear 1/3/6-hour level forecasts. Returns (f1, f3, f6) in cm."""
    f1 = round(current + rise_speed * 1, 1)
    f3 = round(current + rise_speed * 3, 1)
    f6 = round(current + rise_speed * 6, 1)
    return max(0, f1), max(0, f3), max(0, f6)


def eta_to_level(current, rise_speed, target):
    """Estimated time of arrival at a threshold, as a human-readable string."""
    if rise_speed <= 0:
        return "--"
    if current >= target:
        return "EXCEEDED"
    hours = (target - current) / rise_speed
    if hours > 24:
        return ">24 hrs"
    elif hours >= 1:
        return "~" + str(round(hours, 1)) + " hrs"
    else:
        return "~" + str(int(hours * 60)) + " mins"


def all_threshold_etas(current, rise_speed):
    """Convenience: ETAs for all three CWC thresholds as a dict."""
    return {
        "warning": eta_to_level(current, rise_speed, WARNING_LEVEL),
        "danger":  eta_to_level(current, rise_speed, DANGER_LEVEL),
        "extreme": eta_to_level(current, rise_speed, EXTREME_LEVEL),
    }


if __name__ == "__main__":
    # Quick self-test
    history = [10, 11, 12, 13, 15, 17, 20, 24, 29, 35]
    rs = calculate_rise_speed(history)
    print("Rise speed (cm/hr):", rs)
    print("Forecasts 1/3/6h:", calculate_forecast(history[-1], rs))
    print("Threshold ETAs:", all_threshold_etas(history[-1], rs))
