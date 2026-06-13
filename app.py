from flask import Flask, render_template_string, jsonify
import threading
import serial
import time
import joblib
import numpy as np
from collections import deque

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False

app = Flask(__name__)

WARNING_LEVEL = 65
DANGER_LEVEL  = 89
EXTREME_LEVEL = 120
sensor_data = {
    'water_level': 0,
    'rainfall': 0,
    'rainfall_active': 0,
    'temperature': 0,
    'humidity': 0,
    'flow_rate': 0,
    'turbidity': 0,
    'turbidity_desc': 'Clear',
    'soil_moisture': 0,
    'soil_desc': 'Dry',
    'pressure': 0,
    'risk_level': 0,
    'risk_name': 'LOW',
    'risk_color': '#2ecc71',
    'cwc_category': 'NORMAL',
    'rise_speed': 0.0,
    'rise_direction': 'STABLE',
    'forecast_1hr': 0,
    'forecast_3hr': 0,
    'forecast_6hr': 0,
    'warning_status': 'Not Reached',
    'danger_status': 'Not Reached',
    'extreme_status': 'Not Reached',
    'warning_eta': '--',
    'danger_eta': '--',
    'extreme_eta': '--',
    'last_updated': '--',
}

history       = deque(maxlen=100)
water_history = deque(maxlen=60)

ser       = None
demo_mode = True


def calculate_rise_speed():
    if len(water_history) < 5:
        return 0.0
    vals = list(water_history)[-5:]
    diff = vals[-1] - vals[0]
    # Each reading is 1 second apart, 5 readings = 5 seconds
    # Convert cm in 5 seconds to cm per hour
    speed = round(diff * 720, 1)
    if abs(speed) < 2:
        return 0.0
    if speed > 600:  return 600.0
    if speed < -600: return -600.0
    return speed


def calculate_forecast(current, rise_speed):
    f1 = round(current + rise_speed * 1, 1)
    f3 = round(current + rise_speed * 3, 1)
    f6 = round(current + rise_speed * 6, 1)
    return max(0, f1), max(0, f3), max(0, f6)


def eta_to_level(current, rise_speed, target):
    if rise_speed <= 0:
        return '--'
    if current >= target:
        return 'EXCEEDED'
    hours = (target - current) / rise_speed
    if hours > 24:
        return '>24 hrs'
    elif hours >= 1:
        return '~' + str(round(hours, 1)) + ' hrs'
    else:
        return '~' + str(int(hours * 60)) + ' mins'


def get_risk(water_level, rise_speed, ml_pred):
    if water_level >= EXTREME_LEVEL or ml_pred == 3:
        return 3, 'CRITICAL', '#e74c3c', 'EXTREME'
    elif water_level >= DANGER_LEVEL or ml_pred == 2:
        return 2, 'HIGH', '#e67e22', 'SEVERE'
    elif water_level >= WARNING_LEVEL or ml_pred == 1:
        return 1, 'MODERATE', '#f39c12', 'ABOVE NORMAL'
    else:
        return 0, 'LOW', '#2ecc71', 'NORMAL'


def get_turbidity_desc(ntu):
    if ntu < 10:
        return 'Clear'
    elif ntu < 50:
        return 'Slightly Turbid'
    elif ntu < 150:
        return 'Turbid'
    elif ntu < 300:
        return 'Very Turbid'
    else:
        return 'Extremely Turbid'


def get_soil_desc(raw):
    if raw < 300:
        return 'Dry'
    elif raw < 500:
        return 'Moist'
    elif raw < 700:
        return 'Wet'
    elif raw < 900:
        return 'Very Wet'
    else:
        return 'Saturated'


def control_alerts(risk_level):
    global ser, demo_mode
    try:
        if not demo_mode and ser:
            ser.write((str(risk_level) + '\n').encode())
    except Exception:
        pass


def read_serial():
    global ser, demo_mode

    try:
        model  = joblib.load('flood_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except Exception:
        print('Model not found! Run ml_model.py first')
        return

    try:
        ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        time.sleep(2)
        print('Connected to Arduino!')
        demo_mode = False
    except Exception:
        print('Arduino not connected - running in demo mode')
        demo_mode = True
        ser = None

    prev_water = 0
    prev_flow  = 0
    t_start    = time.time()

    while True:
        try:
            if demo_mode:
                import math
                t     = time.time() - t_start
                water = 20 + 10 * math.sin(t / 30)
                line  = (str(round(water, 1)) + ',1,5.0,22.0,75.0,10.0,30.0,400,995.0')
            else:
                line = ser.readline().decode('utf-8').strip()

            if line:
                values = line.split(',')
                if len(values) == 9:
                    v             = [float(x) for x in values]
                    water_level   = v[0]
                    rainfall_act  = int(v[1])
                    rainfall_mm   = v[2]
                    temperature   = v[3]
                    humidity      = v[4]
                    flow_rate     = v[5]
                    turbidity     = v[6]
                    soil_moisture = v[7]
                    pressure      = v[8]

                    water_history.append(water_level)
                    delta_water = water_level - prev_water
                    delta_flow  = flow_rate   - prev_flow
                    prev_water  = water_level
                    prev_flow   = flow_rate
                    features = [
                        water_level * 3,
                        rainfall_act,
                        temperature,
                        humidity, flow_rate, turbidity,
                        soil_moisture, pressure,
                        delta_water * 3,
                        delta_flow
                    ]
                    scaled  = scaler.transform([features])
                    ml_pred = int(model.predict(scaled)[0])
                    rise_speed          = calculate_rise_speed()
                    f1, f3, f6          = calculate_forecast(water_level, rise_speed)
                    risk_lvl, risk_name, risk_color, cwc_cat = get_risk(
                        water_level, rise_speed, ml_pred)

                    if rise_speed > 1:
                        direction = 'RISING ' + str(rise_speed) + ' cm/hr'
                    elif rise_speed < -1:
                        direction = 'FALLING ' + str(abs(rise_speed)) + ' cm/hr'
                    else:
                        direction = 'STABLE'

                    warn_eta   = 'EXCEEDED' if water_level >= WARNING_LEVEL else eta_to_level(water_level, rise_speed, WARNING_LEVEL)
                    danger_eta = 'EXCEEDED' if water_level >= DANGER_LEVEL  else eta_to_level(water_level, rise_speed, DANGER_LEVEL)
                    ext_eta    = 'EXCEEDED' if water_level >= EXTREME_LEVEL else eta_to_level(water_level, rise_speed, EXTREME_LEVEL)

                    sensor_data.update({
                        'water_level':     round(water_level, 1),
                        'rainfall':        rainfall_mm,
                        'rainfall_active': rainfall_act,
                        'temperature':     temperature,
                        'humidity':        humidity,
                        'flow_rate':       round(flow_rate, 1),
                        'turbidity':       round(turbidity, 1),
                        'turbidity_desc':  get_turbidity_desc(turbidity),
                        'soil_moisture':   int(soil_moisture),
                        'soil_desc':       get_soil_desc(soil_moisture),
                        'pressure':        pressure,
                        'risk_level':      risk_lvl,
                        'risk_name':       risk_name,
                        'risk_color':      risk_color,
                        'cwc_category':    cwc_cat,
                        'rise_speed':      rise_speed,
                        'rise_direction':  direction,
                        'forecast_1hr':    f1,
                        'forecast_3hr':    f3,
                        'forecast_6hr':    f6,
                        'warning_status':  'EXCEEDED' if water_level >= WARNING_LEVEL else 'Not Reached',
                        'danger_status':   'EXCEEDED' if water_level >= DANGER_LEVEL  else 'Not Reached',
                        'extreme_status':  'EXCEEDED' if water_level >= EXTREME_LEVEL else 'Not Reached',
                        'warning_eta':     warn_eta,
                        'danger_eta':      danger_eta,
                        'extreme_eta':     ext_eta,
                        'last_updated':    time.strftime('%Y-%m-%d %H:%M:%S'),
                    })

                    history.append({
                        'time':         time.strftime('%H:%M:%S'),
                        'water_level':  round(water_level, 1),
                        'rise_speed':   rise_speed,
                        'risk':         risk_name,
                        'cwc':          cwc_cat,
                        'forecast_1hr': f1,
                    })

                    control_alerts(risk_lvl)
                    print('Water:' + str(water_level) + 'cm | Speed:' + str(rise_speed) + 'cm/hr | Risk:' + risk_name)

        except Exception as e:
            print('Error: ' + str(e))
        time.sleep(1)


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Flood Early Warning System</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:Arial,sans-serif; background:#0a0f1e; color:#e0e0e0; padding:15px; }
        .header { text-align:center; padding:15px; margin-bottom:15px; background:linear-gradient(135deg,#1a2744,#0d1b33); border-radius:12px; border:1px solid #2a3a5c; }
        .header h1 { color:#4fc3f7; font-size:1.6em; }
        .header p { color:#90a4ae; font-size:0.85em; margin-top:4px; }
        .risk-banner { padding:18px; border-radius:12px; text-align:center; margin-bottom:15px; border:2px solid rgba(255,255,255,0.2); }
        .risk-banner h2 { font-size:2em; font-weight:bold; }
        .risk-banner p { font-size:1em; margin-top:5px; }
        .grid-3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-bottom:12px; }
        .grid-4 { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:12px; }
        .card { background:#111827; border-radius:10px; padding:14px; border:1px solid #1e3a5f; }
        .card-title { font-size:0.75em; color:#64b5f6; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }
        .card-value { font-size:1.8em; font-weight:bold; color:#fff; }
        .card-sub { font-size:0.8em; color:#90a4ae; margin-top:4px; }
        .section-title { font-size:0.9em; color:#4fc3f7; text-transform:uppercase; letter-spacing:2px; margin-bottom:10px; padding-bottom:6px; border-bottom:1px solid #1e3a5f; }
        .forecast-card { background:#111827; border-radius:10px; padding:14px; text-align:center; border:1px solid #1e3a5f; }
        .forecast-time { font-size:0.8em; color:#64b5f6; margin-bottom:6px; }
        .forecast-val { font-size:1.6em; font-weight:bold; }
        .f-danger { color:#e74c3c; }
        .f-warn { color:#f39c12; }
        .f-ok { color:#2ecc71; }
        .threshold-card { background:#111827; border-radius:10px; padding:14px; margin-bottom:12px; border:1px solid #1e3a5f; }
        .threshold-row { display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid #1a2744; font-size:0.9em; }
        .threshold-row:last-child { border-bottom:none; }
        .th-label { color:#90a4ae; }
        .th-level { font-weight:bold; color:#fff; }
        .th-eta { color:#f39c12; font-weight:bold; }
        .history-card { background:#111827; border-radius:10px; padding:14px; border:1px solid #1e3a5f; overflow-x:auto; }
        table { width:100%; border-collapse:collapse; font-size:0.85em; }
        th { background:#1e3a5f; color:#4fc3f7; padding:8px; text-align:center; }
        td { padding:7px; text-align:center; border-bottom:1px solid #1a2744; }
        .update-bar { text-align:center; font-size:0.75em; color:#546e7a; margin-top:12px; padding:8px; }
        .speed-box { background:#111827; border-radius:10px; padding:14px; border:1px solid #1e3a5f; text-align:center; }
        .speed-val { font-size:2.2em; font-weight:bold; }
        .rising { color:#e74c3c; }
        .falling { color:#2ecc71; }
        .stable { color:#90a4ae; }
    </style>
    <script>setTimeout(function(){ location.reload(); }, 3000);</script>
</head>
<body>
<div class="header">
    <h1>Flood Early Warning System</h1>
    <p>Teesta and Rangit Rivers, Sikkim / West Bengal, India | CWC Aligned Monitoring</p>
</div>
<div class="risk-banner" style="background:linear-gradient(135deg,{{ risk_color }}22,{{ risk_color }}44);border-color:{{ risk_color }};">
    <h2 style="color:{{ risk_color }};">RISK LEVEL: {{ risk_name }}</h2>
    <p>CWC Category: <strong>{{ cwc_category }}</strong> | Updated: {{ last_updated }}</p>
</div>
<p class="section-title">Live Water Behaviour</p>
<div class="grid-3">
    <div class="card" style="border-color:{{ risk_color }};text-align:center;">
        <div class="card-title">Current Water Level</div>
        <div class="card-value" style="font-size:2.5em;color:{{ risk_color }};">{{ water_level }} cm</div>
        <div class="card-sub">{{ rise_direction }}</div>
    </div>
    <div class="speed-box">
        <div class="card-title">Rise Speed</div>
        <div class="speed-val {% if rise_speed > 1 %}rising{% elif rise_speed < -1 %}falling{% else %}stable{% endif %}">
            {{ rise_speed }} cm/hr
        </div>
        <div class="card-sub">
            {% if rise_speed > 60 %}EXTREMELY FAST - GLOF level
            {% elif rise_speed > 30 %}Very Fast - Flash flood
            {% elif rise_speed > 15 %}Fast - Heavy monsoon
            {% elif rise_speed > 5 %}Moderate rise
            {% elif rise_speed < -5 %}Receding
            {% else %}Stable{% endif %}
        </div>
    </div>
    <div class="card" style="text-align:center;">
        <div class="card-title">Rainfall</div>
        <div class="card-value">{{ rainfall }} mm/hr</div>
        <div class="card-sub">{% if rainfall_active %}Active{% else %}No rainfall{% endif %}</div>
    </div>
</div>
<p class="section-title">Water Level Forecast</p>
<div class="grid-3">
    <div class="forecast-card">
        <div class="forecast-time">In 1 Hour</div>
        <div class="forecast-val {% if forecast_1hr >= 89 %}f-danger{% elif forecast_1hr >= 65 %}f-warn{% else %}f-ok{% endif %}">{{ forecast_1hr }} cm</div>
        <div class="card-sub">{% if forecast_1hr >= 120 %}Above Extreme{% elif forecast_1hr >= 89 %}Above Danger{% elif forecast_1hr >= 65 %}Above Warning{% else %}Safe{% endif %}</div>
    </div>
    <div class="forecast-card">
        <div class="forecast-time">In 3 Hours</div>
        <div class="forecast-val {% if forecast_3hr >= 89 %}f-danger{% elif forecast_3hr >= 65 %}f-warn{% else %}f-ok{% endif %}">{{ forecast_3hr }} cm</div>
        <div class="card-sub">{% if forecast_3hr >= 120 %}Above Extreme{% elif forecast_3hr >= 89 %}Above Danger{% elif forecast_3hr >= 65 %}Above Warning{% else %}Safe{% endif %}</div>
    </div>
    <div class="forecast-card">
        <div class="forecast-time">In 6 Hours</div>
        <div class="forecast-val {% if forecast_6hr >= 89 %}f-danger{% elif forecast_6hr >= 65 %}f-warn{% else %}f-ok{% endif %}">{{ forecast_6hr }} cm</div>
        <div class="card-sub">{% if forecast_6hr >= 120 %}Above Extreme{% elif forecast_6hr >= 89 %}Above Danger{% elif forecast_6hr >= 65 %}Above Warning{% else %}Safe{% endif %}</div>
    </div>
</div>
<p class="section-title">CWC Threshold Status</p>
<div class="threshold-card">
    <div class="threshold-row">
        <span class="th-label">Warning Level</span>
        <span class="th-level">65 cm</span>
        <span>{{ warning_status }}</span>
        <span class="th-eta">ETA: {{ warning_eta }}</span>
    </div>
    <div class="threshold-row">
        <span class="th-label">Danger Level</span>
        <span class="th-level">89 cm</span>
        <span>{{ danger_status }}</span>
        <span class="th-eta">ETA: {{ danger_eta }}</span>
    </div>
    <div class="threshold-row">
        <span class="th-label">Extreme Level</span>
        <span class="th-level">120 cm</span>
        <span>{{ extreme_status }}</span>
        <span class="th-eta">ETA: {{ extreme_eta }}</span>
    </div>
</div>
<p class="section-title">All Sensor Readings</p>
<div class="grid-4">
    <div class="card"><div class="card-title">Temperature</div><div class="card-value">{{ temperature }} C</div></div>
    <div class="card"><div class="card-title">Humidity</div><div class="card-value">{{ humidity }}%</div></div>
    <div class="card"><div class="card-title">Flow Rate</div><div class="card-value">{{ flow_rate }}</div><div class="card-sub">L/min</div></div>
    <div class="card"><div class="card-title">Turbidity</div><div class="card-value">{{ turbidity }}</div><div class="card-sub">{{ turbidity_desc }}</div></div>
    <div class="card"><div class="card-title">Soil Moisture</div><div class="card-value">{{ soil_moisture }}</div><div class="card-sub">{{ soil_desc }}</div></div>
    <div class="card"><div class="card-title">Pressure</div><div class="card-value">{{ pressure }}</div><div class="card-sub">hPa</div></div>
    <div class="card"><div class="card-title">ML Prediction</div><div class="card-value" style="color:{{ risk_color }};font-size:1.2em;">{{ risk_name }}</div><div class="card-sub">Random Forest</div></div>
    <div class="card"><div class="card-title">CWC Category</div><div class="card-value" style="font-size:1.1em;color:{{ risk_color }};">{{ cwc_category }}</div></div>
</div>
<p class="section-title">Recent History</p>
<div class="history-card">
    <table>
        <tr><th>Time</th><th>Water Level</th><th>Rise Speed</th><th>Forecast 1hr</th><th>Risk</th><th>CWC</th></tr>
        {% for h in history %}
        <tr>
            <td>{{ h.time }}</td>
            <td>{{ h.water_level }} cm</td>
            <td>{{ h.rise_speed }} cm/hr</td>
            <td>{{ h.forecast_1hr }} cm</td>
            <td>{{ h.risk }}</td>
            <td>{{ h.cwc }}</td>
        </tr>
        {% endfor %}
    </table>
</div>
<div class="update-bar">Auto-refreshes every 3 seconds | Warning:65cm | Danger:89cm | Extreme:120cm | Teesta at Domohani CWC</div>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML, **sensor_data,
        history=list(reversed(list(history)))[:20])


@app.route('/data')
def data():
    return jsonify(dict(sensor_data))


@app.route('/history')
def get_history():
    return jsonify(list(history))


if __name__ == '__main__':
    t = threading.Thread(target=read_serial, daemon=True)
    t.start()
    print('Dashboard running at http://raspberrypi.local:5000')
    app.run(host='0.0.0.0', port=5000, debug=False)
