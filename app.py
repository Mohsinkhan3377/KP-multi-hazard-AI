"""
KP Multi-Hazard AI System — deployable Gradio app.

This is the deployment version of multi_hazard_kp.ipynb, adapted to run
as a standalone script (Hugging Face Spaces, Render, or any server) instead
of inside Google Colab.

Changes from the notebook:
- No `!pip install` (handled by requirements.txt)
- No `google.colab.files.upload()` — reads multi_hazard_kp.csv from disk
- No `getpass()` for the Groq key — reads it from the GROQ_API_KEY
  environment variable (set this as a "Secret" in your HF Space settings,
  never commit it to the repo)
- `demo.launch(share=True)` → `demo.launch()` (the host already gives you
  a public URL)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"
REQUEST_TIMEOUT = 20

KP_DISTRICTS = {
    "Peshawar": (34.0151, 71.5249),
    "Charsadda": (34.1454, 71.7409),
    "Nowshera": (34.0153, 71.9747),
    "Mardan": (34.1989, 72.0403),
    "Swabi": (34.1200, 72.4700),
    "Swat": (35.2227, 72.4258),
    "Buner": (34.4833, 72.5333),
    "Shangla": (34.8700, 72.6500),
    "Malakand": (34.5700, 71.9300),
    "Lower Dir": (34.7700, 71.8700),
    "Upper Dir": (35.2100, 71.8800),
    "Abbottabad": (34.1688, 73.2215),
    "Mansehra": (34.3300, 73.2000),
    "Haripur": (33.9964, 72.9347),
    "Battagram": (34.6800, 73.0200),
    "Kohat": (33.5900, 71.4400),
    "Hangu": (33.5300, 71.0600),
    "Karak": (33.1200, 71.0900),
    "Bannu": (32.9900, 70.6000),
    "Lakki Marwat": (32.6100, 70.9100),
    "Dera Ismail Khan": (31.8300, 70.9000),
    "Tank": (32.2200, 70.3800),
}

FEATURES = [
    "rainfall_24h", "rainfall_3day", "rainfall_7day", "max_daily_rain",
    "forecast_rain", "max_temperature", "average_temperature",
    "average_humidity", "soil_moisture", "max_wind_speed",
]
TARGETS = ["heavy_rain", "heatwave", "flood"]

DATASET_PATH = os.path.join(os.path.dirname(__file__), "multi_hazard_kp.csv")


# ---------------------------------------------------------------------------
# Live weather (Open-Meteo)
# ---------------------------------------------------------------------------

def get_weather(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "soil_moisture_0_to_1cm", "wind_speed_10m",
        ],
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
        ],
        "past_days": 7,
        "forecast_days": 2,
        "timezone": "auto",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def create_weather_features(data):
    hourly = data["hourly"]
    daily = data["daily"]

    temperature = pd.Series(hourly["temperature_2m"]).dropna()
    humidity = pd.Series(hourly["relative_humidity_2m"]).dropna()
    precipitation = pd.Series(hourly["precipitation"]).dropna()
    soil = pd.Series(hourly["soil_moisture_0_to_1cm"]).dropna()
    wind = pd.Series(hourly["wind_speed_10m"]).dropna()
    daily_rain = pd.Series(daily["precipitation_sum"]).dropna()
    daily_max_temp = pd.Series(daily["temperature_2m_max"]).dropna()

    return {
        "rainfall_24h": precipitation.tail(24).sum(),
        "rainfall_3day": precipitation.tail(72).sum(),
        "rainfall_7day": precipitation.sum(),
        "max_daily_rain": daily_rain.max(),
        "forecast_rain": daily_rain.tail(2).sum(),
        "max_temperature": daily_max_temp.max(),
        "average_temperature": temperature.mean(),
        "average_humidity": humidity.mean(),
        "soil_moisture": soil.mean(),
        "max_wind_speed": wind.max(),
    }


# ---------------------------------------------------------------------------
# Load + clean training data
# ---------------------------------------------------------------------------

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

missing_columns = [c for c in FEATURES + TARGETS if c not in df.columns]
if missing_columns:
    raise RuntimeError(f"Dataset is missing required columns: {missing_columns}")

model_df = df[FEATURES + TARGETS].copy()
for column in FEATURES:
    model_df[column] = pd.to_numeric(model_df[column], errors="coerce")
for target in TARGETS:
    model_df[target] = pd.to_numeric(model_df[target], errors="coerce")

for column in FEATURES:
    model_df[column] = model_df[column].fillna(model_df[column].median())
for target in TARGETS:
    model_df = model_df.dropna(subset=[target])
    model_df[target] = model_df[target].astype(int)

print(f"Dataset ready: {model_df.shape}")


# ---------------------------------------------------------------------------
# Train models (runs once at startup)
# ---------------------------------------------------------------------------

def train_models(data, target):
    X = data[FEATURES]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000)),
    ])
    random_forest = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    xgboost = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42,
    )

    models = {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
        "XGBoost": xgboost,
    }
    for model in models.values():
        model.fit(X_train, y_train)

    return models, X_train, X_test, y_train, y_test


def evaluate_models(models, X_test, y_test):
    results = []
    for name, model in models.items():
        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, zero_division=0),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "F1": f1_score(y_test, prediction, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, probability),
        })
    return pd.DataFrame(results).sort_values("F1", ascending=False)


print("Training Heavy Rain models...")
rain_models_dict, _, rain_X_test, _, rain_y_test = train_models(model_df, "heavy_rain")
rain_results = evaluate_models(rain_models_dict, rain_X_test, rain_y_test)
best_rain_model = rain_models_dict[rain_results.iloc[0]["Model"]]
print("Best Rain model:", rain_results.iloc[0]["Model"])

print("Training Heatwave models...")
heat_models_dict, _, heat_X_test, _, heat_y_test = train_models(model_df, "heatwave")
heat_results = evaluate_models(heat_models_dict, heat_X_test, heat_y_test)
best_heat_model = heat_models_dict[heat_results.iloc[0]["Model"]]
print("Best Heat model:", heat_results.iloc[0]["Model"])

print("Training Flood models...")
flood_models_dict, _, flood_X_test, _, flood_y_test = train_models(model_df, "flood")
flood_results = evaluate_models(flood_models_dict, flood_X_test, flood_y_test)
best_flood_model = flood_models_dict[flood_results.iloc[0]["Model"]]
print("Best Flood model:", flood_results.iloc[0]["Model"])

print("All models trained.")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def prepare_live_input(features):
    row = pd.DataFrame([features])
    return row[FEATURES]


def risk_level(probability):
    if probability < 0.25:
        return "Low"
    elif probability < 0.50:
        return "Moderate"
    elif probability < 0.75:
        return "High"
    else:
        return "Severe"


def predict_all_hazards(features):
    input_data = prepare_live_input(features)

    rain_probability = float(best_rain_model.predict_proba(input_data)[0][1])
    heat_probability = float(best_heat_model.predict_proba(input_data)[0][1])
    flood_probability = float(best_flood_model.predict_proba(input_data)[0][1])

    return {
        "heavy_rain": {"probability": rain_probability, "risk": risk_level(rain_probability)},
        "heatwave": {"probability": heat_probability, "risk": risk_level(heat_probability)},
        "flood": {"probability": flood_probability, "risk": risk_level(flood_probability)},
    }


# ---------------------------------------------------------------------------
# Groq explanation layer
# ---------------------------------------------------------------------------

def explain_with_groq(district, features, predictions):
    if not GROQ_API_KEY:
        return "⚠️ Groq explanation unavailable — no GROQ_API_KEY configured on this server."

    prompt = f"""
You are an AI assistant for an educational multi-hazard environmental risk system.

District:
{district}

Machine-learning predictions:

Heavy Rain: {predictions['heavy_rain']['probability']:.2%} — Risk: {predictions['heavy_rain']['risk']}
Heatwave: {predictions['heatwave']['probability']:.2%} — Risk: {predictions['heatwave']['risk']}
Flood: {predictions['flood']['probability']:.2%} — Risk: {predictions['flood']['risk']}

Weather data:
7-day rainfall: {features['rainfall_7day']:.2f} mm
24-hour rainfall: {features['rainfall_24h']:.2f} mm
Forecast rainfall: {features['forecast_rain']:.2f} mm
Maximum temperature: {features['max_temperature']:.2f} C
Average humidity: {features['average_humidity']:.2f} %
Soil moisture: {features['soil_moisture']:.3f}
Wind: {features['max_wind_speed']:.2f} km/h

Explain the results in simple language. Mention which hazards have the
highest risk and which weather factors contributed. Do not claim that a
disaster will definitely happen. Clearly state that this is an ML-based
preliminary assessment and not an official emergency warning.
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You explain ML environmental risk results clearly."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 250,
    }

    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Groq explanation failed: {e}"


def run_hazard_assessment(district):
    lat, lon = KP_DISTRICTS[district]
    weather = get_weather(lat, lon)
    features = create_weather_features(weather)
    predictions = predict_all_hazards(features)
    explanation = explain_with_groq(district, features, predictions)
    return {
        "district": district,
        "features": features,
        "predictions": predictions,
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def dashboard(district):
    try:
        result = run_hazard_assessment(district)
        predictions = result["predictions"]

        table = pd.DataFrame({
            "Hazard": ["🌧️ Heavy Rain", "🔥 Heatwave", "🌊 Flood"],
            "Probability": [
                predictions["heavy_rain"]["probability"],
                predictions["heatwave"]["probability"],
                predictions["flood"]["probability"],
            ],
            "Risk": [
                predictions["heavy_rain"]["risk"],
                predictions["heatwave"]["risk"],
                predictions["flood"]["risk"],
            ],
        })

        colors = {"Low": "#2ecc71", "Moderate": "#f1c40f", "High": "#e67e22", "Severe": "#e74c3c"}
        bar_colors = [colors[r] for r in table["Risk"]]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(table["Hazard"], table["Probability"], color=bar_colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel("ML Probability")
        ax.set_title(f"Multi-Hazard Risk — {district}")
        fig.tight_layout()

        summary = (
            f"### 📍 {district} — Multi-Hazard Assessment\n\n"
            f"🌧️ Heavy Rain: **{predictions['heavy_rain']['risk']}** "
            f"({predictions['heavy_rain']['probability']:.1%})\n\n"
            f"🔥 Heatwave: **{predictions['heatwave']['risk']}** "
            f"({predictions['heatwave']['probability']:.1%})\n\n"
            f"🌊 Flood: **{predictions['flood']['risk']}** "
            f"({predictions['flood']['probability']:.1%})"
        )

        return summary, table, fig, result["explanation"]

    except Exception as e:
        error_msg = f"⚠️ Error while processing **{district}**:\n\n`{str(e)}`"
        empty_table = pd.DataFrame({"Hazard": [], "Probability": [], "Risk": []})
        empty_fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No data (error occurred)", ha="center", va="center")
        return error_msg, empty_table, empty_fig, "N/A"


def compare_districts(selected_districts):
    if not selected_districts or len(selected_districts) < 2:
        empty_fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Select at least 2 districts to compare", ha="center", va="center")
        return "Select at least 2 districts.", pd.DataFrame(), empty_fig

    rows = []
    for d in selected_districts:
        try:
            lat, lon = KP_DISTRICTS[d]
            weather = get_weather(lat, lon)
            features = create_weather_features(weather)
            predictions = predict_all_hazards(features)

            rows.append({
                "District": d,
                "Heavy Rain": predictions["heavy_rain"]["probability"],
                "Heatwave": predictions["heatwave"]["probability"],
                "Flood": predictions["flood"]["probability"],
                "Highest Risk Hazard": max(
                    predictions, key=lambda h: predictions[h]["probability"]
                ).replace("_", " ").title(),
            })
        except Exception as e:
            rows.append({
                "District": d, "Heavy Rain": None, "Heatwave": None,
                "Flood": None, "Highest Risk Hazard": f"Error: {e}",
            })

    comp_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(max(8, len(selected_districts) * 1.5), 5))
    x = np.arange(len(comp_df))
    width = 0.25

    ax.bar(x - width, comp_df["Heavy Rain"], width, label="Heavy Rain", color="#3498db")
    ax.bar(x, comp_df["Heatwave"], width, label="Heatwave", color="#e67e22")
    ax.bar(x + width, comp_df["Flood"], width, label="Flood", color="#2980b9")

    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["District"], rotation=20, ha="right")
    ax.set_ylabel("ML Probability")
    ax.set_ylim(0, 1)
    ax.set_title("District Comparison — Multi-Hazard Risk")
    ax.legend()
    fig.tight_layout()

    riskiest = comp_df.loc[comp_df[["Heavy Rain", "Heatwave", "Flood"]].max(axis=1).idxmax(), "District"]
    summary = f"### 📊 Comparing {len(selected_districts)} districts\n\n🔴 Highest overall risk right now: **{riskiest}**"

    return summary, comp_df, fig


custom_css = """
#header-box {
    background: linear-gradient(135deg, #1e3a5f 0%, #2c5f8a 100%);
    padding: 28px 32px;
    border-radius: 12px;
    color: white !important;
    margin-bottom: 20px;
}
#header-box h1, #header-box h3, #header-box p { color: white !important; margin: 4px 0; }
.gradio-container { background-color: #f7f8fa !important; font-family: 'Segoe UI', Roboto, Arial, sans-serif !important; }
.gr-button-primary { background: #2c5f8a !important; border: none !important; font-weight: 600 !important; }
.tabitem, .gr-box, .gr-panel { border-radius: 10px !important; box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important; }
.dark-mode { background-color: #0f0f0f !important; color: #f0f0f0 !important; }
.dark-mode #header-box { background: linear-gradient(135deg, #111 0%, #222 100%) !important; }
.dark-mode .gr-box, .dark-mode .gr-panel, .dark-mode table { background-color: #1a1a1a !important; color: #f0f0f0 !important; }
"""

toggle_js = """
() => { document.querySelector('.gradio-container').classList.toggle('dark-mode'); }
"""

theme = gr.themes.Soft(
    primary_hue="blue", secondary_hue="slate", neutral_hue="slate",
).set(
    body_background_fill="#f7f8fa",
    block_background_fill="white",
    block_border_width="1px",
    block_shadow="*shadow_drop_lg",
)

with gr.Blocks(title="KP Multi-Hazard AI System") as demo:
    with gr.Row(elem_id="header-box"):
        with gr.Column(scale=8):
            gr.Markdown(
                """
# 🌍 KP Multi-Hazard AI System
### Machine Learning · Live Weather · AI Explanations
Preliminary risk assessment for Heavy Rain, Heatwave, and Flood across Khyber Pakhtunkhwa.
"""
            )
        with gr.Column(scale=2, min_width=120):
            theme_button = gr.Button("🌓 Toggle Theme", size="sm")

    gr.Markdown("⚠️ *Educational research system — not an official emergency warning service.*")

    theme_button.click(fn=None, js=toggle_js)

    with gr.Tabs():
        with gr.Tab("📍 District Assessment"):
            with gr.Row():
                with gr.Column(scale=1):
                    district = gr.Dropdown(
                        choices=list(KP_DISTRICTS.keys()), value="Peshawar", label="Select District"
                    )
                    run_button = gr.Button("Run Assessment", variant="primary")
                    summary = gr.Markdown()
                with gr.Column(scale=2):
                    chart = gr.Plot(label="Risk Probabilities")

            with gr.Row():
                table = gr.Dataframe(label="Model Predictions")

            with gr.Accordion("🧠 AI Explanation", open=True):
                explanation = gr.Markdown()

            run_button.click(fn=dashboard, inputs=district, outputs=[summary, table, chart, explanation])

        with gr.Tab("📊 Compare Districts"):
            with gr.Row():
                with gr.Column(scale=1):
                    districts_multi = gr.CheckboxGroup(
                        choices=list(KP_DISTRICTS.keys()),
                        value=["Peshawar", "Nowshera", "Charsadda"],
                        label="Select Districts",
                    )
                    compare_button = gr.Button("Compare", variant="primary")
                    compare_summary = gr.Markdown()
                with gr.Column(scale=2):
                    compare_chart = gr.Plot(label="Side-by-Side Comparison")

            with gr.Row():
                compare_table = gr.Dataframe(label="Comparison Table")

            compare_button.click(
                fn=compare_districts, inputs=districts_multi,
                outputs=[compare_summary, compare_table, compare_chart],
            )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=theme,
        css=custom_css
    )