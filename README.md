# 🌍 KP Multi-Hazard AI System

An AI-powered early warning system for **Khyber Pakhtunkhwa** that predicts
**heavy rain**, **heatwave**, and **flood** risk for every KP district using
live weather data, trained machine learning models, and an LLM explanation layer.

> ⚠️ This is an educational research project, not an official emergency warning service.

## How it works

```
Historical Dataset
        │
   Data Cleaning
        │
 Feature Engineering
        │
 ┌──────┼──────┐
 ▼      ▼      ▼
Rain    Heat   Flood     → Logistic Regression / Random Forest / XGBoost
 ML      ML      ML         (best model picked per hazard by F1 score)
 └──────┼──────┘
        ▼
  Live Weather API (Open-Meteo)
        │
  ML Risk Prediction
        │
  Gradio Dashboard  +  Groq LLM Explanation
```

**ML models predict. Groq only explains the prediction in plain language —
it never makes the prediction itself.**

## Features

- Covers all **22 KP districts**
- Three hazard models trained independently (Heavy Rain, Heatwave, Flood)
- Each model evaluated with Accuracy, Precision, Recall, F1, and ROC-AUC
- Live weather features pulled from the [Open-Meteo API](https://open-meteo.com/)
- Interactive **Gradio dashboard** with:
  - Single-district risk assessment
  - Multi-district comparison view
  - Color-coded risk charts (Low → Severe)
  - Dark/light theme toggle
- Plain-language risk explanations generated via **Groq** (`openai/gpt-oss-120b`)

## Dataset

`multi_hazard_kp.csv` — daily weather + hazard-label data for all 22 KP
districts (2021–2023).

**Important:** this dataset is **synthetic**, generated to match realistic
seasonal weather patterns and terrain differences across KP. Hazard labels
(`heavy_rain`, `heatwave`, `flood`) were derived from documented meteorological
thresholds, not invented arbitrarily — but they are not real historical
records. Replace it with verified data from PMD, PDMA KP, or NDMA before
using this for anything beyond a learning/demo project.

| Column | Description |
|---|---|
| `date`, `district`, `latitude`, `longitude` | Identifiers |
| `rainfall_24h`, `rainfall_3day`, `rainfall_7day`, `max_daily_rain`, `forecast_rain` | Rainfall features (mm) |
| `max_temperature`, `average_temperature`, `average_humidity` | Temperature/humidity features |
| `soil_moisture`, `max_wind_speed` | Additional weather features |
| `heavy_rain`, `heatwave`, `flood` | Hazard labels (0/1) |

## Setup

```bash
pip install -r requirements.txt
```

Open `multi_hazard_kp.ipynb` in Jupyter or Google Colab and run the cells
in order. You'll be prompted for a **Groq API key** (get one free at
[console.groq.com](https://console.groq.com)) — it's entered securely via
`getpass` and never stored in the notebook.

To use your own data, replace `multi_hazard_kp.csv` with a file containing
the same columns.

## Tech stack

Python · pandas · scikit-learn · XGBoost · Gradio · Open-Meteo API · Groq API

## Author

Mohsin — AKTI AI/ML Cohort (Trainer: Faiza Ghafar)

## Disclaimer

This project is for educational purposes. Predictions are preliminary and
should not be relied upon as an official flood, heat, or weather warning.
For real emergencies, always follow guidance from PDMA KP / NDMA.
