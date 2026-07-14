# Traffic Prediction Using Machine Learning

A machine learning system that predicts traffic congestion levels (**High**, **Moderate**, **Low**) using real-world traffic and weather-related features. The project includes a full ML pipeline data preprocessing, model training, evaluation, explainability (SHAP & LIME), and a Streamlit web app for real-time predictions.


---

## Overview

Traffic congestion in metropolitan areas leads to longer travel times, higher fuel consumption, and increased pollution. This project builds a reliable traffic level prediction system using classical machine learning models, trained on a dataset combining road, weather, and distance-based features and integrates live data from real-world APIs.


---

## Tech Stack

**Language:** Python

**Core ML & Data Processing:**
- `scikit-learn` model training & evaluation
- `pandas`, `numpy` data preprocessing
- `StandardScaler`, One-Hot & Label Encoding feature engineering

**Explainability:**
- `SHAP` (SHapley Additive exPlanations)
- `LIME` (Local Interpretable Model-agnostic Explanations)

**Web App / Deployment:**
- `Streamlit` — interactive web interface

**Real-Time Data Sources:**
- [TomTom Traffic API](https://developer.tomtom.com/) live traffic data
- [OpenWeather API](https://openweathermap.org/api) real-time weather data

---

## Dataset

- **Source:** `Trafficdatasetwith_levels.csv`
- **Total Records:** 1001
- **Target Classes:** `Traffic_level` → High, Moderate, Low

**Features used:**
| Feature | Description |
|---|---|
| `Weather_Condition` | Rain, Fog, Clear, Snow |
| `Road_type` | Arterial, Local, Highway |
| `Distance_meters` | Distance covered |
| `Longitude` / `Latitude` | Location coordinates |
| `Vehicle_Count` | Scaled using StandardScaler |

**Preprocessing steps:**
- Missing value imputation (mode/median)
- One-Hot Encoding for categorical features
- Label Encoding for target variable
- Feature scaling (StandardScaler)
- 80/20 Train-Test split

---

## Models Trained

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.84 | 0.85 | 0.84 | 0.83 |
| Decision Tree | 0.87 | 0.86 | 0.88 | 0.87 |
| **Random Forest** | **0.91** | **0.92** | **0.91** | **0.91** |
| SVM | 0.89 | 0.88 | 0.89 | 0.88 |
| K-Nearest Neighbors | 0.83 | 0.82 | 0.83 | 0.82 |

**Best Model:** Random Forest Classifier (`n_estimators=100`, `random_state=42`)  **91% accuracy**

---

## Explainability

To make the "black-box" Random Forest model interpretable:
- **SHAP (TreeExplainer)**: used for global & local feature importance (summary plots, waterfall plots). Found that `Vehicle_Count` is the strongest driver of traffic level, with `Weather_Condition` (Rain/Fog) adding to congestion.
- **LIME (LimeTabularExplainer)**: used for local, per-sample explanations to validate model reasoning on individual predictions.

---

## Web Application

A working **Streamlit** app has been built that:
- Accepts live inputs and fetches real-time traffic & weather data via TomTom & OpenWeather APIs
- Predicts traffic level (High/Moderate/Low) instantly
- Visualizes prediction confidence

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### API Keys Setup
Create a `.env` file in the root directory and add your API keys:
```env
TOMTOM_API_KEY=your_tomtom_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

---


---

## Ethical Considerations & Limitations

- Dataset contains no personally identifiable information; real-time GPS/user data would require GDPR-level compliance.
- Model fairness must be validated across diverse regions and traffic patterns before deployment.
- Not yet integrated with live accident/roadblock feeds predictions may not capture sudden anomalies.

---

---

## Team

| Name 
|---|---|
| Abdur Rehman 
| Talha Noor 
| Ali Rana 
| Sufiyan Nadeem 


---

## License

This project is for academic purposes. Feel free to fork and build upon it with attribution.
