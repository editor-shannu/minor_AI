# Medicine Sales Prediction

Predicts medicine sales based on prior disease patterns using  ML pipeline.

Predicts **medicine sales demand** using **disease outbreak patterns**, with an interactive **Streamlit dashboard** supporting Regression, Classification, Clustering, Anomaly Detection, Forecasting, Explainability, Trends & NLP.

📄 **Project Report**: [View Report](YOUR_REPORT_LINK)  
📊 **Live Dashboard**: [Launch Dashboard](YOUR_STREAMLIT_LINK)  
🎥 **Video Demo & Presentation**: [Watch Demo](YOUR_VIDEO_LINK)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Model Performance](#-model-performance)
- [Technical Details](#-technical-details)
- [Disclaimer](#-disclaimer)
- [License](#-license)
- [Support / Contact](#-support--contact)

---

## 🎯 Overview

This project demonstrates how to build an **end-to-end AI system** for predicting **medicine sales volume** using disease outbreak information. It is designed for AIML students and beginners but includes modern AI modules like SHAP, SARIMA forecasting, Google Trends signals, and BERT sentiment analysis.

### ✅ What it does
- Loads **two real datasets**
  - `disease_cases.csv` → disease outbreak cases
  - `medicine_sales.csv` → medicine sales volume
- Merges datasets using `date + location`
- Generates outbreak features like:
  - `month`, `year`
  - `disease_trend` (case growth trend)
- Trains multiple ML models in one project:
  - Linear Regression, Random Forest
  - Logistic Regression (Risk prediction)
  - KMeans, DBSCAN (Clustering)
  - Isolation Forest (Anomaly detection)
  - SARIMA Forecasting
- Interactive dashboard using Streamlit

🎯 **Target Audience:** Students (Minor/Major project), beginners in ML/Healthcare analytics, applied AI learners

---

## ✨ Features

### 🔄 Dataset Handling
- Uses **two CSV datasets**
- Automatic merge using:
  - `date`
  - `location`
- Converts date into:
  - Month / Year features
- Auto-calculates:
  - `disease_trend` (growth % based on cases)

### 🤖 Machine Learning Suite (All-in-One Dashboard)

✅ **Sales Prediction (Regression)**
- Linear Regression
- Random Forest Regressor  
📌 Output: predicted `sales_volume`

✅ **Sales Risk (Classification)**
- Logistic Regression  
📌 Output: Low / Medium / High risk levels

✅ **Clustering**
- KMeans
- DBSCAN  
📌 Output: clusters of cities/regions based on case-sales patterns

✅ **Anomaly Detection**
- Isolation Forest  
📌 Detects abnormal spikes in disease cases or sales volume

✅ **Forecasting**
- SARIMA (SARIMAX) Time-series Forecasting  
📌 Forecast future `sales_volume` or `cases`

✅ **Explainability**
- SHAP feature importance visualization  
📌 Explains model predictions

✅ **External Signals**
- Google Trends analysis using PyTrends  
📌 Tracks keyword interest for outbreak-related terms

✅ **NLP**
- BERT Sentiment Analysis using Transformers  
📌 Analyzes public/user feedback sentiment

---

## 🚀 Quick Start

### 1) Clone & Setup
bash
git clone https://github.com/editor-shannu/minor_AI
cd medicine_sales_prediction
python -m venv env

2) Activate Virtual Environment

✅ Windows:

env\Scripts\activate


✅ macOS/Linux:

source env/bin/activate

3) Install Dependencies
pip install -r requirements.txt

4) Place Dataset CSV files

Put both CSV files inside:

medicine_sales_prediction/data/
├── disease_cases.csv
└── medicine_sales.csv

5) Launch Dashboard
streamlit run app.py


Open browser:

http://localhost:8501

## 📁 Project Structure

```bash
medicine_sales_prediction/
├── configs/
│   ├── data.yaml
│   └── model.yaml
├── data/
│   ├── disease_cases.csv
│   └── medicine_sales.csv
├── models/
│   └── medicine_sales_model.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_predictions.ipynb
├── scripts/
│   ├── preprocess.py
│   └── evaluate.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   └── train.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

```

🛠 Installation
Prerequisites

Python 3.9+ (recommended: 3.10)

Git

Step-by-Step Installation

Clone

git clone https://github.com/editor-shannu/minor_AI
cd medicine_sales_prediction


Create virtual environment

python -m venv env


Activate
✅ Windows:

env\Scripts\activate


✅ macOS/Linux:

source env/bin/activate


Install packages

pip install -r requirements.txt

📊 Model Performance

Model performance depends on dataset quality and city-wise patterns.

Typical metrics shown inside dashboard:

✅ Regression: Train R² / Test R²

✅ Classification: Accuracy

✅ Clustering: Cluster distribution

✅ Anomaly: Count of anomalies detected

✅ Forecasting: Visual forecast trends

🔬 Technical Details
Data Pipeline

Load disease data (disease_cases.csv)

Load sales data (medicine_sales.csv)

Parse and normalize date column

Merge using:

date

location

Feature engineering:

month

year

disease_trend = pct_change(cases) grouped by location

Train models using OOP pipeline

Features Used (ML Input)

cases

disease_trend

month

year

location_encoded

Target

sales_volume

🚨 Disclaimer

IMPORTANT: This project is for educational purposes only.

📚 Designed for learning ML and healthcare analytics

❌ Not recommended for real-world medical inventory decisions

📊 Predictions depend on dataset patterns

👨‍⚕️ Consult experts for real deployments

📄 License

This project is licensed under the MIT License.

🙏 Acknowledgments

Streamlit for dashboard UI

scikit-learn for ML models

statsmodels for SARIMA forecasting

SHAP for explainability

PyTrends for Google Trends analysis

HuggingFace Transformers for BERT sentiment analysis

📞 Support / Contact

👤 Author: Shanmukha Sri Saikumar Medisetty
📧 Email: medisettyshanmukh@gmail.com
🌐 GitHub: https://github.com/editor-shannu
