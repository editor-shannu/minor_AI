# medicine_sales_prediction/app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# ✅ FIX project root imports
sys.path.insert(0, str(Path(__file__).parent))

from src.model import MedicineSalesModel

st.set_page_config(page_title="Medicine AI Suite", layout="wide")
st.title("🏥 Medicine AI Suite (All-in-One)")
st.caption("Regression • Classification • Clustering • Anomaly • Forecasting • SHAP • Trends • BERT")


# -------------------------------------------------
# ✅ Load Dataset (TWO CSV FILES + MERGE)
# -------------------------------------------------
@st.cache_data
def load_data():
    disease_path = "data/disease_cases.csv"
    sales_path = "data/medicine_sales.csv"

    if not Path(disease_path).exists():
        st.error(f"❌ File not found: `{disease_path}`")
        st.stop()

    if not Path(sales_path).exists():
        st.error(f"❌ File not found: `{sales_path}`")
        st.stop()

    disease_df = pd.read_csv(disease_path)
    sales_df = pd.read_csv(sales_path)

    # remove unwanted index column
    if "Unnamed: 0" in disease_df.columns:
        disease_df = disease_df.drop(columns=["Unnamed: 0"])
    if "Unnamed: 0" in sales_df.columns:
        sales_df = sales_df.drop(columns=["Unnamed: 0"])

    # parse date
    disease_df["date"] = pd.to_datetime(disease_df["date"], errors="coerce")
    sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce")

    # ✅ merge
    df = pd.merge(disease_df, sales_df, on=["date", "location"], how="inner")

    # month/year
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    # disease_trend
    df["disease_trend"] = df.groupby("location")["cases"].pct_change().fillna(0)

    return df


df = load_data()


# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
st.sidebar.header("⚙️ Control Panel")

mode = st.sidebar.selectbox(
    "Choose AI Module",
    [
        "Sales Prediction (Regression)",
        "Sales Risk (Classification)",
        "Clustering",
        "Anomaly Detection",
        "Forecasting (SARIMA)",
        "Explainability (SHAP)",
        "Google Trends Analysis",
        "BERT Sentiment Analysis",
    ]
)

st.sidebar.markdown("---")
show_data = st.sidebar.checkbox("📊 Show dataset preview", value=False)

if show_data:
    st.subheader("📊 Dataset Preview (Merged from 2 CSV files)")
    st.dataframe(df.head(200), use_container_width=True)
    st.write("Columns:", list(df.columns))
    st.write("Total rows:", len(df))


cities = sorted(df["location"].dropna().unique().tolist())


# -------------------------------------------------
# Helper UI Input
# -------------------------------------------------
def build_input_df():
    st.subheader("🧾 Input Features")

    col1, col2, col3 = st.columns(3)
    with col1:
        cases = st.slider("🦠 Cases", int(df["cases"].min()), int(df["cases"].max()), int(df["cases"].median()))
    with col2:
        month = st.slider("📅 Month", 1, 12, int(df["month"].mode()[0]))
    with col3:
        trend = st.slider("📈 Trend %", -200.0, 200.0, 0.0) / 100

    location = st.selectbox("🏙️ Location", cities)

    year = st.slider("📆 Year", int(df["year"].min()), int(df["year"].max()), int(df["year"].max()))

    return pd.DataFrame({
        "cases": [cases],
        "disease_trend": [trend],
        "month": [month],
        "year": [year],
        "location": [location],
    })


# -------------------------------------------------
# MODE 1: Regression
# -------------------------------------------------
if mode == "Sales Prediction (Regression)":
    st.header("✅ Sales Prediction (Regression)")

    model_choice = st.selectbox("Select Model", ["linear", "rf"])
    model = MedicineSalesModel(model_type=model_choice)

    input_df = build_input_df()

    if st.button("🚀 Train + Predict", type="primary", use_container_width=True):
        stats = model.train_regression(df)
        pred = model.predict_regression(input_df)[0]
        st.success(f"🎯 Predicted Sales Volume: **{pred:.0f} units**")
        st.json(stats)


# -------------------------------------------------
# MODE 2: Classification
# -------------------------------------------------
elif mode == "Sales Risk (Classification)":
    st.header("✅ Sales Risk (Classification)")

    model = MedicineSalesModel(model_type="logistic")
    input_df = build_input_df()

    if st.button("🚀 Train + Predict Risk", type="primary", use_container_width=True):
        stats = model.train_classification(df)
        risk = model.predict_classification(input_df)[0]

        risk_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
        st.success(f"📌 Predicted Sales Risk: **{risk_map.get(int(risk))}**")
        st.json(stats)


# -------------------------------------------------
# MODE 3: Clustering
# -------------------------------------------------
elif mode == "Clustering":
    st.header("✅ Clustering")

    algo = st.selectbox("Clustering Algorithm", ["kmeans", "dbscan"])
    model = MedicineSalesModel(model_type=algo)

    if st.button("🚀 Run Clustering", type="primary", use_container_width=True):
        labels = model.fit_clustering(df)
        df_out = df.copy()
        df_out["cluster"] = labels
        st.success("✅ Done")
        st.dataframe(df_out.head(200), use_container_width=True)


# -------------------------------------------------
# MODE 4: Anomaly
# -------------------------------------------------
elif mode == "Anomaly Detection":
    st.header("✅ Anomaly Detection")

    contamination = st.slider("Contamination", 0.01, 0.20, 0.02)

    model = MedicineSalesModel(model_type="isolation")
    model.anomaly_detector.set_params(contamination=float(contamination))

    if st.button("🚀 Detect", type="primary", use_container_width=True):
        labels = model.fit_anomaly(df)
        df_out = df.copy()
        df_out["anomaly"] = labels
        st.dataframe(df_out[df_out["anomaly"] == -1].head(200), use_container_width=True)


# -------------------------------------------------
# MODE 5: Forecasting (SARIMA)
# -------------------------------------------------
elif mode == "Forecasting (SARIMA)":
    st.header("✅ Forecasting (SARIMA)")
    st.info("Install: pip install statsmodels")

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as e:
        st.error(e)
        st.stop()

    value_col = st.selectbox("Value column", ["sales_volume", "cases"])
    steps = st.slider("Forecast days", 7, 180, 30)

    series = df.groupby("date")[value_col].sum().sort_index()

    if st.button("📈 Forecast", type="primary", use_container_width=True):
        model_ts = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                           enforce_stationarity=False)
        res = model_ts.fit(disp=False)
        forecast = res.forecast(steps=steps)

        st.dataframe(forecast.reset_index().rename(columns={"index": "date", 0: "forecast"}),
                     use_container_width=True)

        fig = plt.figure()
        plt.plot(series.index, series.values, label="history")
        plt.plot(forecast.index, forecast.values, label="forecast")
        plt.legend()
        st.pyplot(fig)


# -------------------------------------------------
# MODE 6: SHAP
# -------------------------------------------------
elif mode == "Explainability (SHAP)":
    st.header("✅ SHAP Explainability")
    st.warning("Install: pip install shap")

    import shap

    model = MedicineSalesModel(model_type="rf")

    if st.button("🚀 Train + Explain", type="primary", use_container_width=True):
        model.train_regression(df)

        X, _ = model.prepare_features(df, predict_mode=True)
        Xs = X[:100]

        explainer = shap.Explainer(model.predictor.model)
        shap_values = explainer(Xs)

        fig = plt.figure()
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig)


# -------------------------------------------------
# MODE 7: Google Trends
# -------------------------------------------------
elif mode == "Google Trends Analysis":
    st.header("✅ Google Trends Analysis")
    st.warning("Install: pip install pytrends")

    from pytrends.request import TrendReq

    keywords = st.text_input("Keywords", "fever,cough,dengue")
    timeframe = st.text_input("Timeframe", "today 5-y")

    if st.button("🌍 Fetch Trends", type="primary", use_container_width=True):
        kw = [k.strip() for k in keywords.split(",") if k.strip()]
        pytrends = TrendReq(hl="en-US", tz=330)
        pytrends.build_payload(kw, timeframe=timeframe, geo="IN")
        trends_df = pytrends.interest_over_time()

        if "isPartial" in trends_df.columns:
            trends_df = trends_df.drop(columns=["isPartial"])

        st.dataframe(trends_df.tail(100), use_container_width=True)


# -------------------------------------------------
# MODE 8: BERT
# -------------------------------------------------
elif mode == "BERT Sentiment Analysis":
    st.header("✅ BERT Sentiment Analysis")
    st.warning("Install: pip install transformers torch")

    from transformers import pipeline

    txt = st.text_area("Enter text", "Medicine worked great!\nSide effects are bad.")
    texts = [t.strip() for t in txt.split("\n") if t.strip()]

    if st.button("🧠 Analyze", type="primary", use_container_width=True):
        pipe = pipeline("sentiment-analysis")
        results = pipe(texts)

        out = pd.DataFrame({
            "text": texts,
            "sentiment": [r["label"] for r in results],
            "score": [r["score"] for r in results],
        })
        st.dataframe(out, use_container_width=True)

