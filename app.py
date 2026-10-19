import os, sys, json, joblib, numpy as np, pandas as pd, plotly.express as px, streamlit as st
os.chdir(os.path.dirname(os.path.abspath(__file__)))
st.set_page_config(page_title="Fraud Command Center", layout="wide")

@st.cache_data
def load():
    return pd.read_csv("processed_transactions.csv.gz"), pd.read_csv("hotspots.csv")
df, hot = load()
CATS = sorted(df.merchant_category.unique())
view = st.sidebar.radio("View", ["Overview", "Live Feed", "Risk Map", "Historical Analytics", "Merchant-City Matrix", "ML Scoring Simulator"])

if view == "Overview":
    a, b, c = st.columns(3)
    a.metric("Transactions", f"{len(df):,}"); b.metric("Fraud rate", f"{df.is_fraud.mean()*100:.2f}%")
    c.metric("HIGH-risk", f"{(df.risk_level == 'HIGH').sum():,}")
    st.dataframe(df.risk_level.value_counts().rename("count"))
elif view == "Live Feed":
    p = "live_predictions.jsonl"
    if os.path.exists(p):
        st.dataframe(pd.read_json(p, lines=True).tail(50))
    else:
        st.info("Run the streaming producer and processor first.")
elif view == "Risk Map":
    g = df.groupby("city").agg(lat=("latitude", "mean"), lon=("longitude", "mean"), fraud=("is_fraud", "mean"), n=("is_fraud", "size")).reset_index()
    g["fraud_pct"] = g.fraud * 100
    st.plotly_chart(px.scatter_geo(g, lat="lat", lon="lon", size="n", color="fraud_pct", hover_name="city",
                                   scope="asia", color_continuous_scale="YlOrRd"), use_container_width=True)
elif view == "Historical Analytics":
    h = (df.groupby("hour_of_day").is_fraud.mean() * 100).reset_index()
    st.plotly_chart(px.line(h, x="hour_of_day", y="is_fraud", markers=True, title="Fraud % by hour"), use_container_width=True)
    ch = (df.groupby("channel").is_fraud.mean() * 100).reset_index()
    st.plotly_chart(px.bar(ch, x="channel", y="is_fraud", title="Fraud % by channel"), use_container_width=True)
elif view == "Merchant-City Matrix":
    pv = (df.pivot_table(index="city", columns="channel", values="is_fraud", aggfunc="mean") * 100).round(2)
    st.dataframe(pv); st.subheader("Top hotspot merchants"); st.dataframe(hot)
else:
    model = joblib.load("rf.joblib"); cols = list(model.feature_names_in_)
    c1, c2 = st.columns(2)
    amount = c1.number_input("Amount (INR)", 10.0, 500000.0, 5000.0)
    median = c1.number_input("Customer median amount (INR)", 10.0, 500000.0, 5000.0)
    dist = c1.slider("Distance from home (km)", 0, 1500, 10)
    vel = c2.slider("Transactions in last hour", 0, 10, 0)
    hr = c2.slider("Hour of day", 0, 23, 12); online = c2.checkbox("Online"); foreign = c2.checkbox("Foreign")
    cat = c1.selectbox("Merchant category", CATS)
    row = {"amount": amount, "amount_ratio": amount / median, "distance_from_home_km": dist, "txn_count_1h": vel,
           "hour_of_day": hr, "day_of_week": 2, "is_weekend": 0, "is_night": int(0 <= hr <= 4),
           "is_online": int(online), "is_foreign": int(foreign)}
    for c in cols:
        if c.startswith("cat_"): row[c] = int(c[4:] == cat)
    p = float(model.predict_proba(pd.DataFrame([row])[cols])[0, 1])
    st.metric("Fraud probability", f"{p*100:.2f}%"); st.error("ALERT: HIGH risk") if p >= 0.5 else st.success("No alert")
