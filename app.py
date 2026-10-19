import os, joblib, numpy as np, pandas as pd, plotly.express as px, streamlit as st
os.chdir(os.path.dirname(os.path.abspath(__file__)))
st.set_page_config(page_title="Fraud Command Center", page_icon="🛡️", layout="wide")
st.markdown("""<style>
[data-testid="stSidebar"]{background:#000}
[data-testid="stSidebar"] *{color:#e5e5e5 !important}
[data-testid="stMetric"]{background:#fff;border:1px solid #e5e5e5;border-radius:12px;padding:14px 18px}
[data-testid="stMetricLabel"]{color:#737373}
.block-container{padding-top:2rem;max-width:1200px}
h1,h2,h3{letter-spacing:-0.01em}
.card{background:#fff;border:1px solid #e5e5e5;border-radius:12px;padding:16px 18px;margin-bottom:12px}
.muted{color:#737373;font-size:13px}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load():
    d = pd.read_csv("processed_transactions.csv.gz", parse_dates=["event_time"])
    d["date"] = d.event_time.dt.date
    return d, pd.read_csv("hotspots.csv")
df, hot = load()
CATS = sorted(df.merchant_category.unique())

def style(fig, h=330):
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=h, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(color="#0a0a0a"), title_font_size=14)
    fig.update_xaxes(gridcolor="#eeeeee"); fig.update_yaxes(gridcolor="#eeeeee")
    return fig

st.sidebar.markdown("### 🛡️ FraudWatch")
view = st.sidebar.radio("Menu", ["Overview", "Live Feed", "Risk Map", "Historical Analytics", "Merchant-City Matrix", "ML Scoring Simulator"])
st.sidebar.caption("Credit Card Fraud Detection and Real-Time Transaction Analytics")
st.title(view)

if view == "Overview":
    st.caption("1 Jan to 31 Mar 2026 · synthetic transactions")
    a, b, c, d = st.columns(4)
    a.metric("Transactions", f"{len(df):,}"); b.metric("Fraud rate", f"{df.is_fraud.mean()*100:.2f}%")
    c.metric("Fraud cases", f"{int(df.is_fraud.sum()):,}"); d.metric("HIGH-risk alerts", f"{(df.risk_level == 'HIGH').sum():,}")
    daily = df.groupby("date").size().reset_index(name="transactions")
    f = px.line(daily, x="date", y="transactions", title="Transactions volume (per day)", color_discrete_sequence=["#6d28d9"])
    st.plotly_chart(style(f), use_container_width=True)
    l, r = st.columns([1, 1])
    with l:
        st.subheader("Risk distribution")
        for k, t in [("LOW", "score below 25"), ("MEDIUM", "25 to 45"), ("HIGH", "45 and above, alert raised")]:
            share = float((df.risk_level == k).mean())
            st.write(f"**{k}** — {share*100:.1f}%  ·  {t}"); st.progress(share)
    with r:
        st.subheader("Top hotspot merchants")
        st.dataframe(hot.head(5), use_container_width=True, hide_index=True)
elif view == "Live Feed":
    p = "live_predictions.jsonl"
    if os.path.exists(p):
        st.dataframe(pd.read_json(p, lines=True).tail(50), use_container_width=True)
    else:
        st.info("The live feed needs the streaming scripts (run locally): transaction_producer.py and stream_processor.py.")
elif view == "Risk Map":
    g = df.groupby("city").agg(lat=("latitude", "mean"), lon=("longitude", "mean"), fraud=("is_fraud", "mean"), n=("is_fraud", "size")).reset_index()
    g["fraud_pct"] = (g.fraud * 100).round(2)
    f = px.scatter_geo(g, lat="lat", lon="lon", size="n", color="fraud_pct", hover_name="city", scope="asia",
                       color_continuous_scale="Greys", size_max=40)
    st.plotly_chart(style(f, 520), use_container_width=True)
    st.dataframe(g[["city", "n", "fraud_pct"]].sort_values("fraud_pct", ascending=False), use_container_width=True, hide_index=True)
elif view == "Historical Analytics":
    h = (df.groupby("hour_of_day").is_fraud.mean() * 100).reset_index()
    st.plotly_chart(style(px.bar(h, x="hour_of_day", y="is_fraud", title="Fraud % by hour of day", color_discrete_sequence=["#0a0a0a"])), use_container_width=True)
    ch = (df.groupby("channel").is_fraud.mean() * 100).reset_index()
    ca = (df.groupby("merchant_category").is_fraud.mean() * 100).sort_values().reset_index()
    l, r = st.columns(2)
    l.plotly_chart(style(px.bar(ch, x="channel", y="is_fraud", title="Fraud % by channel", color_discrete_sequence=["#0a0a0a"])), use_container_width=True)
    r.plotly_chart(style(px.bar(ca, x="is_fraud", y="merchant_category", orientation="h", title="Fraud % by category", color_discrete_sequence=["#0a0a0a"])), use_container_width=True)
elif view == "Merchant-City Matrix":
    pv = (df.pivot_table(index="city", columns="channel", values="is_fraud", aggfunc="mean") * 100).round(2)
    st.plotly_chart(style(px.imshow(pv, text_auto=True, color_continuous_scale="Greys", title="Fraud % : city × channel"), 420), use_container_width=True)
    st.subheader("Top hotspot merchants"); st.dataframe(hot, use_container_width=True, hide_index=True)
else:
    st.caption("Enter a transaction. The rule-based score and the trained Random Forest both rate it.")
    model = joblib.load("rf.joblib"); cols = list(model.feature_names_in_)
    c1, c2 = st.columns(2)
    amount = c1.number_input("Amount (INR)", 10.0, 500000.0, 27000.0)
    median = c1.number_input("Customer median amount (INR)", 10.0, 500000.0, 6000.0)
    dist = c1.slider("Distance from home (km)", 0, 1500, 320)
    cat = c1.selectbox("Merchant category", CATS)
    vel = c2.slider("Transactions in last hour", 0, 10, 3)
    hr = c2.slider("Hour of day", 0, 23, 1); online = c2.checkbox("Online", True); foreign = c2.checkbox("Foreign")
    ratio = amount / median
    s = 0.35*min(ratio/6, 1)*100 + 0.25*min(dist/500, 1)*100 + 0.25*min(vel/5, 1)*100 + 0.15*min((40 if hr <= 4 else 0) + 30*online + 30*foreign, 100)
    lvl = "LOW" if s < 25 else "MEDIUM" if s < 45 else "HIGH"
    row = {"amount": amount, "amount_ratio": ratio, "distance_from_home_km": dist, "txn_count_1h": vel, "hour_of_day": hr,
           "day_of_week": 2, "is_weekend": 0, "is_night": int(hr <= 4), "is_online": int(online), "is_foreign": int(foreign)}
    for c in cols:
        if c.startswith("cat_"): row[c] = int(c[4:] == cat)
    p = float(model.predict_proba(pd.DataFrame([row])[cols])[0, 1])
    m1, m2, m3 = st.columns(3)
    m1.metric("Rule-based risk score", f"{s:.1f} / 100"); m2.metric("Risk level", lvl); m3.metric("Random Forest fraud probability", f"{p*100:.2f}%")
    if lvl == "HIGH" or p >= 0.5: st.error("ALERT: this transaction is HIGH risk")
    else: st.success("No alert")
