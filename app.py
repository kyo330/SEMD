import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Spanish Energy Market Dashboard", page_icon="⚡", layout="wide")

# ---- light theme for all Plotly charts: white background, dark text ----
import plotly.io as pio
_light = pio.templates["plotly_white"]
_light.layout.font.color = "#1A1A2E"
_light.layout.title.font.color = "#1A1A2E"
_light.layout.paper_bgcolor = "#FFFFFF"
_light.layout.plot_bgcolor = "#FFFFFF"
_light.layout.xaxis = dict(gridcolor="#E5E7EB", linecolor="#9CA3AF", tickfont=dict(color="#1A1A2E"))
_light.layout.yaxis = dict(gridcolor="#E5E7EB", linecolor="#9CA3AF", tickfont=dict(color="#1A1A2E"))
_light.layout.legend = dict(font=dict(color="#1A1A2E"))
pio.templates.default = _light

# Streamlit charts should not override with dark theme
PLOTLY_KW = dict(use_container_width=True, theme=None)

DATA_DIR = "data"

# ---------------------------------------------------------------- data loading
@st.cache_data(show_spinner="Loading energy data...")
def load_energy():
    df = pd.read_csv(f"{DATA_DIR}/energy_dataset.csv", parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Madrid")
    df = df.set_index("time").sort_index()
    # drop columns that are entirely empty / constant zero
    df = df.drop(columns=[c for c in df.columns if df[c].isna().all() or (df[c].fillna(0) == 0).all()])
    return df

@st.cache_data(show_spinner="Loading weather data...")
def load_weather():
    w = pd.read_csv(f"{DATA_DIR}/weather_features.csv", parse_dates=["dt_iso"])
    w["dt_iso"] = pd.to_datetime(w["dt_iso"], utc=True).dt.tz_convert("Europe/Madrid")
    w["temp_c"] = w["temp"] - 273.15  # Kelvin -> Celsius
    w["city_name"] = w["city_name"].str.strip()
    # de-duplicate (dataset has duplicate hourly rows per city)
    w = w.drop_duplicates(subset=["dt_iso", "city_name"])
    return w

@st.cache_data(show_spinner=False)
def national_temp(w: pd.DataFrame) -> pd.Series:
    """Population-weighted national temperature proxy."""
    weights = {"Madrid": 6.6, "Barcelona": 5.5, "Valencia": 2.5, "Seville": 1.9, "Bilbao": 1.0}
    pivot = w.pivot_table(index="dt_iso", columns="city_name", values="temp_c")
    cols = [c for c in pivot.columns if c in weights]
    ww = np.array([weights[c] for c in cols])
    return (pivot[cols] * ww).sum(axis=1) / ww.sum()

energy = load_energy()
weather = load_weather()
temp_nat = national_temp(weather)

GEN_COLS = [c for c in energy.columns if c.startswith("generation")]
GEN_GROUPS = {
    "Nuclear": ["generation nuclear"],
    "Gas": ["generation fossil gas"],
    "Coal": ["generation fossil hard coal", "generation fossil brown coal/lignite"],
    "Oil": ["generation fossil oil"],
    "Wind": ["generation wind onshore", "generation wind offshore"],
    "Solar": ["generation solar"],
    "Hydro": ["generation hydro run-of-river and poundage", "generation hydro water reservoir",
              "generation hydro pumped storage consumption"],
    "Other": ["generation biomass", "generation waste", "generation other", "generation other renewable",
              "generation geothermal", "generation fossil coal-derived gas"],
}

# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚡ Controls")
dmin, dmax = energy.index.min().date(), energy.index.max().date()
date_range = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
else:
    start, end = dmin, dmax
freq = st.sidebar.selectbox("Resample frequency", ["Hourly", "Daily", "Weekly"], index=1)
FREQ_MAP = {"Hourly": "h", "Daily": "D", "Weekly": "W"}

e = energy.loc[str(start):str(end)]
t = temp_nat.loc[str(start):str(end)]
rule = FREQ_MAP[freq]
e_rs = e.resample(rule).mean()
t_rs = t.resample(rule).mean()

st.title("Spanish Energy Market Dashboard")
st.caption("ENTSO-E load, generation & day-ahead prices + OpenWeather city weather · Spain 2015–2018 · Kaggle dataset by nicholasjhana")

tab_prices, tab_gen, tab_weather, tab_forecast, tab_trade, tab_news = st.tabs(
    ["💶 Prices", "🏭 Generation Mix", "🌡️ Weather vs Demand", "🔮 Demand Forecast", "📈 Trading Signals",
     "📰 AI Market Drivers"])

# ---------------------------------------------------------------- prices tab
with tab_prices:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg day-ahead price", f"{e['price day ahead'].mean():.1f} €/MWh")
    c2.metric("Avg actual price", f"{e['price actual'].mean():.1f} €/MWh")
    c3.metric("Max actual price", f"{e['price actual'].max():.1f} €/MWh")
    c4.metric("Avg load", f"{e['total load actual'].mean()/1000:.1f} GW")

    fig = go.Figure()
    fig.add_scatter(x=e_rs.index, y=e_rs["price day ahead"], name="Day-ahead", line=dict(width=1.2))
    fig.add_scatter(x=e_rs.index, y=e_rs["price actual"], name="Actual", line=dict(width=1.2))
    # spike markers: hours > mean + 3σ (on hourly data)
    thr = e["price actual"].mean() + 3 * e["price actual"].std()
    spikes = e[e["price actual"] > thr]
    if len(spikes):
        fig.add_scatter(x=spikes.index, y=spikes["price actual"], mode="markers",
                        marker=dict(color="red", size=5), name=f"Spikes (> {thr:.0f} €/MWh)")
    fig.update_layout(title=f"Electricity prices ({freq.lower()})", yaxis_title="€/MWh", height=420,
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, **PLOTLY_KW)

    col1, col2 = st.columns(2)
    with col1:
        monthly = energy["price actual"].groupby(energy.index.month).mean()
        figm = px.bar(x=monthly.index, y=monthly.values,
                      labels={"x": "Month", "y": "€/MWh"}, title="Seasonality: avg price by month")
        st.plotly_chart(figm, **PLOTLY_KW)
    with col2:
        hourly = energy["price actual"].groupby(energy.index.hour).mean()
        figh = px.line(x=hourly.index, y=hourly.values, markers=True,
                       labels={"x": "Hour of day", "y": "€/MWh"}, title="Daily shape: avg price by hour")
        st.plotly_chart(figh, **PLOTLY_KW)

    st.subheader("Gas generation ↔ power price")
    st.caption("No TTF gas price in this dataset, so gas-fired generation volume is used as the gas-market proxy.")
    daily = energy[["price actual", "generation fossil gas", "total load actual"]].resample("D").mean().dropna()
    corr = daily.corr().round(2)
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                  title="Daily correlations"), **PLOTLY_KW)
    with cc2:
        st.plotly_chart(px.scatter(daily, x="generation fossil gas", y="price actual", opacity=0.3,
                                   trendline="ols" if False else None,
                                   labels={"generation fossil gas": "Gas generation (MW)",
                                           "price actual": "Price (€/MWh)"},
                                   title="Gas-fired generation vs price (daily)"), **PLOTLY_KW)

# ---------------------------------------------------------------- generation tab
with tab_gen:
    grouped = pd.DataFrame({name: e[[c for c in cols if c in e.columns]].sum(axis=1, min_count=1)
                            for name, cols in GEN_GROUPS.items()
                            if any(c in e.columns for c in cols)})
    grouped_rs = grouped.resample(rule).mean()
    fig = go.Figure()
    for col in grouped_rs.columns:
        fig.add_scatter(x=grouped_rs.index, y=grouped_rs[col], name=col, stackgroup="one", mode="none")
    fig.update_layout(title=f"Generation mix ({freq.lower()})", yaxis_title="MW", height=480,
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, **PLOTLY_KW)

    col1, col2 = st.columns(2)
    with col1:
        share = grouped.mean().sort_values(ascending=False)
        st.plotly_chart(px.pie(values=share.values, names=share.index, title="Average generation share"),
                        **PLOTLY_KW)
    with col2:
        ren = grouped[[c for c in ["Wind", "Solar", "Hydro"] if c in grouped]].sum(axis=1)
        ren_share = (ren / grouped.sum(axis=1)).resample("W").mean() * 100
        st.plotly_chart(px.line(x=ren_share.index, y=ren_share.values,
                                labels={"x": "", "y": "%"}, title="Renewable share (weekly avg)"),
                        **PLOTLY_KW)

# ---------------------------------------------------------------- weather tab
with tab_weather:
    df = pd.DataFrame({"load": e["total load actual"], "temp": t}).dropna()
    daily = df.resample("D").mean().dropna()

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=daily.index, y=daily["load"]/1000, name="Load (GW)", yaxis="y1")
        fig.add_scatter(x=daily.index, y=daily["temp"], name="Temp (°C)", yaxis="y2", line=dict(color="orange"))
        fig.update_layout(title="Daily load vs national temperature",
                          yaxis=dict(title="GW"), yaxis2=dict(title="°C", overlaying="y", side="right"),
                          height=420, legend=dict(orientation="h"))
        st.plotly_chart(fig, **PLOTLY_KW)
    with c2:
        figs = px.scatter(daily, x="temp", y="load", opacity=0.35,
                          labels={"temp": "Temperature (°C)", "load": "Load (MW)"},
                          title="The U-curve: demand vs temperature")
        # quadratic fit line
        coef = np.polyfit(daily["temp"], daily["load"], 2)
        xs = np.linspace(daily["temp"].min(), daily["temp"].max(), 100)
        figs.add_scatter(x=xs, y=np.polyval(coef, xs), name="Quadratic fit", line=dict(color="red"))
        st.plotly_chart(figs, **PLOTLY_KW)

    pearson = daily["temp"].corr(daily["load"])
    comfort = 18
    daily["hdd"] = (comfort - daily["temp"]).clip(lower=0)
    daily["cdd"] = (daily["temp"] - comfort).clip(lower=0)
    st.markdown(
        f"Linear correlation temp↔load: **{pearson:.2f}**, which is weak, because the relationship is U-shaped "
        f"(heating in winter, cooling in summer). Degree-day correlations: "
        f"HDD↔load **{daily['hdd'].corr(daily['load']):.2f}**, CDD↔load **{daily['cdd'].corr(daily['load']):.2f}**.")

    city = st.selectbox("City temperature detail", sorted(weather["city_name"].unique()))
    wc = weather[weather["city_name"] == city].set_index("dt_iso")["temp_c"].loc[str(start):str(end)].resample("D").mean()
    st.plotly_chart(px.line(x=wc.index, y=wc.values, labels={"x": "", "y": "°C"},
                            title=f"Daily mean temperature: {city}"), **PLOTLY_KW)

# ---------------------------------------------------------------- forecast tab
with tab_forecast:
    st.subheader("Next-day demand forecast (linear regression)")
    st.caption("Features: temperature (HDD/CDD), calendar effects, and lagged load. Trained on the selected range minus the final 60 days, which are held out for testing.")

    df = pd.DataFrame({"load": energy["total load actual"], "temp": temp_nat}).dropna()
    daily = df.resample("D").mean().dropna()
    daily["hdd"] = (18 - daily["temp"]).clip(lower=0)
    daily["cdd"] = (daily["temp"] - 18).clip(lower=0)
    daily["dow"] = daily.index.dayofweek
    daily["weekend"] = (daily["dow"] >= 5).astype(int)
    daily["month"] = daily.index.month
    daily["load_lag1"] = daily["load"].shift(1)
    daily["load_lag7"] = daily["load"].shift(7)
    daily = daily.dropna()
    daily = pd.get_dummies(daily, columns=["dow", "month"], drop_first=True)

    features = [c for c in daily.columns if c not in ("load", "temp")]
    test_days = 60
    train, test = daily.iloc[:-test_days], daily.iloc[-test_days:]
    model = LinearRegression().fit(train[features], train["load"])
    pred = model.predict(test[features])

    mae = mean_absolute_error(test["load"], pred)
    mape = (np.abs(test["load"] - pred) / test["load"]).mean() * 100
    r2 = r2_score(test["load"], pred)
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:,.0f} MW")
    m2.metric("MAPE", f"{mape:.1f} %")
    m3.metric("R²", f"{r2:.3f}")

    fig = go.Figure()
    fig.add_scatter(x=test.index, y=test["load"], name="Actual")
    fig.add_scatter(x=test.index, y=pred, name="Predicted", line=dict(dash="dash"))
    fig.update_layout(title=f"Hold-out test: last {test_days} days", yaxis_title="MW", height=420,
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, **PLOTLY_KW)

    coefs = pd.Series(model.coef_, index=features).reindex(
        ["hdd", "cdd", "weekend", "load_lag1", "load_lag7"]).dropna()
    st.plotly_chart(px.bar(x=coefs.index, y=coefs.values,
                           labels={"x": "Feature", "y": "Coefficient (MW per unit)"},
                           title="Key model coefficients"), **PLOTLY_KW)
    hdd_c = coefs.get("hdd", 0)
    cdd_c = coefs.get("cdd", 0)
    if hdd_c > 0 and cdd_c > 0:
        dd_note = ("Positive HDD/CDD coefficients confirm the U-curve: each degree-day of heating or "
                   "cooling need adds load.")
    else:
        dd_note = ("Note: with lagged load in the model, the degree-day coefficients capture only the "
                   "*incremental* weather effect beyond yesterday's load, so their signs can differ from "
                   "the raw U-curve correlations.")
    st.markdown(dd_note + " Lagged load captures persistence; the weekend dummy captures the demand drop.")

# ---------------------------------------------------------------- trading tab
with tab_trade:
    st.subheader("Day-ahead vs imbalance spread strategy")
    st.markdown(
        "A simple *virtual trading* strategy: each hour, a model predicts the **actual (spot) price** using "
        "information known at day-ahead auction time (the day-ahead price itself, forecast load, forecast wind/solar, "
        "calendar). If the predicted spot exceeds the day-ahead price by a threshold, **go long** (buy day-ahead, "
        "settle at spot); if it is below by the threshold, **go short**. PnL per traded hour = position × "
        "(actual − day-ahead) in €/MWh. Chronological train/test split with no look-ahead.")

    thr = st.slider("Signal threshold (€/MWh edge required to trade)", 0.0, 10.0, 2.0, 0.5)
    test_frac = st.slider("Test fraction (most recent data)", 0.1, 0.5, 0.25, 0.05)
    cost = st.number_input("Transaction cost per trade (€/MWh)", 0.0, 5.0, 0.1, 0.1)

    sig = pd.DataFrame(index=energy.index)
    sig["da"] = energy["price day ahead"]
    sig["spot"] = energy["price actual"]
    for c in ["total load forecast", "forecast wind onshore day ahead", "forecast solar day ahead"]:
        if c in energy.columns:
            sig[c] = energy[c]
    sig["hour"] = sig.index.hour
    sig["dow"] = sig.index.dayofweek
    sig["month"] = sig.index.month
    sig = sig.dropna()
    sig = pd.get_dummies(sig, columns=["hour", "dow", "month"], drop_first=True)

    feats = [c for c in sig.columns if c != "spot"]
    n_test = int(len(sig) * test_frac)
    tr, te = sig.iloc[:-n_test], sig.iloc[-n_test:]
    mdl = LinearRegression().fit(tr[feats], tr["spot"])
    te = te.copy()
    te["pred_spot"] = mdl.predict(te[feats])
    te["edge"] = te["pred_spot"] - te["da"]
    te["pos"] = np.where(te["edge"] > thr, 1, np.where(te["edge"] < -thr, -1, 0))
    te["pnl"] = te["pos"] * (te["spot"] - te["da"]) - np.abs(te["pos"]) * cost

    traded = te[te["pos"] != 0]
    n_trades = len(traded)
    hit = (traded["pnl"] > 0).mean() * 100 if n_trades else 0.0
    total = te["pnl"].sum()
    avg = traded["pnl"].mean() if n_trades else 0.0
    daily_pnl = te["pnl"].resample("D").sum()
    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(365) if daily_pnl.std() > 0 else 0.0
    cum = te["pnl"].cumsum()
    dd = (cum - cum.cummax()).min()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Trades", f"{n_trades:,}")
    m2.metric("Hit rate", f"{hit:.1f} %")
    m3.metric("Total PnL", f"{total:,.0f} €/MWh")
    m4.metric("Avg PnL / trade", f"{avg:.2f} €/MWh")
    m5.metric("Sharpe (daily, ann.)", f"{sharpe:.2f}")

    fig = go.Figure()
    fig.add_scatter(x=cum.index, y=cum.values, name="Strategy", line=dict(width=1.5))
    bh = (te["spot"] - te["da"]).cumsum()  # always-long benchmark
    fig.add_scatter(x=bh.index, y=bh.values, name="Always-long benchmark", line=dict(width=1.2, dash="dot"))
    fig.update_layout(title=f"Cumulative PnL on test set (max drawdown {dd:,.0f} €/MWh)",
                      yaxis_title="€/MWh (per 1 MWh traded each hour)", height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, **PLOTLY_KW)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(traded, x="pnl", nbins=60, title="PnL distribution per trade",
                                     labels={"pnl": "€/MWh"}), **PLOTLY_KW)
    with c2:
        by_hour = traded.copy()
        by_hour["h"] = by_hour.index.hour
        st.plotly_chart(px.bar(by_hour.groupby("h")["pnl"].mean(),
                               labels={"value": "Avg PnL (€/MWh)", "h": "Hour"},
                               title="Avg PnL by hour of day").update_layout(showlegend=False), **PLOTLY_KW)

    st.caption("⚠️ Educational toy model on 2015–2018 data: ignores liquidity, execution at exact prices, "
               "and imbalance penalty mechanics. Not trading advice.")

# ---------------------------------------------------------------- AI market drivers tab
with tab_news:
    st.subheader("AI-structured market drivers")
    st.markdown(
        "Live European energy headlines are scraped from public RSS feeds, then an LLM API "
        "converts each into a **structured signal**: commodity, region, direction, impact, and a one-line "
        "rationale, the kind of unstructured→structured workflow a market analyst runs every morning.")

    try:
        groq_key = st.secrets.get("GROQ_API_KEY", "")
        gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:  # no secrets.toml at all
        groq_key = gemini_key = ""
    provider = "groq" if groq_key else "gemini"
    api_key = groq_key or gemini_key
    if not api_key:
        st.warning(
            "No `GROQ_API_KEY` or `GEMINI_API_KEY` found in Streamlit secrets.\n\n"
            "1. Get a free key: Groq at https://console.groq.com/keys (starts `gsk_`) "
            "or Gemini at https://aistudio.google.com/apikey (starts `AIzaSy`)\n"
            "2. Locally: add `GROQ_API_KEY = \"...\"` to `.streamlit/secrets.toml` (git-ignored)\n"
            "3. Streamlit Cloud: App → Settings → Secrets → paste the same line, then rerun.")
    else:
        import news_agent

        @st.cache_data(ttl=3600, show_spinner="Scraping feeds and querying the LLM (cached hourly)...")
        def cached_drivers():
            return news_agent.get_market_drivers(api_key, provider=provider)

        col_a, col_b = st.columns([1, 5])
        if col_a.button("🔄 Refresh now"):
            cached_drivers.clear()

        try:
            result = cached_drivers()
        except Exception as exc:
            st.error(f"News pipeline failed: {exc}")
            result = None

        if result:
            sigs = pd.DataFrame(result["signals"])
            col_b.caption(f"Last updated {result['fetched_at']} · {result['n_headlines']} headlines scanned · "
                          f"{len(sigs)} relevant signals · refreshes at most once per hour")
            if sigs.empty:
                st.info("No relevant energy-market signals found in the current feed window.")
            else:
                c1, c2, c3 = st.columns(3)
                bull = (sigs["direction"] == "bullish").sum()
                bear = (sigs["direction"] == "bearish").sum()
                c1.metric("Bullish signals", bull)
                c2.metric("Bearish signals", bear)
                c3.metric("Net sentiment", f"{bull - bear:+d}",
                          help="bullish minus bearish signal count")

                cc1, cc2 = st.columns(2)
                with cc1:
                    st.plotly_chart(px.histogram(sigs, x="commodity", color="direction", barmode="group",
                                                 title="Signals by commodity & direction"), **PLOTLY_KW)
                with cc2:
                    st.plotly_chart(px.histogram(sigs, x="region", color="direction", barmode="group",
                                                 title="Signals by region"), **PLOTLY_KW)

                icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
                for _, s in sigs.sort_values("impact", ascending=False).iterrows():
                    st.markdown(
                        f"{icon.get(s['direction'],'⚪')} **{s['headline']}**  \n"
                        f"`{s['commodity'].upper()}` · {s['region']} · impact {'★' * int(s['impact'])}: "
                        f"{s['rationale']} *({s['source']})*")

    st.caption("⚠️ LLM-generated interpretation of public headlines. A research aid, not trading advice.")
