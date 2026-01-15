import os
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor


st.image("image/copyrights.png", caption="© 2026 ", width=100)
# --------------------------------------------------
# CONFIG STREAMLIT
# --------------------------------------------------
st.set_page_config(
    page_title="Dashboard industriel – Cosmétiques",
    layout="wide"
)

st.title("Dashboard industriel – Cosmétiques")

st.markdown(" Ce dashboard interactif permet d'explorer les données industrielles et commerciales d'une entreprise de cosmétiques. ")

st.subheader("Auteur : Anthony DJOUMBISSI")

# --------------------------------------------------
# CHEMIN ROBUSTE DU DATASET
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Adaptez si besoin : 200k pour Cloud (recommandé), 1M en local
DATA_PATH = os.path.join(BASE_DIR, "data", "cosmetics_industrial_dataset.csv.gz")

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Fichier introuvable : {path}")
        st.stop()

    df = pd.read_csv(path, parse_dates=["date"])

    # Sécuriser certains types (facultatif, mais utile)
    for c in ["sku", "region", "channel", "category"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df

df = load_data(DATA_PATH)

# --------------------------------------------------
# SIDEBAR – FILTRES
# --------------------------------------------------
st.sidebar.header("Filtres")

date_min, date_max = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input(
    "Période",
    (date_min.date(), date_max.date())
)



page = st.sidebar.radio(
    "Page",
    ["1) Analyse exploratoire", "2) Analyse explicative", "3) Analyse prédictive"]
)

def multiselect_all(label: str, series: pd.Series):
    opts = sorted(series.dropna().unique().tolist())
    return st.sidebar.multiselect(label, opts, default=opts)

selected_skus = multiselect_all("SKU", df["sku"])
selected_regions = multiselect_all("Région", df["region"])
selected_channels = multiselect_all("Canal (channel)", df["channel"])
selected_categories = multiselect_all("Catégorie", df["category"])

# Filtrage global
mask = (
    (df["date"].dt.date >= date_range[0]) &
    (df["date"].dt.date <= date_range[1]) &
    (df["sku"].isin(selected_skus)) &
    (df["region"].isin(selected_regions)) &
    (df["channel"].isin(selected_channels)) &
    (df["category"].isin(selected_categories))
)


dff = df.loc[mask].copy()

if dff.empty:
    st.warning("Aucune donnée ne correspond aux filtres sélectionnés. Élargissez vos critères.")
    st.stop()

def fmt_int(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")

# --------------------------------------------------
# PAGE 1 – ANALYSE EXPLORATOIRE
# --------------------------------------------------
if page.startswith("1"):
    st.title("1) Analyse exploratoire des données")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes (filtrées)", fmt_int(len(dff)))
    c2.metric("SKU distincts", fmt_int(dff["sku"].nunique()))
    c3.metric("Période", f"{dff['date'].min().date()} → {dff['date'].max().date()}")
    c4.metric("Taux promo", f"{(dff['promo_flag'].mean()*100):.1f}%")

    st.subheader("Aperçu des données")
    st.dataframe(dff.head(50))

    st.subheader("Qualité des données (valeurs manquantes)")
    miss = (dff.isna().mean() * 100).sort_values(ascending=False).head(15).to_frame("missing_%")
    st.dataframe(miss.style.format("{:.2f}"))

    st.subheader("Statistiques descriptives (numériques)")
    num_cols = [
        "units_produced","units_sold","defects_units","returns_units","lead_time_days",
        "net_price_xaf","revenue_xaf","gross_margin_xaf","operating_margin_xaf"
    ]
    num_cols = [c for c in num_cols if c in dff.columns]
    st.dataframe(dff[num_cols].describe().T)

    st.subheader("Séries temporelles – CA & unités vendues")
    daily = (
        dff.set_index("date")
        .resample("D")[["revenue_xaf", "units_sold"]]
        .sum()
        .sort_index()
    )

    colA, colB = st.columns(2)
    with colA:
        st.caption("CA quotidien (XAF)")
        st.line_chart(daily[["revenue_xaf"]])
    with colB:
        st.caption("Unités vendues quotidiennes")
        st.line_chart(daily[["units_sold"]])

    st.subheader("Top catégories (CA)")
    by_cat = (
        dff.groupby("category", observed=True)["revenue_xaf"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .to_frame("revenue_xaf")
    )
    st.bar_chart(by_cat)

# --------------------------------------------------
# PAGE 2 – ANALYSE EXPLICATIVE
# --------------------------------------------------
elif page.startswith("2"):
    st.title("2) Analyse explicative (KPI & drivers)")

    revenue = dff["revenue_xaf"].sum()
    gross_margin = dff["gross_margin_xaf"].sum()
    operating_margin = dff["operating_margin_xaf"].sum()
    units = dff["units_sold"].sum()
    produced = dff["units_produced"].sum()
    defects = dff["defects_units"].sum()
    returns = dff["returns_units"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CA (XAF)", fmt_int(revenue))
    c2.metric("Marge brute (XAF)", fmt_int(gross_margin))
    c3.metric("Marge op. (XAF)", fmt_int(operating_margin))
    c4.metric("Unités vendues", fmt_int(units))
    c5.metric("Taux défauts", f"{(defects / max(produced, 1) * 100):.2f}%")

    st.subheader("Performance par catégorie")
    perf_cat = (
        dff.groupby("category", observed=True)
        .agg(
            revenue_xaf=("revenue_xaf", "sum"),
            units_sold=("units_sold", "sum"),
            gross_margin_xaf=("gross_margin_xaf", "sum"),
            defects_units=("defects_units", "sum"),
            units_produced=("units_produced", "sum"),
        )
        .sort_values("revenue_xaf", ascending=False)
    )
    perf_cat["gm_%"] = perf_cat["gross_margin_xaf"] / perf_cat["revenue_xaf"] * 100
    perf_cat["defect_%"] = perf_cat["defects_units"] / perf_cat["units_produced"] * 100

    st.dataframe(
        perf_cat.style.format({
            "revenue_xaf": "{:,.0f}",
            "units_sold": "{:,.0f}",
            "gross_margin_xaf": "{:,.0f}",
            "gm_%": "{:.2f}",
            "defect_%": "{:.2f}",
        })
    )

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Top 15 SKU (CA)")
        top_sku = (
            dff.groupby("sku", observed=True)["revenue_xaf"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .to_frame("revenue_xaf")
        )
        st.bar_chart(top_sku)

    with colB:
        st.subheader("Canaux (CA)")
        by_channel = (
            dff.groupby("channel", observed=True)["revenue_xaf"]
            .sum()
            .sort_values(ascending=False)
            .to_frame("revenue_xaf")
        )
        st.bar_chart(by_channel)

    st.subheader("Série temporelle – CA quotidien")
    daily_rev = dff.set_index("date").resample("D")["revenue_xaf"].sum().to_frame("revenue_xaf")
    st.line_chart(daily_rev)

# --------------------------------------------------
# PAGE 3 – ANALYSE PRÉDICTIVE
# --------------------------------------------------
else:
    st.title("3) Analyse prédictive – Prévision des ventes (J+7)")

    horizon = 7

    # Liste des SKU disponibles dans le périmètre filtré
    available_skus = sorted(dff["sku"].unique().tolist())
    sku = st.selectbox("Choisir un SKU (périmètre filtré)", available_skus)

    # Série journalière (unités vendues) pour le SKU
    g = (
        dff.loc[dff["sku"] == sku]
        .set_index("date")
        .resample("D")[["units_sold", "promo_flag", "discount_rate"]]
        .sum()
        .sort_index()
    )

    # Feature engineering simple & robuste
    g["dow"] = g.index.dayofweek
    g["month"] = g.index.month
    g["lag_1"] = g["units_sold"].shift(1)
    g["lag_7"] = g["units_sold"].shift(7)
    g["lag_14"] = g["units_sold"].shift(14)
    g["roll_7"] = g["units_sold"].shift(1).rolling(7).mean()
    g["roll_28"] = g["units_sold"].shift(1).rolling(28).mean()

    # Cible = unités vendues à J+7
    g["y"] = g["units_sold"].shift(-horizon)

    feat_cols = ["promo_flag", "discount_rate", "dow", "month", "lag_1", "lag_7", "lag_14", "roll_7", "roll_28"]
    data = g.dropna().copy()

    if len(data) < 90:
        st.warning("Historique insuffisant après features. Élargissez la période ou les filtres.")
        st.stop()

    X = data[feat_cols]
    y = data["y"]

    split = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = (mean_squared_error(y_test, preds) ** 0.5)  # robuste (sans squared=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (unités)", f"{mae:.2f}")
    c2.metric("RMSE (unités)", f"{rmse:.2f}")
    c3.metric("Taille test (jours)", fmt_int(len(y_test)))

    st.subheader("Réel vs Prévu (sur la période test)")
    out = pd.DataFrame({"y_true": y_test, "y_pred": preds}, index=y_test.index)
    st.line_chart(out)

    st.subheader("Projection (14 prochains jours) – démo")
    proj_dates = pd.date_range(data.index.max() + pd.Timedelta(days=1), periods=14, freq="D")
    tmp = g.copy()
    proj = []

    for d in proj_dates:
        row = {
            "promo_flag": 0,
            "discount_rate": 0,
            "dow": d.dayofweek,
            "month": d.month,
            "lag_1": tmp["units_sold"].iloc[-1],
            "lag_7": tmp["units_sold"].iloc[-7] if len(tmp) >= 7 else tmp["units_sold"].iloc[-1],
            "lag_14": tmp["units_sold"].iloc[-14] if len(tmp) >= 14 else tmp["units_sold"].iloc[-1],
            "roll_7": tmp["units_sold"].iloc[-7:].mean() if len(tmp) >= 7 else tmp["units_sold"].iloc[-1],
            "roll_28": tmp["units_sold"].iloc[-28:].mean() if len(tmp) >= 28 else tmp["units_sold"].iloc[-1],
        }
        x = pd.DataFrame([row])[feat_cols]
        yhat = float(model.predict(x)[0])
        yhat = max(0.0, yhat)

        proj.append([d, yhat])
        tmp.loc[d, "units_sold"] = yhat

    proj_df = pd.DataFrame(proj, columns=["date", "pred_units_sold"]).set_index("date")
    st.line_chart(proj_df)



st.image("image/porte drapeau SEAHORSE.png", caption="© 2026 ", width=100)