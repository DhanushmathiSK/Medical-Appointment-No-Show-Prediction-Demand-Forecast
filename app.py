import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import date
import matplotlib.pyplot as plt 
import numpy as np

st.set_page_config(page_title="Clinic Appointment Tool", layout="wide")
st.title("Clinic Appointment Tool")

MODEL_DIR = Path(__file__).parent / "model"


@st.cache_resource
def load_noshow():
    model = joblib.load(MODEL_DIR / "noshow_model.joblib")
    cols = joblib.load(MODEL_DIR / "noshow_columns.joblib")
    defaults = joblib.load(MODEL_DIR / "noshow_defaults.joblib")
    return model, cols, defaults


def build_row(f, cols, defaults):
    """Turn the form answers into one row with the exact 61 training columns."""
    row = dict(defaults)  # start from typical values (weather etc.)

    # reset the one-hot groups we control, so medians don't interfere
    flags_to_keep = {"specialty_unknown", "disability_unknown"}
    for c in cols:
        if c.startswith(("specialty_", "gender_", "disability_", "place_", "age_group_")) \
                and c not in flags_to_keep:
            row[c] = 0
    row["appointment_shift_morning"] = 0

    # age
    age = f["age"] if not f["age_unknown"] else defaults["age"]
    row["age"] = age
    row["age_missing"] = int(f["age_unknown"])
    row["under_12_years_old"] = int(age <= 12)
    row["over_60_years_old"] = int(age > 60)
    row["young_adult"] = int(18 < age <= 40)
    if age <= 12:
        label = "0-12"
    elif age <= 18:
        label = "13-18"
    elif age <= 40:
        label = "19-40"
    elif age <= 60:
        label = "41-60"
    else:
        label = "60+"
    if f"age_group_{label}" in row:
        row[f"age_group_{label}"] = 1

    # categorical choices (baseline category = no column, stays 0)
    for prefix, value in [("specialty", f["specialty"]),
                          ("gender", f["gender"]),
                          ("disability", f["disability"]),
                          ("place", f["place"])]:
        name = f"{prefix}_{value}"
        if name in row:
            row[name] = 1

    # "unknown" flags
    spec_unknown = int(f["specialty"] == "Unknown")
    dis_unknown = int(f["disability"] == "Unknown")
    place_unknown = int(f["place"] == "Unknown")
    row["specialty_unknown"] = spec_unknown
    row["disability_unknown"] = dis_unknown
    row["missing_count"] = spec_unknown + dis_unknown + place_unknown

    # appointment time and date
    row["appointment_time"] = f["hour"]
    row["appointment_shift_morning"] = int(f["hour"] < 12)
    d = f["date"]
    row["day_of_week"] = d.weekday()
    row["month"] = d.month
    row["week_of_year"] = d.isocalendar()[1]
    row["day_of_month"] = d.day
    row["is_weekend"] = int(d.weekday() >= 5)

    # health and other yes/no fields
    for k in ["Hipertension", "Diabetes", "Alcoholism", "Handcap",
              "Scholarship", "SMS_received", "patient_needs_companion"]:
        row[k] = int(f[k])

    return pd.DataFrame([row])[cols].astype(float)


tab1, tab2, tab3 = st.tabs(["No-Show Predictor", "Demand Forecaster", "Insights"])

with tab1:
    st.subheader("Predict the risk of a missed appointment")
    try:
        model, cols, defaults = load_noshow()
    except Exception as e:
        st.error(f"Could not load the model files from the 'model' folder: {e}")
        st.stop()

    with st.form("noshow_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Patient**")
            age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
            age_unknown = st.checkbox("Age not recorded")
            gender = st.selectbox("Gender", ["F", "M", "I"])
            disability = st.selectbox("Disability", ["Unknown", "intellectual", "motor"])
            place = st.selectbox("City", [
                "Unknown", "ITAJAÍ", "B. CAMBORIU", "CAMBORIU", "NAVEGANTES",
                "ITAPEMA", "BOMBINHAS", "PENHA", "PORTO BELO", "BALN. PIÇARRAS", "Other"])

        with c2:
            st.markdown("**Appointment**")
            specialty = st.selectbox("Specialty", [
                "Unknown", "physiotherapy", "psychotherapy", "speech therapy",
                "occupational therapy", "pedagogo", "enf", "assist", "sem especialidade"])
            appt_date = st.date_input("Appointment date", value=date.today())
            hour = st.number_input("Appointment hour (24h)", min_value=6, max_value=20, value=9, step=1)
            SMS_received = st.checkbox("SMS reminder sent")

        with c3:
            st.markdown("**Health and other**")
            Hipertension = st.checkbox("Hypertension")
            Diabetes = st.checkbox("Diabetes")
            Alcoholism = st.checkbox("Alcoholism")
            Handcap = st.checkbox("Handicap")
            Scholarship = st.checkbox("Scholarship")
            patient_needs_companion = st.checkbox("Needs a companion")

        submitted = st.form_submit_button("Predict risk")

    if submitted:
        form = dict(age=age, age_unknown=age_unknown, gender=gender, disability=disability,
                    place=place, specialty=specialty, date=appt_date, hour=int(hour),
                    SMS_received=SMS_received, Hipertension=Hipertension, Diabetes=Diabetes,
                    Alcoholism=Alcoholism, Handcap=Handcap, Scholarship=Scholarship,
                    patient_needs_companion=patient_needs_companion)
        try:
            with st.spinner("Calculating risk..."):
                X_row = build_row(form, cols, defaults)
                score = float(model.predict_proba(X_row)[0, 1])

            if score >= 0.60:
                band, action = "HIGH", "Call the patient to confirm, and offer a different time if needed."
            elif score >= 0.45:
                band, action = "MEDIUM", "Send a reminder and keep the slot on a watch list."
            else:
                band, action = "LOW", "Standard reminder is enough."

            m1, m2 = st.columns(2)
            m1.metric("Risk score", f"{score:.0%}")
            m2.metric("Risk level", band)
            st.progress(min(max(score, 0.0), 1.0))
            st.info(action)
            st.caption("The score comes from a model trained with class weights, so it is a "
                      "relative risk ranking, not an exact chance. Weather values use typical defaults.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

@st.cache_resource
def load_demand():
    model = joblib.load(MODEL_DIR / "demand_model.joblib")
    cols = joblib.load(MODEL_DIR / "demand_columns.joblib")
    hist = pd.read_csv(MODEL_DIR / "daily_history.csv", index_col=0, parse_dates=True)["appointments"]
    return model, cols, hist


def forecast(model, cols, hist, days):
    series = hist.copy()
    out = []
    for _ in range(days):
        d = series.index[-1] + pd.Timedelta(days=1)
        vals = series.values
        row = {"dow": d.dayofweek, "month": d.month, "day": d.day,
               "lag_1": vals[-1], "lag_2": vals[-2], "lag_3": vals[-3],
               "lag_7": vals[-7], "lag_14": vals[-14],
               "roll7": vals[-7:].mean(), "roll14": vals[-14:].mean()}
        pred = max(float(model.predict(pd.DataFrame([row])[cols])[0]), 0.0)
        series.loc[d] = pred
        out.append((d, pred))
    return pd.DataFrame(out, columns=["date", "forecast"]).set_index("date")


with tab2:
    st.subheader("Forecast daily appointment volume")
    try:
        d_model, d_cols, hist = load_demand()
    except Exception as e:
        st.error(f"Could not load the forecasting files: {e}")
        st.stop()

    c1, c2 = st.columns(2)
    horizon = c1.slider("Days to forecast", 1, 14, 7)
    show_days = c2.slider("Days of history to show", 14, 120, 45)
    st.caption(f"Data ends on {hist.index[-1].date()}. The forecast starts the next day.")

    if st.button("Run forecast"):
        try:
            with st.spinner("Forecasting..."):
                fc = forecast(d_model, d_cols, hist, horizon)
            fig, ax = plt.subplots(figsize=(10, 4))
            hist.tail(show_days).plot(ax=ax, label="Actual")
            fc["forecast"].plot(ax=ax, label="Forecast", linestyle="--", marker="o")
            ax.set_ylabel("Appointments")
            ax.legend()
            st.pyplot(fig)
            st.dataframe(fc.round(0).astype(int).rename(columns={"forecast": "Expected appointments"}))
            st.metric("Total expected in period", int(fc["forecast"].sum()))
            st.caption("Accuracy is limited: in testing, R² was about 0.29 and typical error about 143 "
                       "appointments per day. Use for rough planning only.")
        except Exception as e:
            st.error(f"Forecast failed: {e}")

@st.cache_data
def load_insights():
    return pd.read_csv(MODEL_DIR / "insights_data.csv")


def rate_chart(df, col, title, top=None):
    r = df.groupby(col)["no_show"].mean().mul(100).sort_values()
    if top:
        r = r.tail(top)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    r.plot(kind="barh", ax=ax)
    ax.set_xlabel("No-show rate (%)")
    ax.set_title(title)
    return fig


with tab3:
    st.subheader("Key patterns in the data")
    try:
        ins = load_insights()
    except Exception as e:
        st.error(f"Could not load insights data: {e}")
        st.stop()

    k1, k2, k3 = st.columns(3)
    k1.metric("Appointments", f"{len(ins):,}")
    k2.metric("Overall no-show rate", f"{ins['no_show'].mean():.1%}")
    k3.metric("Model ROC-AUC", "0.786")

    a, b = st.columns(2)
    a.pyplot(rate_chart(ins, "age_group", "No-show rate by age group"))
    b.pyplot(rate_chart(ins, "specialty", "No-show rate by specialty"))

    c, d_ = st.columns(2)
    c.pyplot(rate_chart(ins, "place", "No-show rate by city (top 10 by volume)",)
             if False else rate_chart(
                 ins[ins["place"].isin(ins["place"].value_counts().head(10).index)],
                 "place", "No-show rate by city (top 10 by volume)"))
    d_.pyplot(rate_chart(ins, "SMS_received", "No-show rate: SMS received (0 = no, 1 = yes)"))

    st.markdown("""
**Main findings**
- Young adults (19 to 40) miss the most appointments, and patients over 60 the fewest.
- Missing city, specialty or disability is linked to much higher no-show rates.
- SMS reminders, rain and day of week show almost no effect.
- Suggested action: call high-risk patients instead of sending more SMS.
""")