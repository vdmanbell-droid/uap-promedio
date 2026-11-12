"""
Predictor de promedio (UAP)
Formulario corto -> promedio semestral estimado.
Modelo: promedio de Ridge regression y Random Forest, entrenado con la encuesta a alumnos de la UAP.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="¿Cuál será tu promedio?", page_icon="🎓", layout="centered")

DATA_FILE = Path(__file__).parent / "uap_survey.xlsx"
FEATURES = ["midterm", "courses", "attendance_pct", "study_hours_day", "preparation", "sleep_n", "job_b"]
SLEEP = ["Muy mala", "Mala", "Normal", "Buena", "Muy buena"]
SLEEP_EN = ["Very bad", "Bad", "Normal", "Good", "Very good"]
ATT = {"Menos de 70%": 65, "80%": 80, "90%": 90, "100%": 100}
PREP = {1: "1 · Nada preparado", 2: "2 · Poco", 3: "3 · Regular", 4: "4 · Bien preparado", 5: "5 · Muy preparado"}

QUESTIONS = {
    "midterm": "What was your average grade on the midterm",
    "gpa_prev": "What was your overall GPA",
    "courses": "How many courses are you currently taking",
    "attendance": "What is your average attendance percentage",
    "study_hours_day": "On average, how many hours do you study per day",
    "preparation": "How would you describe your level of preparation",
    "job": "Do you currently have a job",
    "sleep": "How would you rate the quality of your sleep",
}


@st.cache_data
def load_data(path):
    raw = pd.read_excel(path)
    raw.columns = [str(c).replace("\xa0", " ").strip().lstrip("¿¡ ") for c in raw.columns]
    df = pd.DataFrame({k: raw[[c for c in raw.columns if c.startswith(p)]].bfill(axis=1).iloc[:, 0].values
                       for k, p in QUESTIONS.items()})
    num = lambda s: pd.to_numeric(s.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    df["midterm"] = pd.to_numeric(df["midterm"], errors="coerce")
    df["gpa_prev"] = num(df["gpa_prev"]).replace(0, np.nan)
    for col in ["courses", "study_hours_day", "preparation"]:
        df[col] = num(df[col])
    df["attendance_pct"] = df["attendance"].astype(str).str.strip().map(
        {"Less than 70%": 65, "80%": 80, "90%": 90, "100%": 100})
    df["sleep_n"] = df["sleep"].astype(str).str.strip().map({s: i + 1 for i, s in enumerate(SLEEP_EN)})
    df["job_b"] = (df["job"].astype(str).str.strip().str.lower() == "yes").astype(int)
    return df[df["gpa_prev"].notna()].reset_index(drop=True)


def make_model():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", VotingRegressor([
            ("ridge", RidgeCV(alphas=np.logspace(-2, 3, 30))),
            ("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, random_state=42)),
        ])),
    ])


@st.cache_resource(show_spinner=False)
def train():
    df = load_data(DATA_FILE)
    X, y = df[FEATURES], df["gpa_prev"].astype(float)
    cv = cross_validate(make_model(), X, y, cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=1),
                        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"])
    model = make_model().fit(X, y)
    return model, y, {"mae": -cv["test_neg_mean_absolute_error"].mean(),
                      "rmse": -cv["test_neg_root_mean_squared_error"].mean(),
                      "r2": cv["test_r2"].mean()}


with st.spinner("Preparando el modelo…"):
    model, y_train, metrics = train()

st.title("🎓 ¿Cuál será tu promedio?")
st.write("Contesta 7 preguntas rápidas y te damos una estimación de tu promedio semestral, "
         "basada en las respuestas de alumnos de la UAP.")

with st.form("formulario"):
    midterm = st.slider("1. ¿Qué promedio sacaste en tus parciales?", 5.0, 10.0, 9.0, 0.1)
    c1, c2 = st.columns(2)
    courses = c1.number_input("2. ¿Cuántas materias llevas?", min_value=1, max_value=14, value=8, step=1)
    att = c2.selectbox("3. ¿Cuál es tu asistencia promedio?", list(ATT), index=2)
    hours = c1.number_input("4. ¿Cuántas horas estudias al día?", min_value=0, max_value=10, value=1, step=1)
    job = c2.radio("5. ¿Trabajas actualmente?", ["No", "Sí"], horizontal=True)
    prep = st.select_slider("6. ¿Qué tan preparado(a) te sentiste para tus exámenes?",
                            options=list(PREP), value=4, format_func=PREP.get)
    sleep = st.select_slider("7. ¿Cómo dormiste en semana de exámenes?", options=SLEEP, value="Normal")
    sent = st.form_submit_button("Calcular mi promedio", type="primary")

if sent:
    row = pd.DataFrame([{
        "midterm": midterm, "courses": courses, "attendance_pct": ATT[att], "study_hours_day": hours,
        "preparation": prep, "sleep_n": SLEEP.index(sleep) + 1, "job_b": int(job == "Sí"),
    }])[FEATURES]
    pred = float(np.clip(model.predict(row)[0], 0, 10))
    err = metrics["mae"]
    lo, hi = max(pred - err, 0), min(pred + err, 10)
    pct = (y_train < pred).mean() * 100

    st.divider()
    st.markdown(
        f"""
<div style="text-align:center;padding:8px 0 4px">
  <div style="font-size:1rem;opacity:.75">Tu promedio estimado es</div>
  <div style="font-size:4.2rem;font-weight:700;line-height:1.1">{pred:.1f}</div>
  <div style="font-size:1rem;opacity:.75">probablemente entre <b>{lo:.1f}</b> y <b>{hi:.1f}</b></div>
</div>
""",
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_bar(x=[10], y=[""], orientation="h", marker_color="rgba(150,150,150,0.18)", hoverinfo="skip", width=0.5)
    fig.add_bar(x=[pred], y=[""], orientation="h", marker_color="#2E5EAA", hoverinfo="skip", width=0.5)
    fig.add_shape(type="rect", x0=lo, x1=hi, y0=-0.35, y1=0.35, line_width=0, fillcolor="rgba(224,122,46,0.35)")
    fig.update_layout(barmode="overlay", height=110, margin=dict(l=10, r=10, t=10, b=30), showlegend=False,
                      xaxis=dict(range=[5, 10], dtick=1, title=None), yaxis=dict(visible=False),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    if pct >= 50:
        st.success(f"Estarías por encima del **{pct:.0f}%** de los alumnos encuestados.")
    else:
        st.info(f"Estarías por encima del **{pct:.0f}%** de los alumnos encuestados. ¡Todavía hay margen para subir!")

    tips = []
    if prep <= 3:
        tips.append("Sentirte mejor preparado(a) se asocia con mejores calificaciones.")
    if ATT[att] < 90:
        tips.append("Subir tu asistencia a 90% o más podría ayudarte.")
    if tips:
        st.markdown("**Sugerencias:** " + " ".join(tips))
else:
    st.caption("Llena el formulario y presiona **Calcular mi promedio**.")

with st.expander("¿Cómo funciona?"):
    st.markdown(
        f"""
- El modelo se entrenó con las respuestas de **alumnos de la UAP** a la encuesta (promedio del semestre anterior).
- Combina dos modelos de *machine learning*: **Ridge regression** y **Random Forest**; la estimación es el promedio de ambos.
- Probado con validación cruzada (alumnos que el modelo no vio), se equivoca en promedio **±{metrics['mae']:.2f} puntos**.
- La pregunta que más pesa es tu **calificación de parciales**.
- Es una estimación con pocos datos: tómala como referencia, no como un resultado garantizado.
"""
    )
