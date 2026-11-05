# app.py - Prototipo web: clasificación de masas mamarias con SVM (proyecto académico)
import numpy as np
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.set_page_config(page_title="SVM - Masas mamarias", page_icon="🩺", layout="wide")


@st.cache_resource
def train_model():
    """Entrena el modelo final del notebook: mismo split, mismos hiperparámetros."""
    data = load_breast_cancer()
    X = data.data
    y = np.where(data.target == 0, 1, 0)  # 1 = maligno, 0 = benigno
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe, list(data.feature_names), X, y


model, feature_names, X_data, y_data = train_model()
means = X_data.mean(axis=0)
EJEMPLOS = {
    "Promedio del dataset": means,
    "Muestra benigna real (ejemplo)": X_data[np.where(y_data == 0)[0][0]],
    "Muestra maligna real (ejemplo)": X_data[np.where(y_data == 1)[0][0]],
}

# Estado inicial de los 30 campos
for i in range(len(feature_names)):
    st.session_state.setdefault(f"f_{i}", float(means[i]))


def cargar_ejemplo():
    vals = EJEMPLOS[st.session_state["ejemplo"]]
    for i in range(len(feature_names)):
        st.session_state[f"f_{i}"] = float(vals[i])


st.title("🩺 Prototipo experimental de IA: clasificación de masas mamarias")
st.markdown(
    "**Proyecto Final - Inteligencia Artificial (UPB 2026)**  \n"
    "*Integrantes:* Juan Felipe Cardona, Daniel Villa y Valeria Gómez.")
st.warning(
    "⚠️ **Aviso importante:** prototipo académico y experimental. No es una herramienta "
    "de diagnóstico médico ni sustituye la valoración de un profesional de la salud.")

st.sidebar.header("Entradas (30 características FNA)")
st.sidebar.selectbox("Cargar un ejemplo", list(EJEMPLOS), key="ejemplo", on_change=cargar_ejemplo)
valores = []
for i, name in enumerate(feature_names):
    valores.append(st.sidebar.number_input(str(name), min_value=0.0, format="%.4f", key=f"f_{i}"))

col1, col2 = st.columns([2, 1])
with col1:
    if st.button("Clasificar muestra", type="primary"):
        x = np.array(valores).reshape(1, -1)
        pred = int(model.predict(x)[0])
        proba = model.predict_proba(x)[0]
        st.subheader("Resultado experimental")
        if pred == 1:
            st.error(f"⚠️ **Predicción: MALIGNO** (probabilidad estimada: {proba[1]*100:.2f} %)")
            st.markdown("Las mediciones se parecen a las de muestras malignas del conjunto de entrenamiento.")
        else:
            st.success(f"✅ **Predicción: BENIGNO** (probabilidad estimada: {proba[0]*100:.2f} %)")
            st.markdown("Las mediciones se parecen a las de muestras benignas del conjunto de entrenamiento.")
        st.info(f"Probabilidades del modelo: Benigno {proba[0]*100:.2f} % | Maligno {proba[1]*100:.2f} %")
        st.caption("Las probabilidades provienen del SVM (escalado de Platt) y no están calibradas "
                   "clínicamente. Ningún resultado de esta herramienta debe usarse para decisiones médicas.")
with col2:
    with st.expander("Sobre el modelo y sus limitaciones", expanded=True):
        st.markdown(
            "- **Modelo:** SVM con kernel RBF (C = 10, gamma = scale) y `StandardScaler`.\n"
            "- **Datos:** Breast Cancer Wisconsin (Diagnostic), 569 muestras.\n"
            "- **Desempeño en prueba (114 muestras):** accuracy 97.37 %, sensibilidad 92.86 %, "
            "F1 96.30 %, 3 falsos negativos.\n"
            "- **Limitaciones:** una sola cohorte histórica, sin validación externa, "
            "muestra de prueba pequeña y umbral por defecto (0.5).")
