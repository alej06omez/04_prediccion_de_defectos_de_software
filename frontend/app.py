import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
cfg = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text(encoding="utf-8"))
API = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title=cfg["titulo"], layout="wide", page_icon="🧠")
st.title("🧠 " + cfg["titulo"])
st.caption("Curso de Inteligencia Artificial | Docente: Julian Rene Muñoz Burbano")

with st.sidebar:
    st.subheader("Sesion")
    rol = st.selectbox("Rol", ["operativo", "analista", "admin"])
    api_key = st.text_input("API key (si aplica)", type="password")
    st.caption("IA estima P(defecto). El semaforo lo definen umbrales y reglas (cobertura, modulo critico).")

headers = {"X-Role": rol}
if api_key:
    headers["X-API-Key"] = api_key


def get(path, **kwargs):
    return requests.get(API + path, headers=headers, timeout=20, **kwargs)


def post(path, **kwargs):
    return requests.post(API + path, headers=headers, timeout=60, **kwargs)


@st.cache_data(ttl=30)
def load_schema():
    r = get("/schema")
    r.raise_for_status()
    return r.json()


try:
    schema = load_schema()
    features = schema["features"]
except requests.RequestException:
    schema = {"umbrales_riesgo": cfg.get("umbrales_riesgo", {})}
    features = ["loc", "v(g)", "ev(g)", "iv(g)", "n", "lOComment", "branchCount"]
    st.warning("API no disponible. Levante uvicorn en el puerto 8000.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Dashboard de riesgo", "Prediccion por modulo", "Modelo y entrenamiento", "Historial", "IA vs reglas"]
)

with tab1:
    st.markdown("Ranking de modulos PC1 segun probabilidad de defecto y nivel de negocio.")
    limit = st.slider("Modulos a mostrar", 10, 100, 40)
    if st.button("Actualizar ranking", type="primary"):
        try:
            r = get("/ranking", params={"limit": limit})
            r.raise_for_status()
            data = pd.DataFrame(r.json())
            st.session_state["ranking"] = data
        except requests.RequestException as e:
            st.error(f"No se pudo cargar el ranking: {e}")
    data = st.session_state.get("ranking")
    if data is not None and not data.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Modulos", len(data))
        c2.metric("Riesgo alto", int((data["nivel_riesgo"] == "alto").sum()))
        c3.metric("P(defecto) media", f"{data['probabilidad_defecto'].mean():.2f}")
        st.bar_chart(data.set_index("module_id")["probabilidad_defecto"].head(15))
        nivel = st.multiselect("Filtrar nivel", ["alto", "medio", "bajo"], default=["alto", "medio", "bajo"])
        vista = data[data["nivel_riesgo"].isin(nivel)]
        st.dataframe(vista, use_container_width=True)
        st.download_button(
            "Exportar ranking CSV",
            vista.to_csv(index=False).encode("utf-8-sig"),
            "ranking_riesgo.csv",
            "text/csv",
        )
    else:
        st.info("Entrene el modelo y pulse Actualizar ranking.")

with tab2:
    with st.expander("Calcular métricas desde código C", expanded=False):
        analysis_module_id = st.text_input("ID del módulo analizado", "codigo-pegado", key="analysis_module_id")
        pasted_code = st.text_area("Código C", height=240, key="code_to_analyze")
        if st.button("Calcular métricas", key="analyze_code_button", type="primary"):
            try:
                response = post("/api/analyze-code", json={"code": pasted_code, "module_id": analysis_module_id})
                response.raise_for_status()
                analysis = response.json()
                st.session_state["analyzed_metrics"] = analysis["metrics"]
                st.session_state["analyzed_module_id"] = analysis_module_id
                st.success("Métricas calculadas de forma determinista. Se precargaron en el formulario inferior.")
                st.json(analysis)
            except requests.RequestException as e:
                st.error(f"No se pudo analizar el código: {e}")
    with st.form("inference"):
        module_id = st.text_input("module_id", st.session_state.get("analyzed_module_id", "auth-service"))
        cols = st.columns(3)
        metricas = {}
        defaults = {"loc": 180.0, "v(g)": 12.0, "lOComment": 2.0, "branchCount": 20.0}
        defaults.update(st.session_state.get("analyzed_metrics", {}))
        for i, name in enumerate(features):
            metricas[name] = cols[i % 3].number_input(name, value=float(defaults.get(name, 1.0)))
        cobertura = st.slider("Cobertura de tests (regla de negocio)", 0.0, 1.0, 0.4)
        critico = st.checkbox("Modulo critico (regla de negocio)")
        descripcion = st.text_input("Descripcion opcional")
        submit = st.form_submit_button("Analizar y guardar", type="primary")
    if submit:
        payload = {
            "module_id": module_id,
            "metricas": metricas,
            "cobertura_tests": cobertura,
            "es_modulo_critico": critico,
            "descripcion": descripcion,
        }
        try:
            r = post("/predict", json=payload)
            r.raise_for_status()
            out = r.json()
            st.success(f"{out['module_id']}: {out['nivel_riesgo']} (P={out['probabilidad_defecto']})")
            st.json(out)
        except requests.RequestException as e:
            st.error(f"No se pudo completar: {e}")

with tab3:
    if rol == "admin":
        if st.button("Reentrenar con PC1 procesado"):
            try:
                r = post("/train")
                r.raise_for_status()
                st.success("Modelo actualizado")
                st.json(r.json())
            except requests.RequestException as e:
                st.error(str(e))
    else:
        st.info("El reentrenamiento lo ejecuta el administrador.")
    if rol in ("analista", "admin"):
        try:
            r = get("/model")
            r.raise_for_status()
            info = r.json()
            st.write("Version:", info.get("version_modelo"))
            st.write("Algoritmo:", info.get("algoritmo"))
            st.write("Hash dataset:", info.get("dataset_hash"))
            st.json(info.get("metrics", {}))
        except requests.RequestException as e:
            st.warning(str(e))
    else:
        st.caption("Las metricas de evaluacion estan visibles para analista y admin.")

with tab4:
    if st.button("Actualizar registros"):
        try:
            r = get("/records")
            r.raise_for_status()
            frame = pd.json_normalize(r.json())
            st.dataframe(frame, use_container_width=True)
            st.download_button(
                "Exportar CSV",
                frame.to_csv(index=False).encode("utf-8-sig"),
                "registros.csv",
                "text/csv",
            )
        except requests.RequestException as e:
            st.error(str(e))

with tab5:
    st.markdown(
        """
**Modelo de IA:** probabilidad de que un modulo tenga defectos, entrenada con metricas McCabe/Halstead de NASA PC1.

**Reglas de negocio (deterministas):**
- Umbrales: bajo < {medio}, medio >= {medio}, alto >= {alto}.
- Si el modulo es critico, un bajo se eleva a medio.
- Si la cobertura es menor a {cov:.0%}, LOC >= {loc:.0f} y v(g)/LOC >= {densidad:.3f}, el nivel pasa a alto.

**No hace el modelo:** bloquear un merge, asignar un revisor, ni decidir el semaforo final cuando hay overrides.
        """.format(
            medio=schema.get("umbrales_riesgo", {}).get("medio", 0.3),
            alto=schema.get("umbrales_riesgo", {}).get("alto", 0.6),
            cov=cfg.get("cobertura_minima", 0.5),
            loc=cfg.get("loc_sin_cobertura_alto", 200),
            densidad=cfg.get("densidad_complejidad_alta", 0.15),
        )
    )
    st.code(API + "/docs")
