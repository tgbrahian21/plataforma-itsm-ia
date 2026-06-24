import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Plataforma ITSM - IA", layout="wide")

st.title("Plataforma ITSM Asistida por IA")
st.markdown("Sistema automatizado para la clasificación de incidencias y evaluación de SLA.")
st.divider()


col_izq, col_der = st.columns([1, 2])

with col_izq:
    st.subheader("Configuración")
    modelo_visual = st.radio("Seleccione el motor:", ["Baseline (Random Forest)", "Retador (XGBoost)"], index=1)
    modelo_api = "xgboost" if "XGBoost" in modelo_visual else "baseline"
    
    st.markdown("---")
    descripcion_ticket = st.text_area(
        "Ingrese el reporte del usuario (Inglés):",
        height=200,
        placeholder="The server is completely down..."
    )
    btn_ejecutar = st.button("Ejecutar Análisis", type="primary", use_container_width=True)

with col_der:
    st.subheader("Panel de Resultados y Métricas")
    
    if btn_ejecutar and descripcion_ticket.strip():
        with st.spinner("Ejecutando pipeline de NLP y redes tensoriales..."):
            try:
                res = requests.post("http://127.0.0.1:8000/predict", json={"descripcion": descripcion_ticket, "modelo_elegido": modelo_api})
                
                if res.status_code == 200:
                    datos = res.json()
                    
                    
                    tab_cat, tab_sla, tab_datos = st.tabs(["Clasificación Temática", "Riesgo de SLA", "Tabla de Datos Puros"])
                    
                    with tab_cat:
                        st.markdown(f"**Categoría Principal:** `{datos['categoria_predicha']}`")
                        
                        df_top = pd.DataFrame({
                            "Categoría": list(datos["top_categorias"].keys()),
                            "Probabilidad (%)": list(datos["top_categorias"].values())
                        })
                        fig_bar = px.bar(df_top, x="Probabilidad (%)", y="Categoría", orientation='h', 
                                         title="Top 3 Probabilidades de Categoría", color="Probabilidad (%)", color_continuous_scale="Blues")
                        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_bar, use_container_width=True)

                    with tab_sla:
                        riesgo = datos["riesgo_sla_porcentaje"]
                        color_alerta = "red" if riesgo >= 50 else "green"
                        
                        
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = riesgo,
                            title = {'text': "Riesgo de Incumplimiento SLA (%)"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': color_alerta},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgreen"},
                                    {'range': [50, 75], 'color': "orange"},
                                    {'range': [75, 100], 'color': "salmon"}
                                ],
                                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                            }
                        ))
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    with tab_datos:
                        st.markdown("### Tabla Consolidada de Veredictos")
    
                        df_resultados = pd.DataFrame({
                            "Métrica": ["Motor Analítico", "Categoría Asignada", "Confianza Categoría", "Alerta SLA", "Riesgo SLA Calculado"],
                            "Valor": [datos['modelo_usado'].upper(), datos['categoria_predicha'], f"{datos['confianza_categoria']*100:.1f}%", datos['prediccion_incumplimiento_sla'], f"{riesgo}%"]
                        })
                        st.dataframe(df_resultados, use_container_width=True, hide_index=True)
                        
                else:
                    st.error("Error en el servidor.")
            except Exception as e:
                st.error(f"Error de conexión: {e}")