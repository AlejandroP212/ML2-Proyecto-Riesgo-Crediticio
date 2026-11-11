"""
Aplicación de despliegue básico del modelo final (Entrega final, 8.3).

Ejecutar desde la raíz del repositorio:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.despliegue import (NOMBRES, cargar_artefactos, construir_registro, explicar, predecir,  # noqa: E402
                            prob_corregida, validar, verificar)

st.set_page_config(page_title="Riesgo crediticio — ML II", page_icon="📊", layout="wide")


@st.cache_resource
def recursos():
    modelo, meta, referencia = cargar_artefactos()
    return modelo, meta, verificar()


modelo, meta, chequeos = recursos()
umbral_prob = prob_corregida(meta["umbral_decision"], meta["prior_mora_entrenamiento"])
# Zona de revisión: probabilidades entre la mitad y 1,5 veces el umbral (casos límite, notebook 06).
zona_baja, zona_alta = 0.5 * umbral_prob, 1.5 * umbral_prob

# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Modelo")
    st.markdown(
        f"**Regresión Logística** con pesos de clase balanceados (C = {meta['hiperparametros']['C']}).\n\n"
        f"- Balanced accuracy en CV anidada: **{meta['metricas']['cv_anidada_balanced_accuracy']:.3f}**\n"
        f"- Balanced accuracy en prueba: **{meta['metricas']['prueba']['Balanced accuracy']:.3f}**\n"
        f"- Morosos detectados en prueba: **{meta['metricas']['prueba']['Morosos detectados']}**\n"
        f"- Umbral de decisión: **{umbral_prob:.1%}** de probabilidad estimada de mora"
    )
    st.subheader("Verificación al iniciar")
    for nombre, ok in chequeos.items():
        st.markdown(("✅ " if ok else "❌ ") + nombre)
    st.caption("Proyecto académico con datos simulados. No usar para decisiones reales de crédito.")

# ---------------------------------------------------------------- formulario
st.title("Evaluación de riesgo crediticio")
st.write("Ingrese los datos del solicitante. El modelo estima la probabilidad de que incumpla (mora mayor a 90 días) "
         "y explica qué variables pesan en su resultado.")

contrato = meta["contrato_entrada"]
with st.form("solicitud"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Perfil laboral")
        empleo = st.radio("Tipo de empleo", ["Informal", "Formal"], horizontal=True)
        antig_nd = st.checkbox("Antigüedad no disponible")
        antig = st.number_input("Antigüedad en el empleo actual (años)", 0.0, 40.0, 2.0, 0.5, disabled=antig_nd)
        ingreso_nd = st.checkbox("Ingreso no disponible")
        ingreso = st.number_input("Ingreso mensual (en salarios mínimos)", 0.1, 30.0, 1.9, 0.1, disabled=ingreso_nd)
        edad = st.slider("Edad", 18, 69, 30)
    with c2:
        st.subheader("Historial crediticio")
        score_nd = st.checkbox("Score no disponible")
        score = st.number_input("Score crediticio (300 a 850)", 300, 850, 600, 5, disabled=score_nd)
        moras = st.number_input("Moras previas registradas", 0, 10, 0, 1)
        creditos = st.number_input("Créditos activos", 0, 10, 1, 1)
    with c3:
        st.subheader("Crédito solicitado")
        monto = st.number_input("Monto (en salarios mínimos)", 0.1, 100.0, 4.0, 0.1)
        plazo = st.selectbox("Plazo (meses)", contrato["plazo_meses"]["valores"], index=2)
    enviado = st.form_submit_button("Evaluar solicitud", type="primary", use_container_width=True)

if not enviado:
    st.info("Complete el formulario y presione **Evaluar solicitud**.")
    st.stop()

# ---------------------------------------------------------------- predicción
datos = {"edad": edad, "es_informal": 1 if empleo == "Informal" else 0,
         "antiguedad_laboral": None if antig_nd else antig, "ingreso_smmlv": None if ingreso_nd else ingreso,
         "monto_credito_smmlv": monto, "plazo_meses": plazo, "creditos_activos": creditos,
         "moras_previas": moras, "score_crediticio": None if score_nd else score}
registro = construir_registro(datos, meta)
resultado = predecir(modelo, registro, meta)
p = resultado["probabilidad_mora"]

st.divider()
r1, r2, r3 = st.columns(3)
r1.metric("Probabilidad estimada de mora", f"{p:.1%}" if p >= 0.001 else "< 0,1 %")
cuota = monto / plazo * 1.2
r2.metric("Cuota mensual estimada", f"{cuota:.2f} SMMLV")
rel = registro.iloc[0]["relacion_cuota_ingreso"]
r3.metric("Relación cuota/ingreso", "no disponible" if np.isnan(rel) else f"{rel:.0%}")

if resultado["rechazar"]:
    st.error(f"**Decisión del modelo: rechazar (riesgo alto).** La probabilidad supera el umbral de {umbral_prob:.1%}.")
else:
    st.success(f"**Decisión del modelo: aprobar.** La probabilidad está por debajo del umbral de {umbral_prob:.1%}.")
if zona_baja <= p <= zona_alta:
    st.warning(f"**Caso límite** (probabilidad entre {zona_baja:.1%} y {zona_alta:.1%}): se recomienda revisión "
               "por un analista antes de decidir.")

avisos = validar(registro, meta)
if avisos:
    with st.expander(f"⚠️ Coherencia con los datos de entrenamiento ({len(avisos)} aviso(s))", expanded=True):
        for a in avisos:
            st.write("- " + a)

# ---------------------------------------------------------------- explicación
st.subheader("¿Por qué este resultado?")
contrib = explicar(modelo, registro)
top = contrib.head(6)[::-1]
col_g, col_t = st.columns([3, 2])
with col_g:
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.barh([NOMBRES[c] for c in top.index], top.values, color=np.where(top.values > 0, "#c0392b", "#2b5c8f"))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Contribución al riesgo frente al solicitante promedio (log-odds)")
    ax.set_title("Rojo: sube el riesgo · Azul: lo baja", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
with col_t:
    fila = registro.iloc[0]

    def con_valor(c):
        v = fila[c]
        if np.isnan(v):
            texto = "no disponible"
        elif c == "es_informal":
            texto = "sí" if v == 1 else "no"
        elif c == "relacion_cuota_ingreso":
            texto = f"{v:.0%}"
        else:
            unidades = {"antiguedad_laboral": " años", "ingreso_smmlv": " SMMLV", "monto_credito_smmlv": " SMMLV",
                        "plazo_meses": " meses", "edad": " años"}
            texto = f"{v:g}{unidades.get(c, '')}"
        return f"{NOMBRES[c]} ({texto})"

    sube = [con_valor(c) for c, v in contrib.items() if v > 0.25]
    baja = [con_valor(c) for c, v in contrib.items() if v < -0.25]
    st.markdown("**Suben el riesgo:** " + (", ".join(sube) if sube else "ninguna variable de forma importante."))
    st.markdown("**Bajan el riesgo:** " + (", ".join(baja) if baja else "ninguna variable de forma importante."))
    st.caption("Las contribuciones se miden frente al solicitante promedio del entrenamiento y coinciden con los "
               "valores SHAP del modelo. Describen cómo decide el modelo, no relaciones causales.")

with st.expander("Limitaciones del modelo"):
    for lim in meta["limitaciones"]:
        st.write("- " + lim)
