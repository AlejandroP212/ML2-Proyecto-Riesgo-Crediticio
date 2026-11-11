"""
Lógica de inferencia del modelo final, compartida por la aplicación (app/app.py) y la verificación.

Reúne en un solo lugar lo que debe ser idéntico entre entrenamiento y uso:
- el orden de las variables y el contrato de entrada (rangos vistos en entrenamiento),
- el cálculo de la variable derivada `relacion_cuota_ingreso`, con la misma fórmula de la simulación,
- la corrección de la probabilidad por la tasa real de mora (el modelo se entrenó con pesos balanceados),
- la explicación de cada predicción (contribuciones exactas del modelo lineal, equivalentes a SHAP).

Verificación desde la terminal:
    python src/despliegue.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DIR_MODELOS = RAIZ / "modelos"

NOMBRES = {
    "edad": "Edad",
    "es_informal": "Empleo informal",
    "antiguedad_laboral": "Antigüedad laboral",
    "ingreso_smmlv": "Ingreso",
    "monto_credito_smmlv": "Monto del crédito",
    "plazo_meses": "Plazo",
    "creditos_activos": "Créditos activos",
    "moras_previas": "Moras previas",
    "score_crediticio": "Score crediticio",
    "relacion_cuota_ingreso": "Relación cuota/ingreso",
}


def cargar_artefactos(directorio=DIR_MODELOS):
    """Carga el Pipeline entrenado, sus metadatos y la predicción de referencia."""
    directorio = Path(directorio)
    modelo = joblib.load(directorio / "modelo_final.joblib")
    meta = json.loads((directorio / "modelo_final_metadata.json").read_text(encoding="utf-8"))
    referencia = json.loads((directorio / "prediccion_referencia.json").read_text(encoding="utf-8"))
    return modelo, meta, referencia


def relacion_cuota_ingreso(monto_credito_smmlv, plazo_meses, ingreso_smmlv):
    """Misma fórmula de src/simulacion_credito.py: cuota = monto / plazo × 1,2; piso de ingreso de 0,5 SMMLV."""
    if ingreso_smmlv is None or pd.isna(ingreso_smmlv):
        return np.nan
    cuota = (monto_credito_smmlv / plazo_meses) * 1.2
    return round(cuota / max(ingreso_smmlv, 0.5), 3)


def construir_registro(datos, meta):
    """Arma la fila de entrada en el orden del entrenamiento y calcula la variable derivada.
    Los datos faltantes se pasan como None: el Pipeline los imputa con la mediana del entrenamiento."""
    fila = {k: (np.nan if datos.get(k) is None else float(datos[k])) for k in meta["orden_variables"]
            if k != "relacion_cuota_ingreso"}
    fila["relacion_cuota_ingreso"] = relacion_cuota_ingreso(
        fila["monto_credito_smmlv"], fila["plazo_meses"], fila["ingreso_smmlv"])
    return pd.DataFrame([fila])[meta["orden_variables"]]


def validar(registro, meta):
    """Compara cada valor con el contrato de entrada. Devuelve una lista de advertencias (vacía si todo está en rango)."""
    avisos = []
    for col, regla in meta["contrato_entrada"].items():
        v = registro.iloc[0][col]
        if pd.isna(v):
            avisos.append(f"{NOMBRES[col]}: dato no disponible; se usa la mediana del entrenamiento ({regla['mediana']:g}).")
        elif "valores" in regla and v not in regla["valores"]:
            avisos.append(f"{NOMBRES[col]}: el valor {v:g} no se vio en el entrenamiento (valores válidos: {regla['valores']}).")
        elif not regla["min"] <= v <= regla["max"]:
            avisos.append(f"{NOMBRES[col]}: {v:g} está fuera del rango del entrenamiento "
                          f"({regla['min']:g} a {regla['max']:g}); la predicción es una extrapolación.")
    return avisos


def prob_corregida(p, prior):
    """Lleva la probabilidad del modelo (entrenado con clases balanceadas) a la escala de la mora real."""
    odds = p / (1 - p) * prior / (1 - prior)
    return odds / (1 + odds)


def predecir(modelo, registro, meta):
    p = float(modelo.predict_proba(registro)[:, 1][0])
    return {"probabilidad_modelo": p,
            "probabilidad_mora": float(prob_corregida(p, meta["prior_mora_entrenamiento"])),
            "rechazar": p >= meta["umbral_decision"]}


def explicar(modelo, registro):
    """Contribución de cada variable al riesgo, en log-odds, frente al solicitante promedio del entrenamiento.
    Como el escalador centra las variables en la media del entrenamiento, la contribución es coeficiente × valor
    estandarizado: coincide exactamente con SHAP (LinearExplainer) para este modelo."""
    z = modelo[:-1].transform(registro)[0]
    contrib = modelo[-1].coef_[0] * z
    return pd.Series(contrib, index=registro.columns).sort_values(key=np.abs, ascending=False)


def verificar(directorio=DIR_MODELOS, datos=RAIZ / "data" / "credito_colombia_simulado.csv"):
    """Comprobaciones de coherencia entre entrenamiento y uso."""
    modelo, meta, ref = cargar_artefactos(directorio)
    resultados = {}

    entrada = pd.DataFrame([ref["entrada"]])[meta["orden_variables"]]
    p = float(modelo.predict_proba(entrada)[:, 1][0])
    resultados["Predicción de referencia reproducida"] = abs(p - ref["probabilidad_modelo"]) <= ref["tolerancia"]

    datos_ref = {k: v for k, v in ref["entrada"].items() if k != "relacion_cuota_ingreso"}
    registro = construir_registro(datos_ref, meta)
    resultados["La app construye la misma fila que el entrenamiento"] = np.allclose(
        registro.values, entrada.values, equal_nan=True)

    if Path(datos).exists():
        df = pd.read_csv(datos).dropna(subset=["ingreso_smmlv"])
        calculada = [relacion_cuota_ingreso(m, pl, i) for m, pl, i in
                     zip(df.monto_credito_smmlv, df.plazo_meses, df.ingreso_smmlv)]
        resultados["Fórmula de cuota/ingreso igual a la de los datos"] = np.allclose(
            calculada, df.relacion_cuota_ingreso, atol=1e-3)

    import sklearn
    resultados[f"Versión de scikit-learn ({sklearn.__version__}) igual a la del entrenamiento"] = (
        sklearn.__version__ == meta["versiones"]["scikit-learn"])
    return resultados


if __name__ == "__main__":
    todo_ok = True
    for nombre, ok in verificar().items():
        print(("✔ " if ok else "✘ ") + nombre)
        todo_ok &= bool(ok)
    raise SystemExit(0 if todo_ok else 1)
