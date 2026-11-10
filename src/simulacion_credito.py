"""
Simulación fundamentada de solicitantes de crédito en Colombia.

Genera el dataset de la Segunda entrega (6.000 registros, semilla 42). El resultado es
idéntico byte a byte a `entregas/entrega_2/datasets/credito_colombia_simulado.csv`, que se
generó con `entregas/entrega_2/generar_datos_simulados.py` (archivo original, sin modificar).
Este módulo reproduce exactamente las mismas operaciones, en el mismo orden, con comentarios
que explican cada distribución, cada parámetro y su fundamento.

Uso:
    python src/simulacion_credito.py            # escribe data/credito_colombia_simulado.csv

    from src.simulacion_credito import generar_dataset_credito_colombia
    df = generar_dataset_credito_colombia(n_samples=6000, seed=42)

Nota sobre reproducibilidad: se usa el generador global de NumPy (`np.random.seed`) porque así
se generó el dataset entregado. Cambiar el orden de las llamadas aleatorias cambia los datos.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------
# Parámetros de calibración (fuentes y supuestos)
# ---------------------------------------------------------------------------------------------
P_INFORMAL = 0.545          # Informalidad laboral nacional, DANE, abril-junio de 2026 (54,5 %)
PREVALENCIA_MORA = 0.085    # Tasa objetivo de incumplimiento. Base: cartera vencida del sistema
                            # (3,8 %, Superintendencia Financiera, marzo de 2026), ajustada al alza
                            # por el mayor riesgo del segmento informal (supuesto del equipo).

# Coeficientes del logit latente de incumplimiento (supuestos del equipo: fijan la dirección y el
# tamaño relativo de cada efecto; no se estimaron con datos reales). Ver `logit` más abajo.
COEF_LOGIT = {
    "intercepto": -2.40,
    "es_informal": 0.80,               # odds × e^0.80 ≈ 2,2 para un informal
    "relacion_cuota_ingreso": 1.30,    # +0,1 en la relación → odds × e^0.13 ≈ 1,14
    "antiguedad_laboral": -0.10,       # cada año de antigüedad → odds × e^-0.10 ≈ 0,90
    "score_centrado_600": -0.007,      # +100 puntos de score → odds × e^-0.70 ≈ 0,50
    "moras_previas": 0.50,             # cada mora previa → odds × e^0.50 ≈ 1,65
    "ingreso_smmlv": -0.12,            # +1 SMMLV de ingreso → odds × e^-0.12 ≈ 0,89
    "ruido_sd": 0.45,                  # heterogeneidad no observada N(0; 0,45)
}


def generar_dataset_credito_colombia(n_samples=6000, seed=42):
    """Devuelve un DataFrame con 10 predictores y la variable objetivo `incumple`."""
    np.random.seed(seed)

    # 1. Demografía y tipo de empleo -----------------------------------------------------------
    # Edad uniforme entre 18 y 69 años. No entra en el logit: es una variable sin efecto real
    # sobre la mora (sirve para comprobar que el modelo no le da importancia).
    edad = np.random.randint(18, 70, size=n_samples)

    # Informalidad ~ Bernoulli(p = 0,545). Es la variable que condiciona a casi todas las demás.
    es_informal = np.random.choice([1, 0], size=n_samples, p=[P_INFORMAL, 1 - P_INFORMAL])

    # 2. Antigüedad laboral (años) ~ Exponencial ------------------------------------------------
    # Muchos con poca antigüedad y pocos con mucha. `scale` es la MEDIA (= 1/λ):
    # 2,0 años para informales (mayor rotación) y 5,5 años para formales.
    # Nota: np.where evalúa las dos ramas, así que se generan n valores de cada distribución.
    antiguedad = np.where(
        es_informal == 1,
        np.random.exponential(scale=2.0, size=n_samples),
        np.random.exponential(scale=5.5, size=n_samples),
    ).round(1)

    # 3. Ingreso mensual (múltiplos del SMMLV) ~ Lognormal -------------------------------------
    # Positivo y con cola larga a la derecha. `mean` y `sigma` son la media y la desviación del
    # LOGARITMO del ingreso: la mediana es e^mean → e^0,55 ≈ 1,7 SMMLV (informal) y
    # e^1,10 ≈ 3,0 SMMLV (formal), lo que reproduce la brecha salarial por tipo de empleo.
    ingreso_smmlv = np.where(
        es_informal == 1,
        np.random.lognormal(mean=0.55, sigma=0.45, size=n_samples),
        np.random.lognormal(mean=1.10, sigma=0.55, size=n_samples),
    ).round(2)

    # 4. Crédito solicitado ---------------------------------------------------------------------
    # Monto ~ Lognormal con mediana e^1,4 ≈ 4,1 SMMLV; plazo uniforme entre 5 plazos comerciales.
    monto_credito_smmlv = np.random.lognormal(mean=1.4, sigma=0.65, size=n_samples).round(2)
    plazo_meses = np.random.choice([12, 24, 36, 48, 60], size=n_samples)

    # 5. Historial crediticio ~ Poisson (conteos de eventos poco frecuentes) ---------------------
    # Créditos activos: media 1,4 para todos (no entra en el logit).
    # Moras previas: media 0,75 (informal) frente a 0,25 (formal).
    creditos_activos = np.random.poisson(lam=1.4, size=n_samples)
    moras_previas = np.where(
        es_informal == 1,
        np.random.poisson(lam=0.75, size=n_samples),
        np.random.poisson(lam=0.25, size=n_samples),
    )

    # 6. Score crediticio ~ Normal truncada a [300, 850] (rango de los burós) --------------------
    # Media 585 (informal, menos historial bancario) frente a 675 (formal).
    score_crediticio = np.where(
        es_informal == 1,
        np.random.normal(loc=585, scale=75, size=n_samples),
        np.random.normal(loc=675, scale=70, size=n_samples),
    ).clip(300, 850).round(0)

    # 7. Capacidad de pago ----------------------------------------------------------------------
    # Cuota mensual aproximada = (monto / plazo) × 1,2  (el 1,2 representa los intereses).
    # Relación cuota/ingreso, con un ingreso mínimo de 0,5 SMMLV para evitar divisiones extremas.
    cuota_estimada_smmlv = (monto_credito_smmlv / plazo_meses) * 1.2
    relacion_cuota_ingreso = (cuota_estimada_smmlv / np.maximum(ingreso_smmlv, 0.5)).round(3)

    # 8. Variable objetivo: logit latente de incumplimiento -------------------------------------
    c = COEF_LOGIT
    logit = (
        c["intercepto"]
        + c["es_informal"] * es_informal
        + c["relacion_cuota_ingreso"] * relacion_cuota_ingreso
        + c["antiguedad_laboral"] * antiguedad
        + c["score_centrado_600"] * (score_crediticio - 600)
        + c["moras_previas"] * moras_previas
        + c["ingreso_smmlv"] * ingreso_smmlv
        + np.random.normal(0, c["ruido_sd"], size=n_samples)
    )
    prob_incumplimiento = 1 / (1 + np.exp(-logit))

    # Se marca como "incumple" al 8,5 % de solicitantes con MAYOR probabilidad (percentil 91,5).
    # Así la prevalencia queda fijada exactamente en 8,5 %. Es un corte determinista, no un
    # sorteo Bernoulli: por eso la mora queda muy concentrada en los perfiles de mayor riesgo.
    corte = np.percentile(prob_incumplimiento, 91.5)  # 91,5 = 100 × (1 − 0,085)
    incumple = (prob_incumplimiento > corte).astype(int)

    df = pd.DataFrame({
        "edad": edad,
        "es_informal": es_informal,
        "antiguedad_laboral": antiguedad,
        "ingreso_smmlv": ingreso_smmlv,
        "monto_credito_smmlv": monto_credito_smmlv,
        "plazo_meses": plazo_meses,
        "creditos_activos": creditos_activos,
        "moras_previas": moras_previas,
        "score_crediticio": score_crediticio,
        "relacion_cuota_ingreso": relacion_cuota_ingreso,
        "incumple": incumple,
    })

    # 9. Valores faltantes completamente al azar (MCAR) ----------------------------------------
    # Se introducen DESPUÉS de calcular la mora (la mora usa los valores completos) para exigir
    # imputación en el preprocesamiento: ingreso 4 %, antigüedad 5 %, score 3 %.
    df.loc[np.random.rand(n_samples) < 0.04, "ingreso_smmlv"] = np.nan
    df.loc[np.random.rand(n_samples) < 0.05, "antiguedad_laboral"] = np.nan
    df.loc[np.random.rand(n_samples) < 0.03, "score_crediticio"] = np.nan

    return df


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parents[1]
    salida = raiz / "data" / "credito_colombia_simulado.csv"
    salida.parent.mkdir(exist_ok=True)
    df = generar_dataset_credito_colombia(n_samples=6000, seed=42)
    df.to_csv(salida, index=False)
    print(f"Dataset generado en {salida.relative_to(raiz)}: {df.shape}")
    print(f"Tasa de incumplimiento: {df['incumple'].mean():.2%}")
    print(f"Tasa de informalidad:   {df['es_informal'].mean():.2%}")
