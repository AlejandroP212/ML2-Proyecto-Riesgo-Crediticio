import numpy as np
import pandas as pd
import os

def generar_dataset_credito_colombia(n_samples=6000, seed=42):
    np.random.seed(seed)
    
    # 1. Demographics & Employment
    edad = np.random.randint(18, 70, size=n_samples)
    # Informalidad laboral en Colombia: ~54.5% (DANE)
    es_informal = np.random.choice([1, 0], size=n_samples, p=[0.545, 0.455])
    
    # Antigüedad laboral en años
    antiguedad = np.where(
        es_informal == 1,
        np.random.exponential(scale=2.0, size=n_samples),
        np.random.exponential(scale=5.5, size=n_samples)
    ).round(1)
    
    # Ingreso mensual en múltiplos de SMMLV (Lognormal)
    ingreso_smmlv = np.where(
        es_informal == 1,
        np.random.lognormal(mean=0.55, sigma=0.45, size=n_samples), # ~ 1.7 SMMLV
        np.random.lognormal(mean=1.10, sigma=0.55, size=n_samples)  # ~ 3.0 SMMLV
    ).round(2)
    
    # Monto del crédito en SMMLV y plazo
    monto_credito_smmlv = np.random.lognormal(mean=1.4, sigma=0.65, size=n_samples).round(2)
    plazo_meses = np.random.choice([12, 24, 36, 48, 60], size=n_samples)
    
    # Historial de crédito
    creditos_activos = np.random.poisson(lam=1.4, size=n_samples)
    moras_previas = np.where(
        es_informal == 1,
        np.random.poisson(lam=0.75, size=n_samples),
        np.random.poisson(lam=0.25, size=n_samples)
    )
    
    score_crediticio = np.where(
        es_informal == 1,
        np.random.normal(loc=585, scale=75, size=n_samples),
        np.random.normal(loc=675, scale=70, size=n_samples)
    ).clip(300, 850).round(0)
    
    # Relación Cuota/Ingreso estimada (Asumiendo tasa de interés promedio)
    # Cuota mensual aprox en SMMLV
    cuota_estimada_smmlv = (monto_credito_smmlv / plazo_meses) * 1.2
    relacion_cuota_ingreso = (cuota_estimada_smmlv / np.maximum(ingreso_smmlv, 0.5)).round(3)
    
    # Latent Logit para Probabilidad de Incumplimiento
    logit = (
        - 2.40
        + 0.80 * es_informal
        + 1.30 * relacion_cuota_ingreso
        - 0.10 * antiguedad
        - 0.007 * (score_crediticio - 600)
        + 0.50 * moras_previas
        - 0.12 * ingreso_smmlv
        + np.random.normal(0, 0.45, size=n_samples)
    )
    
    prob_incumplimiento = 1 / (1 + np.exp(-logit))
    # Tasa objetivo de morosidad global ~ 8.5% (Tasa mora Superfinanciera ajustada por informalidad)
    incumple = (prob_incumplimiento > np.percentile(prob_incumplimiento, 91.5)).astype(int)
    
    df = pd.DataFrame({
        'edad': edad,
        'es_informal': es_informal,
        'antiguedad_laboral': antiguedad,
        'ingreso_smmlv': ingreso_smmlv,
        'monto_credito_smmlv': monto_credito_smmlv,
        'plazo_meses': plazo_meses,
        'creditos_activos': creditos_activos,
        'moras_previas': moras_previas,
        'score_crediticio': score_crediticio,
        'relacion_cuota_ingreso': relacion_cuota_ingreso,
        'incumple': incumple
    })
    
    # Introducción de faltantes/nulos (Missingness del 3-5%) para exigir imputación
    mask_ingreso = np.random.rand(n_samples) < 0.04
    df.loc[mask_ingreso, 'ingreso_smmlv'] = np.nan
    
    mask_antiguedad = np.random.rand(n_samples) < 0.05
    df.loc[mask_antiguedad, 'antiguedad_laboral'] = np.nan
    
    mask_score = np.random.rand(n_samples) < 0.03
    df.loc[mask_score, 'score_crediticio'] = np.nan
    
    return df

if __name__ == "__main__":
    os.makedirs('datasets', exist_ok=True)
    df = generar_dataset_credito_colombia(n_samples=6000, seed=42)
    output_path = 'datasets/credito_colombia_simulado.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset generado exitosamente en: {output_path}")
    print(f"Dimensiones: {df.shape}")
    print(f"Tasa de Incumplimiento: {df['incumple'].mean():.2%}")
    print(f"Tasa de Informalidad: {df['es_informal'].mean():.2%}")
    print("Valores Nulos por Columna:\n", df.isnull().sum())
