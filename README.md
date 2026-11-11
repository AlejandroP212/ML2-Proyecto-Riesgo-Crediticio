# Clasificación de riesgo crediticio en Colombia con modelos de ensamble

Proyecto integrador de **Machine Learning II** — Universidad Externado de Colombia.
**Integrantes:** David Castro ([@Daveai012](https://github.com/Daveai012)) y Alejandro Pardo ([@AlejandroP212](https://github.com/AlejandroP212)).

Reproducción y adaptación al contexto colombiano del método de *Koshti et al. (2025), "Improving Credit Score
Classification Using Predictive Analysis"*, usando una **simulación fundamentada** de 6.000 solicitantes de crédito
calibrada con la informalidad laboral del DANE (≈54,5 %) y una tasa de mora del 8,5 %.

## Ejecutar

| Notebook | Contenido | Colab |
|---|---|---|
| `notebooks/01_simulacion_y_verificacion.ipynb` | Construye la simulación paso a paso (código completo y justificación de cada distribución y parámetro), verifica que el dataset sea idéntico al de la entrega y que reproduzca la estructura prevista (7.1). Figuras 1–2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlejandroP212/ML2-Proyecto-Riesgo-Crediticio/blob/main/notebooks/01_simulacion_y_verificacion.ipynb) |
| `notebooks/02_modelamiento_y_validacion.ipynb` | Pipeline sin fuga, comparación de modelos supervisados, GridSearch, CV anidada y comparación con el informe (7.2–7.4). Figuras 3–5 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlejandroP212/ML2-Proyecto-Riesgo-Crediticio/blob/main/notebooks/02_modelamiento_y_validacion.ipynb) |
| `notebooks/03_comparacion_equitativa.ipynb` | Compara los nueve modelos del artículo en igualdad de condiciones: pesos de clase, GridSearch y CV anidada para cada uno, importancia por permutación y errores por segmento (7.3–7.4 y 8.1). Figuras 6, 7 y 9 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlejandroP212/ML2-Proyecto-Riesgo-Crediticio/blob/main/notebooks/03_comparacion_equitativa.ipynb) |
| `notebooks/04_busqueda_hiperparametros.ipynb` | Compara los tres buscadores de hiperparámetros del curso (Grid, Random y Optuna-TPE) con el mismo presupuesto y los mismos pliegues para Random Forest y Gradient Boosting (7.3). Figura 8 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlejandroP212/ML2-Proyecto-Riesgo-Crediticio/blob/main/notebooks/04_busqueda_hiperparametros.ipynb) |
| `notebooks/05_modelo_final.ipynb` | Modelo final (Regresión Logística): selección justificada, curva de validación, umbral elegido sin usar la prueba, evaluación única en prueba, contraste con el artículo y artefactos para el despliegue (8.1). Figuras 10–12 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlejandroP212/ML2-Proyecto-Riesgo-Crediticio/blob/main/notebooks/05_modelo_final.ipynb) |
| `notebooks/06_interpretacion.ipynb` | Interpretación del modelo final: coeficientes como odds ratios con intervalos bootstrap, SHAP global y local, dependencia parcial e ICE, y perfil de los errores con sus implicaciones (8.2). Figuras 13–17 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlejandroP212/ML2-Proyecto-Riesgo-Crediticio/blob/main/notebooks/06_interpretacion.ipynb) |

En Colab basta con *Entorno de ejecución → Ejecutar todas*: el notebook clona el repositorio por sí solo.

El código de la simulación también está como módulo en `src/simulacion_credito.py`; `python src/simulacion_credito.py`
regenera `data/credito_colombia_simulado.csv` sin abrir notebooks.

En local:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```

Todo usa la semilla `42`. Las figuras se guardan en `figuras/` y las tablas en `resultados/`. Los notebooks 01 y 02
tardan menos de un minuto; el 03 tarda unos 7 minutos por la validación cruzada anidada de los nueve modelos, el 04
unos 5 minutos, y el 05 y el 06 menos de un minuto cada uno. El 05 usa la tabla que genera el 03, y el 06 usa el
modelo guardado por el 05.

## Aplicación (despliegue básico, Entrega final 8.3)

La aplicación en Streamlit recibe los datos de un solicitante, aplica el mismo Pipeline del entrenamiento, devuelve la
probabilidad estimada de mora y la decisión del modelo, y explica qué variables pesan en ese resultado.

```bash
pip install -r requirements.txt
python src/despliegue.py          # verificación de coherencia entre entrenamiento y uso
streamlit run app/app.py          # abre la aplicación en http://localhost:8501
```

Ambos comandos se ejecutan desde la raíz del repositorio. La aplicación usa los artefactos de `modelos/` (generados por el
notebook 05) y la lógica de `src/despliegue.py`, que comparte con la verificación:

| Qué hace | Cómo garantiza la coherencia con el entrenamiento |
|---|---|
| Construye la fila de entrada | Mismo orden de variables y misma fórmula de `relacion_cuota_ingreso` que la simulación |
| Datos faltantes | Si el usuario marca "no disponible", el Pipeline imputa la mediana del entrenamiento, como en el modelamiento |
| Valida las entradas | Avisa cuando un valor está fuera del rango visto en el entrenamiento (contrato de entrada) |
| Probabilidad | Corregida por la tasa real de mora (8,5 %), porque el modelo se entrenó con pesos balanceados |
| Decisión | Umbral elegido en validación cruzada (notebook 05); marca los casos límite para revisión humana |
| Explicación | Contribución de cada variable, idéntica a los valores SHAP del modelo |
| Al iniciar | Comprueba que el modelo reproduce la predicción de referencia y que la versión de scikit-learn coincide |

## Estructura

```
├── articulo/                  Artículo base (Koshti et al., 2025)
├── entregas/                  ARCHIVOS ORIGINALES, sin modificar
│   ├── entrega_1/             Informe de la Primera entrega (.docx)
│   └── entrega_2/             Informe PDF, notebook, generador, dataset y figuras de la Segunda entrega
├── src/                       simulacion_credito.py, modelos_balanceados.py y despliegue.py (lógica de la app)
├── modelos/                   Modelo final entrenado (.joblib), su contrato de entrada y una predicción de referencia
├── app/                       Aplicación de Streamlit (app.py)
├── notebooks/                 Notebooks reproducibles (creados después de la entrega, ver abajo)
├── data/                      Dataset regenerado por el notebook 01
├── figuras/  resultados/      Salidas de los notebooks
└── CHECKSUMS.txt              SHA-256 y fecha local de cada archivo original
```

## Trazabilidad de las entregas

- La carpeta **`entregas/`** contiene los archivos **tal como se entregaron**. Se subieron en el primer commit, marcado con
  la etiqueta [`entrega-2`](../../tree/entrega-2), y no se han modificado desde entonces.
- `CHECKSUMS.txt` registra el hash SHA-256 y la fecha de modificación local de cada original. GitHub no conserva las fechas
  de los archivos, así que para comprobar que un archivo es idéntico al original se compara su hash:
  `sha256sum -c <(grep -v '^#' CHECKSUMS.txt | cut -d'|' -f1 | sed 's/ *$//')`.
- Los notebooks 01 y 02 se crearon en **noviembre de 2026** para que cada cifra y figura del informe tenga código
  que la genere. Respecto al notebook original agregan:
  - las curvas ROC y el guardado (`savefig`) de todas las figuras;
  - la desviación estándar de la validación cruzada del RF optimizado;
  - la validación cruzada anidada;
  - una tabla que compara cada cifra con la del informe. Las diferencias encontradas se explican al final del notebook 02.
- Los notebooks 03 y 04 y los módulos de `src/` se agregaron después de la retroalimentación de la Segunda entrega
  (ver la sección siguiente). Cada cambio está en su propio commit.

## Cambios después de la retroalimentación de la Segunda entrega

1. **Código de la simulación visible.** El notebook 01 ya no importa el generador: muestra el código completo, variable
   por variable, con la distribución, los parámetros, el fundamento de cada uno y la lectura de los coeficientes del logit.
   El mismo código queda en `src/simulacion_credito.py`. Ambos producen un archivo idéntico (mismo SHA-256) al dataset
   entregado. Se documenta además una limitación de la simulación: la mora del segmento formal es casi nula.
2. **Se retira el GMM de la comparación.** El detector de anomalías no supervisado no usa la etiqueta de mora, así que
   compararlo con clasificadores supervisados no es una comparación en igualdad de condiciones; se eliminó del notebook 02
   y de `resultados/tabla2_comparativa_modelos.csv`. Su fila del informe no tenía código asociado.
   El notebook 02 también señala que las importancias citadas en el texto del informe (sección 4.3) no coinciden con su
   Figura 5 ni con el código.
3. **Comparación en igualdad de condiciones (notebook 03).** En la entrega solo Random Forest tuvo pesos de clase y
   ajuste de hiperparámetros. Ahora los cuatro modelos supervisados tienen pesos balanceados (AdaBoost y Gradient Boosting
   mediante `src/modelos_balanceados.py`), su propio GridSearch y CV anidada con los mismos pliegues. Resultado: Random
   Forest (0,891 ± 0,011), Gradient Boosting (0,888 ± 0,007) y AdaBoost (0,880 ± 0,015) quedan parejos; el árbol queda por
   debajo (0,857 ± 0,016). La ventaja de Random Forest en la entrega se debía sobre todo al tratamiento del desbalance.
   Resultados en `resultados/tabla3_comparacion_equitativa.csv` y `resultados/importancia_permutacion.csv`.
4. **Búsqueda de hiperparámetros con los métodos del curso (notebook 04).** No busca cambiar los hiperparámetros, sino
   **comprobar que la rejilla estaba bien armada**. `GridSearchCV` solo prueba los valores que uno elige (por ejemplo,
   profundidades 6, 8 y 12): si el mejor valor estuviera entre esos puntos, la rejilla no lo encontraría.
   `RandomizedSearchCV` y Optuna (TPE), vistos en la semana 5, exploran todo el rango, y Optuna además aprende de los
   intentos anteriores. Se comparan los tres con el mismo presupuesto (24 y 27 configuraciones), el mismo espacio y los
   mismos pliegues. Las diferencias entre buscadores (≤ 0,004) son menores que la desviación entre pliegues y los tres
   llegan a la misma región de hiperparámetros: **los hiperparámetros elegidos no dependen de cómo se armó la rejilla**.
   Optuna queda listo para la entrega final, donde modelos con más hiperparámetros (XGBoost, LightGBM) harían inviable
   una rejilla completa.
   Resultados en `resultados/tabla4_buscadores_hiperparametros.csv`. Se agrega `optuna` a `requirements.txt`.
5. **Comparación con los nueve modelos del artículo (notebook 03, Entrega final 8.1).** Se agregan Regresión
   Logística, Naïve Bayes, SVM, XGBoost y LightGBM, con las mismas condiciones (pesos de clase, GridSearch y CV anidada
   con los mismos pliegues). Siete de los nueve modelos quedan empatados (0,880–0,892 de balanced accuracy en CV anidada,
   diferencias menores que la variación entre pliegues; Figura 9); el árbol (0,857) y Naïve Bayes (0,784) quedan por
   debajo, como en el artículo. La Regresión Logística obtiene la mejor media (0,892 ± 0,008): la mora se simuló con un
   logit, así que un modelo lineal tiene la forma correcta y los ensambles no tienen no linealidades que aprovechar.
   Se agregan `xgboost` y `lightgbm` a `requirements.txt`.
6. **Modelo final (notebook 05, Entrega final 8.1).** Se elige la **Regresión Logística** (`C = 0,1`,
   `class_weight='balanced'`): mayor balanced accuracy en CV anidada entre siete modelos empatados y el más interpretable
   (regla de parsimonia). Su desempeño es estable frente a C (curva de validación) y el umbral de decisión (0,44) se
   eligió con predicciones *out-of-fold* del entrenamiento, sin usar la prueba. En prueba: balanced accuracy 0,897,
   95 de 102 morosos detectados, ROC-AUC 0,963. Se contrasta con el artículo (`resultados/tabla6_contraste_articulo.csv`)
   y se guardan los artefactos para el despliegue en `modelos/`, con verificación de que el modelo cargado predice igual.
7. **Interpretación (notebook 06, Entrega final 8.2).** Coeficientes como odds ratios con intervalos bootstrap, SHAP
   global y local (exacto para un modelo lineal), dependencia parcial e ICE, y perfil de los errores. Las moras previas,
   la informalidad y la relación cuota/ingreso suben el riesgo; el score, la antigüedad y el ingreso lo bajan; las
   variables de control no tienen efecto. El modelo recupera la estructura de la simulación (coeficientes proporcionales
   a los simulados). Las 150 falsas alarmas son informales que pagan, con un perfil casi igual al de los morosos.
   Se agrega `shap` a `requirements.txt`.
8. **Aplicación en Streamlit (`app/`, Entrega final 8.3).** Recibe los datos del solicitante, predice con el Pipeline
   guardado, muestra la probabilidad estimada de mora y la decisión, avisa sobre datos faltantes o fuera de rango, marca
   los casos límite y explica la predicción con contribuciones equivalentes a SHAP. `src/despliegue.py` concentra la lógica
   compartida y una verificación de coherencia (`python src/despliegue.py`). Se agrega `streamlit` a `requirements.txt`.

## Uso de IA generativa

Se usó Claude (Anthropic) como apoyo para organizar el repositorio, redactar los notebooks de reproducibilidad y
preparar los cambios posteriores a la retroalimentación.
El equipo revisó y puede sustentar cada decisión técnica.
