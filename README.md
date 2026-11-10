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
| `notebooks/03_comparacion_equitativa.ipynb` | Compara todos los modelos supervisados en igualdad de condiciones: pesos de clase, GridSearch y CV anidada para cada uno, importancia por permutación y errores por segmento (7.3–7.4). Figuras 6–7 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlejandroP212/ML2-Proyecto-Riesgo-Crediticio/blob/main/notebooks/03_comparacion_equitativa.ipynb) |

En Colab basta con *Entorno de ejecución → Ejecutar todas*: el notebook clona el repositorio por sí solo.

El código de la simulación también está como módulo en `src/simulacion_credito.py`; `python src/simulacion_credito.py`
regenera `data/credito_colombia_simulado.csv` sin abrir notebooks.

En local:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```

Todo usa la semilla `42`. Las figuras se guardan en `figuras/` y las tablas en `resultados/`.

## Estructura

```
├── articulo/                  Artículo base (Koshti et al., 2025)
├── entregas/                  ARCHIVOS ORIGINALES, sin modificar
│   ├── entrega_1/             Informe de la Primera entrega (.docx)
│   └── entrega_2/             Informe PDF, notebook, generador, dataset y figuras de la Segunda entrega
├── src/                       simulacion_credito.py (simulación documentada) y modelos_balanceados.py
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
  `sha256sum -c <(grep -v '^#' CHECKSUMS.txt | cut -d'|' -f1)`.
- Los notebooks de `notebooks/` se crearon en **noviembre de 2026** para que cada cifra y figura del informe tenga código
  que la genere. Respecto al notebook original agregan:
  - las curvas ROC y el guardado (`savefig`) de todas las figuras;
  - la desviación estándar de la validación cruzada del RF optimizado;
  - la validación cruzada anidada;
  - una tabla que compara cada cifra con la del informe. Las diferencias encontradas se explican al final del notebook 02.

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

## Uso de IA generativa

Se usó Claude (Anthropic) como apoyo para organizar el repositorio y redactar los notebooks de reproducibilidad.
El equipo revisó y puede sustentar cada decisión técnica.
