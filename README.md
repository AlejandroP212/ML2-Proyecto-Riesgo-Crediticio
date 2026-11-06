# Clasificación de riesgo crediticio en Colombia con modelos de ensamble

Proyecto integrador de **Machine Learning II** — Universidad Externado de Colombia.
**Integrantes:** David Castro ([@Daveai012](https://github.com/Daveai012)) y Alejandro Pardo ([@AlejandroP212](https://github.com/AlejandroP212)).

Reproducción y adaptación al contexto colombiano del método de *Koshti et al. (2025), "Improving Credit Score
Classification Using Predictive Analysis"*, usando una **simulación fundamentada** de 6.000 solicitantes de crédito
calibrada con la informalidad laboral del DANE (≈54,5 %) y una tasa de mora del 8,5 %.

## Ejecutar

| Notebook | Contenido | Colab |
|---|---|---|
| `notebooks/01_simulacion_y_verificacion.ipynb` | Genera el dataset, verifica que sea idéntico al de la entrega y que reproduzca la estructura simulada (7.1). Figuras 1–2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlejandroP212/ML2-Proyecto-Riesgo-Crediticio/blob/main/notebooks/01_simulacion_y_verificacion.ipynb) |
| `notebooks/02_modelamiento_y_validacion.ipynb` | Pipeline sin fuga, comparación de modelos, GMM, GridSearch, CV anidada y comparación con el informe (7.2–7.4). Figuras 3–5 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlejandroP212/ML2-Proyecto-Riesgo-Crediticio/blob/main/notebooks/02_modelamiento_y_validacion.ipynb) |

En Colab basta con *Entorno de ejecución → Ejecutar todas*: el notebook clona el repositorio por sí solo.
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
  - la fila *GMM Detección de Anomalías* de la Tabla 2, que no tenía código asociado;
  - las curvas ROC y el guardado (`savefig`) de todas las figuras;
  - la desviación estándar de la validación cruzada del RF optimizado;
  - la validación cruzada anidada;
  - una tabla que compara cada cifra con la del informe. Las diferencias encontradas se explican al final del notebook 02.

## Uso de IA generativa

Se usó Claude (Anthropic) como apoyo para organizar el repositorio y redactar los notebooks de reproducibilidad.
El equipo revisó y puede sustentar cada decisión técnica.
