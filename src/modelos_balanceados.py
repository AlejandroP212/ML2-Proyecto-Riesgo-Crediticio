"""
Versiones de AdaBoost y Gradient Boosting que ponderan las clases como `class_weight='balanced'`.

`AdaBoostClassifier` y `GradientBoostingClassifier` de scikit-learn no tienen el parámetro
`class_weight`, pero sí aceptan `sample_weight` al entrenar. Estas subclases calculan ese peso
dentro de `fit` con la misma fórmula que usa `class_weight='balanced'`:

    peso de cada registro = n_total / (n_clases × n_registros_de_su_clase)

Con 8,5 % de mora, cada moroso pesa ≈ 5,9 y cada cliente al día ≈ 0,55. Calcular el peso dentro de
`fit` garantiza que, en validación cruzada y en GridSearchCV, se use solo la proporción de clases
del pliegue de entrenamiento (sin fuga de información). Los demás parámetros son los de la clase
original, así que funcionan igual con Pipeline, GridSearchCV y clone.
"""

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


class AdaBoostBalanceado(AdaBoostClassifier):
    """AdaBoostClassifier con pesos de clase balanceados."""

    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y))


class GradientBoostingBalanceado(GradientBoostingClassifier):
    """GradientBoostingClassifier con pesos de clase balanceados."""

    def fit(self, X, y, sample_weight=None, monitor=None):
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y), monitor=monitor)
