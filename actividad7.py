from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Keep input and output paths relative to this script.
CARPETA = Path(__file__).resolve().parent
SALIDA = CARPETA / "resultados"
SALIDA.mkdir(exist_ok=True)

datos = pd.read_csv(CARPETA / "Titanic-Dataset.csv")
# Select the outcome and the features used by the models.
variables = ["Sex", "Pclass", "Age", "Fare", "SibSp"]
X = datos[variables]
y = datos["Survived"]
# Use the same stratified train and test sets for both models.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

lineas = [
    "ACTIVIDAD 7: REGRESIÓN LOGÍSTICA - TITANIC",
    f"Pasajeros: {len(datos)}; entrenamiento: {len(X_train)}; prueba: {len(X_test)}",
    "Supervivencia: 0 = no sobrevivió, 1 = sobrevivió.",
    "Valores faltantes en las variables utilizadas:",
    datos[variables + ["Survived"]].isna().sum().to_string(),
]

modelos = {
    "Base": ["Pclass", "Age"],
    "Ampliado": ["Pclass", "Age", "Fare", "SibSp"],
}
metricas = []
probabilidades = pd.DataFrame({
    "PassengerId": datos.loc[X_test.index, "PassengerId"],
    "Supervivencia_real": y_test,
})
fig_roc, ax_roc = plt.subplots(figsize=(7, 5))

for nombre, numericas in modelos.items():
    # Learn missing-value replacements and encoding from training data only.
    preparacion = ColumnTransformer([
        ("numericas", SimpleImputer(strategy="median"), numericas),
        ("sexo", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
        ]), ["Sex"]),
    ])
    modelo = Pipeline([
        ("preparacion", preparacion),
        ("regresion", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    modelo.fit(X_train, y_train)
    # Select the probability of class 1 (survival).
    prob = modelo.predict_proba(X_test)[:, 1]
    probabilidades[f"Probabilidad_{nombre}"] = prob
    auc = roc_auc_score(y_test, prob)
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax_roc.plot(fpr, tpr, label=f"{nombre} (AUC = {auc:.3f})")

    # Save coefficients and their corresponding odds ratios.
    nombres = modelo.named_steps["preparacion"].get_feature_names_out()
    coef = modelo.named_steps["regresion"].coef_[0]
    tabla_coef = pd.DataFrame({"Variable": nombres, "Coeficiente": coef,
                               "Razon_de_odds": np.exp(coef)})
    tabla_coef.to_csv(SALIDA / f"coeficientes_{nombre.lower()}.csv", index=False)
    lineas += [f"\nMODELO {nombre.upper()}",
               f"Variables numéricas: {', '.join(numericas)}; categórica: Sex",
               f"Intercepto: {modelo.named_steps['regresion'].intercept_[0]:.4f}",
               "Coeficientes y razones de odds:", tabla_coef.to_string(index=False),
               f"AUC: {auc:.4f}"]

    for umbral in (0.5, 0.7):
        # Compare predictions and errors at each classification threshold.
        pred = (prob >= umbral).astype(int)
        matriz = confusion_matrix(y_test, pred, labels=[0, 1])
        metricas.append({"Modelo": nombre, "Umbral": umbral,
                         "Accuracy": accuracy_score(y_test, pred),
                         "Precision": precision_score(y_test, pred, zero_division=0),
                         "Recall": recall_score(y_test, pred, zero_division=0),
                         "F1": f1_score(y_test, pred, zero_division=0), "AUC": auc})
        lineas += [f"\nUmbral: {umbral}",
                   "Matriz (filas reales, columnas predichas; orden 0, 1):",
                   str(matriz), classification_report(y_test, pred, zero_division=0)]
        probabilidades[f"Prediccion_{nombre}_{str(umbral).replace('.', '_')}"] = pred
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(matriz, display_labels=["No sobrevivió", "Sobrevivió"]).plot(
            ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(f"Modelo {nombre}: umbral {umbral}")
        fig.tight_layout()
        fig.savefig(SALIDA / f"matriz_{nombre.lower()}_{str(umbral).replace('.', '_')}.png", dpi=150)
        plt.close(fig)

ax_roc.plot([0, 1], [0, 1], "k--", label="Azar")
ax_roc.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos",
           title="Curvas ROC en datos de prueba")
ax_roc.legend()
fig_roc.tight_layout()
fig_roc.savefig(SALIDA / "curvas_roc.png", dpi=150)
plt.close(fig_roc)

# Show observed survival rates by sex without implying causation.
tasas = datos.groupby("Sex")["Survived"].mean()
fig, ax = plt.subplots(figsize=(6, 4))
tasas.plot.bar(ax=ax, color=["#5995c4", "#e49b67"], rot=0)
ax.set(xlabel="Sexo registrado", ylabel="Proporción que sobrevivió",
       title="Supervivencia observada por sexo", ylim=(0, 1))
fig.tight_layout()
fig.savefig(SALIDA / "supervivencia_por_sexo.png", dpi=150)
plt.close(fig)

# Save tables and a readable summary for the report.
pd.DataFrame(metricas).to_csv(SALIDA / "comparacion_metricas.csv", index=False)
probabilidades.to_csv(SALIDA / "probabilidades_prueba.csv", index=False)
lineas += ["\nResumen de métricas:", pd.DataFrame(metricas).to_string(index=False),
           "\nLos coeficientes expresan cambios en log-odds por unidad, manteniendo las demás variables constantes.",
           "Sex_male compara hombres con mujeres (referencia). Un coeficiente positivo aumenta las odds; uno negativo las reduce.",
           "Las magnitudes de coeficientes con distintas unidades no son directamente comparables. Asociación no implica causalidad."]
texto = "\n".join(lineas)
(SALIDA / "resumen.txt").write_text(texto, encoding="utf-8")
print(texto)
print(f"\nResultados guardados en: {SALIDA}")
