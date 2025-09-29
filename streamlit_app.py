# =============================
# Proyecto de Inteligencia Artificial
# Clasificación por DOMINIO (ENAHO muestra 2022) - Streamlit con Gráficos
# =============================

import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC

# =============================
# 1. Configuración inicial
# =============================
st.title("📊 Clasificación por DOMINIO - ENAHO 2022")
st.write("Aplicación de modelos de **Regresión Logística, Árbol de Decisión y SVM**.")

@st.cache_data
def load_data():
    return pd.read_csv("Enaho01-2022-100.csv", low_memory=False, encoding="latin1")

df = load_data()
st.write("✅ Dataset cargado. Filas y columnas:", df.shape)
st.dataframe(df.head())

# =============================
# 2. Selección de variables
# =============================
df = df.dropna(subset=["DOMINIO", "ALTITUD", "LATITUD", "LONGITUD"])

y = df["DOMINIO"]
X = df[["ALTITUD", "LATITUD", "LONGITUD", "AÑO", "MES"]]

st.subheader("Variables utilizadas")
st.write("Predictoras:", list(X.columns))
st.write("Objetivo: `DOMINIO`")
st.write("Clases:", y.unique())

# =============================
# 3. División train/test
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# =============================
# 4. Regresión Logística
# =============================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)
acc_log = accuracy_score(y_test, y_pred_log)
results["Regresión Logística"] = acc_log

st.subheader("📈 Regresión Logística")
st.write("Accuracy:", acc_log)
st.text("Reporte de Clasificación:\n" + classification_report(y_test, y_pred_log))

# Matriz de confusión
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Matriz de Confusión - Regresión Logística")
st.pyplot(fig)

# Curva ROC
y_bin = label_binarize(y_test, classes=np.unique(y))
y_score = log_model.decision_function(X_test_scaled)
fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score.ravel())
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {roc_auc:.2f})")
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Curva ROC - Regresión Logística")
ax.legend(loc="lower right")
st.pyplot(fig)

# =============================
# 5. Árbol de Decisión
# =============================
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)
results["Árbol de Decisión"] = acc_tree

st.subheader("🌳 Árbol de Decisión")
st.write("Accuracy:", acc_tree)
st.text("Reporte de Clasificación:\n" + classification_report(y_test, y_pred_tree))

# Matriz de confusión
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_tree), annot=True, fmt="d", cmap="Greens", ax=ax)
ax.set_title("Matriz de Confusión - Árbol de Decisión")
st.pyplot(fig)

# Importancia de variables
importances = tree_model.feature_importances_
fig, ax = plt.subplots()
sns.barplot(x=X.columns, y=importances, palette="Greens", ax=ax)
ax.set_title("Importancia de Variables - Árbol de Decisión")
plt.xticks(rotation=45)
st.pyplot(fig)

# Visualización del árbol completo
fig, ax = plt.subplots(figsize=(20,10))
plot_tree(tree_model,
          feature_names=X.columns,
          class_names=[str(c) for c in y.unique()],
          filled=True, rounded=True, fontsize=8, ax=ax)
st.pyplot(fig)

# =============================
# 6. SVM
# =============================
svm_model = SVC(probability=True)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, y_pred_svm)
results["SVM"] = acc_svm

st.subheader("⚡ SVM")
st.write("Accuracy:", acc_svm)
st.text("Reporte de Clasificación:\n" + classification_report(y_test, y_pred_svm))

# Matriz de confusión
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt="d", cmap="Oranges", ax=ax)
ax.set_title("Matriz de Confusión - SVM")
st.pyplot(fig)

# Curva ROC
y_score_svm = svm_model.decision_function(X_test_scaled)
fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score_svm.ravel())
roc_auc_svm = auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, color="darkorange", label=f"ROC curve (area = {roc_auc_svm:.2f})")
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Curva ROC - SVM")
ax.legend(loc="lower right")
st.pyplot(fig)

# =============================
# 7. Comparación Final
# =============================
st.subheader("📊 Comparación de Accuracy entre Modelos")

fig, ax = plt.subplots(figsize=(7,5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="Blues_d", ax=ax)
ax.set_ylim(0,1)
ax.set_ylabel("Accuracy")
ax.set_title("Comparación de Modelos")
st.pyplot(fig)

st.success("✅ Proceso completado. Revisa resultados y gráficos arriba.")
