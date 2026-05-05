# 💎 Diamond Price Predictor

Una aplicación interactiva de **Machine Learning** que predice el precio de diamantes basándose en sus características físicas y de calidad.

![Streamlit App](https://img.shields.io/badge/Streamlit-1.45+-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6+-F7931E?logo=scikit-learn)

---

## 🎯 Descripción

Este proyecto implementa un pipeline completo de Machine Learning usando **PyCaret** y **Scikit-learn** para predecir precios de diamantes. La aplicación, desarrollada en **Streamlit**, permite explorar el dataset, evaluar el modelo y simular predicciones en tiempo real.

### ✨ Características

- 📊 **Exploración de Datos**: Vista interactiva del dataset con estadísticas descriptivas
- 📈 **Evaluación del Modelo**: Métricas de rendimiento (R², MAE, RMSE) y visualizaciones
- 🎮 **Simulador**: Predicción de precios ajustando las características del diamante
- ℹ️ **Sobre el Proyecto**: Contexto, herramientas utilizadas y enlaces al repositorio original

---

## 🛠️ Tecnologías

- **Python 3.10+**
- **Streamlit** - Interfaz web interactiva
- **Scikit-learn** - Modelado y evaluación
- **PyCaret** - Comparación rápida de algoritmos
- **Pandas** - Manipulación de datos
- **Matplotlib** - Visualizaciones

---

## 📁 Dataset

El dataset contiene **6,000 registros** de diamantes con las siguientes variables:

| Variable | Descripción | Tipo |
|----------|-------------|------|
| `Carat Weight` | Peso en quilates (1 ct = 0.2g) | Numérico |
| `Cut` | Calidad del corte (Ideal, Premium, Good, etc.) | Categórico |
| `Color` | Color (D=incoloro → J=amarillo) | Categórico |
| `Clarity` | Claridad (IF=perfecto → I1=visible) | Categórico |
| `Polish` | Calidad del pulido | Categórico |
| `Symmetry` | Simetría de las facetas | Categórico |
| `Report` | Certificación (GIA, AGSL, IGI, EGL) | Categórico |
| `Price` | Precio en USD (variable objetivo) | Numérico |

---

## 🏆 Rendimiento del Modelo

El modelo seleccionado es **Extra Trees Regressor** (mejor performance en comparación de PyCaret):

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **R² Score** | 0.9857 | El modelo explica el 98.6% de la varianza del precio |
| **MAE** | ~$652 USD | Error promedio por predicción |
| **RMSE** | ~$863 USD | Error sensible a desviaciones grandes |

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/Dandlrt09/Diamond-Price-App.git
cd Diamond-Price-App
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```
*(Si no tenés el archivo, instala manualmente: `pip install streamlit pandas scikit-learn pycaret matplotlib`)*

### 3. Entrenar el modelo
```bash
python train_model.py
```
Esto generará el archivo `diamond_model.pkl` con el modelo entrenado.

### 4. Ejecutar la app
```bash
streamlit run app.py
```

La app se abrirá en tu navegador en `http://localhost:8501`.

---

## 📊 Estructura del Proyecto

```
Diamond-Price-App/
├── app.py              # Aplicación principal de Streamlit
├── train_model.py      # Script para entrenar y guardar el modelo
├── diamond_model.pkl   # Modelo entrenado (no se sube al repo)
├── requirements.txt    # Dependencias (opcional)
├── .gitignore         # Archivos excluidos
└── README.md          # Este archivo
```

---

## 🔗 Enlaces Relacionados

- **Repositorio original del proyecto**: [DataScience_Proyects - Proyecto 3](https://github.com/Dandlrt09/DataScience_Proyects/tree/main/Proyecto%203)
- **Notebook de entrenamiento**: [Pycaret y Entrenamiento.ipynb](https://github.com/Dandlrt09/DataScience_Proyects/blob/main/Proyecto%203/Pycaret%20y%20Entrenamiento.ipynb)
- **Portafolio personal**: [danieldlrt09.github.io/Portafolio_Personal](https://danieldlrt09.github.io/Portafolio_Personal/)

---

## 📝 Notas

- El archivo `diamond_model.pkl` no se incluye en el repositorio por su tamaño (~58MB). Podés regenerarlo ejecutando `train_model.py`.
- La app usa el dataset `diamond` de PyCaret, que se descarga automáticamente al ejecutar.

---

## 📄 Licencia

Este proyecto es de uso educativo y hace parte del portafolio de Data Science.

**Desarrollado por** [Daniel Del Río](https://github.com/Dandlrt09) 💎
