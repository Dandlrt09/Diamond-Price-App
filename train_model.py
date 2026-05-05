import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from pycaret.datasets import get_data

# Cargar datos
data = get_data('diamond')

# Features y target
X = data.drop('Price', axis=1)
y = data['Price']

# Identificar variables
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Modelo (mismo que el notebook: ExtraTreesRegressor)
modelo = ExtraTreesRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Pipeline
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', modelo)
])

# Split y entrenar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)

# Guardar modelo
joblib.dump(pipe, 'diamond_model.pkl')
print("Modelo guardado como diamond_model.pkl")
