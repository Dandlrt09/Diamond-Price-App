import streamlit as st
import pandas as pd
import joblib

# Configuración de la página
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide"
)

# Cargar modelo (si no existe, lo entrena automáticamente)
@st.cache_resource
def load_model():
    import os
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    
    model_path = 'diamond_model.pkl'
    
    # Si el modelo no existe, entrenarlo
    if not os.path.exists(model_path):
        with st.spinner('Entrenando modelo por primera vez... Esto puede tardar un minuto.'):
            data = pd.read_csv('diamond_data.csv')
            X = data.drop('Price', axis=1)
            y = data['Price']
            
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object']).columns
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', 'passthrough', numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ]
            )
            
            modelo = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', modelo)])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipe.fit(X_train, y_train)
            joblib.dump(pipe, model_path)
    
    return joblib.load(model_path)

model = load_model()

# Título principal
st.title("💎 Predicción de Precios de Diamantes")
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Datos", "📈 Evaluación", "🎮 Simulador", "ℹ️ Sobre el proyecto"])

# Tab 1: Datos
with tab1:
    st.header("📊 Exploración del Dataset")
    st.write("Acá vas a conocer los datos que el modelo usó para aprender a predecir precios.")
    
    data = pd.read_csv('diamond_data.csv')
    
    # Explicación del dataset
    st.info("""
    **¿Qué es este dataset?**
    Contiene información real de **6,000 diamantes**. Cada fila representa un diamante 
    con sus características físicas y su precio real de venta. El modelo aprendió 
    patrones entre estas características y el precio.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vista de los datos")
        st.caption("Mostrando las primeras 10 filas del dataset:")
        st.dataframe(data.head(10), width='stretch')
        st.success(f"**Total de diamantes:** {data.shape[0]:,} registros")
        st.info(f"**Variables disponibles:** {data.shape[1]} columnas")
    
    with col2:
        st.subheader("Estadísticas clave")
        st.caption("Resumen numérico de las variables:")
        st.dataframe(data.describe(), width='stretch')
        st.caption("""
        💡 **¿Qué significa esto?**
        - El **25%** de los diamantes pesan menos de 0.61 quilates
        - El **50%** (mediana) pesan 0.82 quilates
        - El **75%** pesan más de 1.05 quilates
        - Los precios van desde USD $326 hasta $18,842
        """)
    
    st.markdown("---")
    
    st.subheader("📋 ¿Qué variables influyen en el precio?")
    st.write("El precio de un diamante se determina por las 4C's (Cut, Color, Clarity, Carat) + características adicionales:")
    
    variables_info = pd.DataFrame({
        'Variable': ['Carat Weight', 'Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report', 'Price'],
        '¿Qué es?': [
            'Peso de la piedra. 1 quilate = 0.2 gramos',
            'Calidad del corte geométrico (Ideal es el mejor)',
            'Color desde D (incoloro) hasta J (liger amarillo)',
            'Pureza: presencia de inclusiones (IF = perfecto, I1 = visible)',
            'Acabado superficial de la piedra',
            'Precisión de las faccetas alineadas',
            'Laboratorio que certifica las características',
            'Precio final de venta en dólares USD'
        ],
        'Importancia': [
            '⭐⭐⭐⭐⭐ Crítica (el peso define la base del precio)',
            '⭐⭐⭐⭐ Afecta cuánto brilla el diamante',
            '⭐⭐⭐ Afecta el valor visual',
            '⭐⭐⭐ Determina pureza interna',
            '⭐⭐ Mejora la apariencia',
            '⭐⭐ Mejora la apariencia',
            '⭐ Da confianza al comprador',
            '🎯 Variable a predecir'
        ]
    })
    
    st.dataframe(variables_info, width='stretch', hide_index=True)

# Tab 2: Evaluación
with tab2:
    st.header("📈 ¿Qué tan bueno es el modelo?")
    st.write("Acá evaluamos si el modelo realmente sabe predecir precios o solo está adivinando.")
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    data = pd.read_csv('diamond_data.csv')
    X = data.drop('Price', axis=1)
    y = data['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_pred = model.predict(X_test)
    
    # Explicación de las métricas
    st.subheader("🎯 Métricas de rendimiento")
    st.info("""
    **¿Cómo evaluamos al modelo?**
    - El modelo NO vio estos 1,200 diamantes durante el entrenamiento (son datos de prueba)
    - El modelo intenta predecir el precio y comparamos con el precio real
    - Así medimos qué tan certero es
    """)
    
    col1, col2, col3 = st.columns(3)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
        st.caption("**¿Qué es?** El modelo explica el 98.6% de las variaciones de precio. 1.0 sería perfecto.")
    
    with col2:
        st.metric("Error Promedio (MAE)", f"${mae:.2f}")
        st.caption("**¿Qué es?** En promedio, el modelo se equivoca por ~$652 USD por diamante.")
    
    with col3:
        st.metric("Error Cuadrático (RMSE)", f"${rmse:.2f}")
        st.caption("**¿Qué es?** Castiga errores grandes. Simil al MAE pero más sensible.")
    
    # Analogía para entender
    st.success("""
    💡 **Para ponerlo en contexto:**
    - Si comprás un diamante de $5,000, el modelo te diría un precio entre $4,350 y $5,650 (error típico)
    - Si comprás uno de $15,000, el rango sería entre $14,350 y $15,650
    - ¡Eso es MUY preciso para un modelo automático!
    """)
    
    st.markdown("---")
    
    # Gráficos
    st.subheader("📊 Visualización del rendimiento")
    
    import matplotlib.pyplot as plt
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Gráfico 1: Distribución de errores**")
        st.caption("Muestra cuánto se equivocó el modelo en cada predicción")
        
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        errors = y_test.values - y_pred
        ax1.hist(errors, bins=50, edgecolor='black', color='skyblue', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error cero')
        ax1.set_xlabel('Error (USD)')
        ax1.set_ylabel('Cantidad de diamantes')
        ax1.set_title('¿Cuánto se equivocó el modelo?')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
        
        st.caption("""
        ✅ **Interpretación:** La mayoría de los errores están cerca de $0 (línea roja).
        Si el gráfico fuera plano y ancho, el modelo sería malo.
        """)
    
    with col2:
        st.write("**Gráfico 2: Real vs Predicho**")
        st.caption("Comparación directa: ¿Coincide lo predicho con lo real?")
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(y_test, y_pred, alpha=0.5, s=10, color='dodgerblue')
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Predicción perfecta')
        ax2.set_xlabel('Precio Real (USD)')
        ax2.set_ylabel('Precio Predicho (USD)')
        ax2.set_title('¿Acierta el modelo?')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        
        st.caption("""
        ✅ **Interpretación:** Los puntos cerca de la línea roja significan 
        predicción perfecta. ¡La mayoría están pegados a la línea!
        """)

# Tab 3: Simulador
with tab3:
    st.header("Simulador de Precios")
    st.write("Ajustá los parámetros del diamante para predecir su precio:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Características físicas")
        carat_weight = st.slider("Peso en quilates", 0.2, 5.0, 1.0, 0.1)
    
    with col2:
        st.subheader("Calidad y certificación")
        cut = st.selectbox("Corte", ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair'])
        color = st.selectbox("Color", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
        clarity = st.selectbox("Claridad", ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'])
        polish = st.selectbox("Pulido", ['EX', 'VG', 'G', 'ID', 'Fair'])
        symmetry = st.selectbox("Simetría", ['EX', 'VG', 'G', 'ID', 'Fair'])
        report = st.selectbox("Certificación", ['GIA', 'AGSL', 'IGI', 'EGL'])
    
    if st.button("💎 Predecir Precio", type="primary", width='stretch'):
        input_data = pd.DataFrame({
            'Carat Weight': [carat_weight],
            'Cut': [cut],
            'Color': [color],
            'Clarity': [clarity],
            'Polish': [polish],
            'Symmetry': [symmetry],
            'Report': [report]
        })
        
        prediction = model.predict(input_data)[0]
        
        st.success(f"**Precio estimado: ${prediction:,.2f} USD**")
        
        st.info("💡 *Este es un valor referencial basado en el modelo de Machine Learning entrenado.*")

# Tab 4: Sobre el proyecto
with tab4:
    st.header("Sobre este proyecto")
    
    st.markdown("""
    ### 🎯 Objetivo
    Este proyecto implementa un pipeline completo de **Machine Learning** utilizando **PyCaret** 
    para predecir el precio de diamantes basándose en sus características físicas y de calidad.
    
    ### 🛠️ Herramientas utilizadas
    - **Python** y **Scikit-learn** para el modelado
    - **PyCaret** para comparación rápida de algoritmos
    - **Extra Trees Regressor** como modelo final (mejor performance en comparación)
    - **Streamlit** para el despliegue interactivo
    
    ### 📊 Dataset
    El dataset contiene **6,000 registros** de diamantes con las siguientes variables:
    - **Carat Weight**: Peso en quilates (numérico)
    - **Cut**: Calidad del corte (categórico)
    - **Color**: Color del diamante (categórico, de D a J)
    - **Clarity**: Claridad de la piedra (categórico)
    - **Polish**: Calidad del pulido (categórico)
    - **Symmetry**: Simetría (categórico)
    - **Report**: Entidad certificadora (categórico)
    
    ### 🏆 Rendimiento del modelo
    - **R² Score**: 0.9857 (el modelo explica el 98.6% de la varianza)
    - **Mean Absolute Error**: ~$652 USD
    - **Algoritmo**: Extra Trees Regressor con 100 estimadores
    
    ### 📁 Repositorio
    Podés ver el notebook original y el código fuente en el repositorio de [DataScience_Proyects](https://github.com/Dandlrt09/DataScience_Proyects).
    """)

# Footer
st.markdown("---")
st.caption("💎 Diamond Price Predictor | Desarrollado con Streamlit y Scikit-learn")
