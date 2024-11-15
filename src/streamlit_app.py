import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# # Cargar el mejor modelo XGBoost guardado
# with open('../models/ModeloSimpleFifa_best_linear_model.pkl', 'rb') as file:
#     modelo = pickle.load(file)

# Puedes agregar más casos de uso o métricas si lo deseas.

model_choice = st.selectbox("Selecciona el modelo", ['XGBoost', 'Regresión Lineal'])

if model_choice == 'Regresión Lineal':
    with open('../models/ModeloSimpleFifa_best_linear_model.pkl', 'rb') as file:
        modelo = pickle.load(file)
elif model_choice == 'XGBoost':
    with open('../models/ModeloSimpleFifa_best_xgb_model.pkl', 'rb') as file:
        modelo = pickle.load(file)

# Título de la app
st.title("Predicción de Goles con XGBoost")

# Descripción
st.write("""
    Esta aplicación predice el número de goles de un jugador en base a sus características.
    """)

# Entrada de datos del usuario (ejemplo de características de un jugador)
overall = st.number_input("overall", min_value=35, max_value=99, value=60)
shooting = st.number_input("shooting", min_value=35, max_value=99, value=60)
Rating_x = st.slider("Rating_x", min_value=0.0, max_value=10.0, value=5.0)
value_eur = st.slider("value_eur", min_value=200000, max_value=3000000, value=1000000)
# Goals_x = st.number_input("Edad", min_value=18, max_value=40, value=25)

# Procesar las entradas
input_data = np.array([overall,value_eur,shooting,Rating_x]).reshape(1, -1)

# Escalar las entradas (esto debe coincidir con el preprocesamiento usado en el entrenamiento del modelo)
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Hacer la predicción con el modelo cargado
if st.button("Predecir"):
    prediction = modelo.predict(input_data_scaled)
    st.write(f"La predicción de goles es: {prediction[0]}")

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Mostrar separador
st.write('-------------------------------')

# Cargar modelo de regresión lineal
model_path_lr = '../models/Fifa-StatsReales_best_linear_model.pkl'
with open(model_path_lr, 'rb') as file:
    model_lr = pickle.load(file)

X_test_lr = pd.read_csv("../data/X_test_fifa_stats_LinearR.csv", index_col=0)
Y_test_lr = pd.read_csv("../data/Y_test_fifa_stats_LinearR.csv", index_col=0)

# Cambiar los índices de los DataFrames para que empiecen desde 1
X_test_lr.index = range(1, len(X_test_lr) + 1)
Y_test_lr.index = range(1, len(Y_test_lr) + 1)

# Mostrar los DataFrames para inspección
st.write('# Modelo regresión lineal')
st.dataframe(X_test_lr)
st.dataframe(Y_test_lr)

# Definir los casos de uso
casos_uso_lr = {
    'Caso de uso 1': X_test_lr.iloc[0],
    'Caso de uso 2': X_test_lr.iloc[1],
    'Caso de uso 3': X_test_lr.iloc[2],
    'Caso de uso 4': X_test_lr.iloc[3],
    'Caso de uso 5': X_test_lr.iloc[4]
}

# Interfaz de usuario para regresión lineal
st.title('Predicción para casos de uso - Regresión Lineal')

# Selección de caso de uso con clave única
seleccion_lr = st.selectbox("Elige un caso de uso:", list(casos_uso_lr.keys()), key="selectbox_lr")

# Escalar los datos del caso de uso seleccionado
X_input_lr = np.array(casos_uso_lr[seleccion_lr]).reshape(1, -1)

# Hacer la predicción
prediccion_lr = model_lr.predict(X_input_lr)

# Convertir la predicción a un número entero positivo
prediccion_entera_lr = max(0, int(round(float(prediccion_lr[0]))))

# Mostrar la predicción
st.success(f"Predicción para {seleccion_lr}: {prediccion_entera_lr} goles")


# Separador para la sección de Red Neuronal
st.write('-------------------------------')

# Cargar modelo de red neuronal
model_path_rn = '../models/Fifa-StatsReales_best_RN_model.pkl'
with open(model_path_rn, 'rb') as file:
    model_rn = pickle.load(file)

x_test_rn = pd.read_csv("../data/X_test_fifa_stats.csv", index_col=0)
y_test_rn = pd.read_csv("../data/y_test_fifa_stats.csv", index_col=0)

# Cambiar los índices de los DataFrames para que empiecen desde 1
x_test_rn.index = range(1, len(x_test_rn) + 1)
y_test_rn.index = range(1, len(y_test_rn) + 1)

# Mostrar los DataFrames para inspección
st.write('# Modelo Red Neuronal')
st.dataframe(x_test_rn)
st.dataframe(y_test_rn)

# Definir los casos de uso
casos_uso_rn = {
    'Caso de uso 1': x_test_rn.iloc[0],
    'Caso de uso 2': x_test_rn.iloc[1],
    'Caso de uso 3': x_test_rn.iloc[2],
    'Caso de uso 4': x_test_rn.iloc[3],
    'Caso de uso 5': x_test_rn.iloc[4]
}

# Escalar los datos de entrada usando StandardScaler
scaler = StandardScaler()
X_test_scaled_rn = scaler.fit_transform(x_test_rn)  # Escalar todo X_test

# Interfaz de usuario para red neuronal
st.title('Predicción para casos de uso - Red Neuronal')

# Selección de caso de uso con clave única
seleccion_rn = st.selectbox("Elige un caso de uso:", list(casos_uso_rn.keys()), key="selectbox_rn")

# Escalar los datos del caso de uso seleccionado
X_input_rn = np.array(casos_uso_rn[seleccion_rn]).reshape(1, -1)
X_input_scaled_rn = scaler.transform(X_input_rn)  # Aplicamos el escalador

# Hacer la predicción
prediccion_rn = model_rn.predict(X_input_scaled_rn)

# Convertir la predicción a un número entero positivo
prediccion_entera_rn = max(0, int(round(float(prediccion_rn[0]))))

# Mostrar la predicción
st.success(f"Predicción para {seleccion_rn}: {prediccion_entera_rn} goles")
