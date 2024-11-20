

import streamlit as st
import numpy as np
import math
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

# Título de la app
st.title("Predicción de Goles")
st.write("""
    Esta aplicación predice el número de goles de un jugador en base a sus características.
""")

# Cargar los datos de prueba (si los tienes)
df_para_pruebas = pd.read_csv("../data/test/df_pruebas_3_features.csv", index_col=0)
df_para_pruebas = pd.read_csv("../data/test/df_pruebas_4_features.csv", index_col=0)
df_para_pruebas = pd.read_csv("../data/test/df_pruebas_5_features.csv", index_col=0)
st.dataframe(df_para_pruebas, hide_index=True)


# Cargar los modelos y sus escaladores
modelos = {}
# for modelo_name in ['4_random_forest', '5_xgboost', '2_regresion_poly_3',  '6_lightgbm', '7_red_neuronal']:
# for modelo_name in ['1_regresion_lineal', '2_regresion_poly_2', '2_regresion_poly_3', '3_decision_tree', '4_random_forest', '5_xgboost', '6_lightgbm', '7_red_neuronal']:
for modelo_name in ['7_red_neuronal','2_regresion_poly_3']:
    with open(f'../models/mae/5features/{modelo_name}.pkl', 'rb') as file:
        modelos[modelo_name] = pickle.load(file)



# Entrada de datos: Se ingresa como texto
st.write("Por favor ingresa los valores de las características del jugador.")
input_text = st.text_area("Ingresa los valores separados por espacios")

# Procesamiento del texto ingresado
if input_text:
    try:
        # Convertir el texto en una lista de números flotantes
        input_values = list(map(float, input_text.split()))

        # Comprobar si tiene el número correcto de elementos
        if len(input_values) == 5:
            input_data = np.array(input_values).reshape(1, -1)  # Crear el array con los valores
            st.write("Datos procesados:", input_data)
        else:
            st.error("Por favor, ingrese exactamente 5 valores.")

    except ValueError:
        st.error("Hubo un error al procesar los datos. Asegúrese de ingresar solo números separados por espacios.")


# Botón para hacer las predicciones
if st.button("Predecir"):
    predicciones = {}

    # Iterar sobre cada modelo y usar su propio escalador si está presente
    for nombre_modelo, modelo_data in modelos.items():
        modelo = modelo_data['model']  # El modelo
        scaler = modelo_data.get('scaler', None)  # El escalador, si existe

        # Escalar los datos de entrada si el modelo tiene un escalador
        if scaler:
            input_data_scaled = scaler.transform(input_data)

        else:
            input_data_scaled = input_data  # Usar los datos originales si no hay escalador

        # Hacer la predicción
        pred = modelo.predict(input_data_scaled)

        # Limitar la predicción a un rango razonable (por ejemplo, 0 a 100)
        pred = max(0, min(pred[0], 100))

        if pred - int(pred) >= 0.5:
            pred = int(pred) + 1  # Redondear hacia arriba
        else:
            pred = int(pred)  # Redondear hacia abajo
        
        # Crear un rango de 2 unidades por arriba y por abajo de la predicción
        rango_inferior = pred - 1
        rango_superior = pred + 1

        # Imprimir el rango
        st.success(f"La predicción está en este rango: {rango_inferior} - {rango_superior}")

        
        
        
        
        
        
        
        # Redondear la predicción a un valor entero
        
    #     predicciones[nombre_modelo] = resultado_redon

        
       
    #     # predicciones[nombre_modelo] = pred

    # # Mostrar las predicciones
    # for model_name, pred in predicciones.items():
    #     st.subheader(f"Resultado de {model_name}:")
    #     st.write(f"Predicción de goles ({model_name}): {pred}")

    # # Promedio de las predicciones
    # pred_combinada = sum(predicciones.values()) / len(predicciones)
    # pred_combinada_redondeada = math.floor(pred_combinada)

    # st.subheader("Predicción Promediada:")
    # st.write(f"Predicción combinada (Promediada): {pred_combinada_redondeada}")

