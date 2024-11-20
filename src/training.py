import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures
from tensorflow import keras
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import re
import os
from sklearn.linear_model import Ridge, Lasso, ElasticNet


def train_and_evaluate_linar_model(ruta_guardar, X_train, X_test, y_train, y_test):
    # Crear una lista vacía para almacenar los resultados
    model_results = []

    match = re.search(r'pca(\d+)', ruta_guardar)

    if match:
        num_pca = int(match.group(1)) # Extraer el número capturado
        print(f"El número después de 'pca' es: {num_pca}")
    else:
        num_pca = len(X_train)
    
    # Crear el pipeline con los pasos de preprocesamiento y modelo
    pipe = Pipeline(steps=[("scaler", StandardScaler()),
                           ("pca", PCA()),
                           ('classifier', LinearRegression())
    ])
    
    # Definir los parámetros de búsqueda para el GridSearch
    linear_params = {
        'scaler': [StandardScaler(), MinMaxScaler(), None],
        # 'pca__n_components': [num_pca],
        # 'pca__n_components': [None,10, 0.95],
        'classifier': [LinearRegression()]
    }
    
    # Definir el espacio de búsqueda
    search_space = [
        linear_params
    ]
    
    # Configurar GridSearchCV
    gs = GridSearchCV(estimator=pipe,
                      param_grid=search_space,
                      cv=10,
                      scoring='neg_mean_absolute_error',
                      verbose=2,
                      n_jobs=-1)
    
    # Entrenar el modelo con GridSearchCV
    gs.fit(X_train, y_train)
    
    # Guardar el mejor modelo en un archivo .pkl
    best_model = gs.best_estimator_
    gs.best_estimator_
    gs
    best_scaler = gs.best_estimator_.named_steps['scaler']


    # with open(ruta_guardar, 'wb') as file:
    #     pickle.dump(best_model, file)
    with open(ruta_guardar, 'wb') as file:
        pickle.dump({
            'model': best_model,
            'scaler': best_scaler
        }, file)
    
    # Evaluar el modelo
    Y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, Y_pred)
    mae = mean_absolute_error(y_test, Y_pred)
    r2 = r2_score(y_test, Y_pred)
    
    # # Imprimir los resultados
    # print(f"Mean Squared Error: {mse}")
    # print(f"Mean Absolute Error: {mae}")
    # print(f"R-squared: {r2}")
    
    # Almacenar los resultados en la lista
    model_results.append({
        'Model': 'Linear Regression',
        'Best_model':gs.best_estimator_,
        'Best_params':gs.best_params_,
        'Best_score':gs.best_score_,
        'MSE': mse,
        'MAE': mae,
        'R-squared': r2
    })
    
    # Convertir los resultados a un DataFrame
    results_df = pd.DataFrame(model_results)
    
    # # Mostrar los resultados
    # print(results_df)
    
    return best_model, results_df

# --------------------------------------------

def train_and_evaluate_polynomial_model(ruta_guardar, X_train, X_test, y_train, y_test,results_df):
    match = re.search(r'pca(\d+)', ruta_guardar)

    if match:
        num_pca = int(match.group(1))  # Extraer el número capturado
        print(f"El número después de 'pca' es: {num_pca}")
    else:
        num_pca = len(X_train)
    
    match = re.search(r'\d+_regresion(_pca\d+)?_poly_(\d+)\.pkl$', ruta_guardar)

    # Extraer el número encontrado
    if match:
        grado = int(match.groups()[-1])
        # print(grado)
    else:
        print("No se encontró el número")
    


    # Crear una lista vacía para almacenar los resultados
    model_results = []
    
    # Crear el pipeline con los pasos de preprocesamiento y modelo
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("pca", PCA()),
        ("classifier", LinearRegression())
    ])
    
    
    # Definir los parámetros de búsqueda para el GridSearch
    polynomial_params = {
        'scaler': [StandardScaler(), None],
        # 'scaler': [StandardScaler(), MinMaxScaler(), None],
        'polynomial__degree': [grado],  # Se puede añadir más grados si se quiere probar
        # 'polynomial__interaction_only': [True, False],  # Solo para ElasticNet
        # 'pca__n_components': [num_pca],
        # 'pca__n_components': [10, 0.95],
        # 'classifier': [Ridge(), Lasso(), ElasticNet()],
        'classifier': [ElasticNet()],
        'classifier__alpha': np.arange(0.05, 0.15, 0.01).tolist(), # Valores entre 0.05 y 0.15, con un paso de 0.01
        'classifier__l1_ratio': np.arange(0.05, 0.15, 0.01).tolist()
    }
    
    # Definir el espacio de búsqueda
    search_space = [
        polynomial_params
    ]
    
    # Configurar GridSearchCV
    gs = GridSearchCV(estimator=pipe,
                      param_grid=search_space,
                      cv=5,
                      scoring='neg_mean_absolute_error',
                      verbose=2,
                      n_jobs=-1)
    
    
    # Entrenar el modelo con GridSearchCV
    gs.fit(X_train, y_train)
    
    # # Guardar el mejor modelo en un archivo .pkl
    # best_model = gs.best_estimator_
    # with open(ruta_guardar, 'wb') as file:
    #     pickle.dump(best_model, file)
    # Guardar el mejor modelo en un archivo .pkl
    best_model = gs.best_estimator_
    gs.best_estimator_
    best_scaler = gs.best_estimator_.named_steps['scaler']
    # with open(ruta_guardar, 'wb') as file:
    #     pickle.dump(best_model, file)
    with open(ruta_guardar, 'wb') as file:
        pickle.dump({
            'model': best_model,
            'scaler': best_scaler
        }, file)
    

    # Evaluar el modelo
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # # Imprimir los resultados
    # print(f"Mean Squared Error: {mse}")
    # print(f"Mean Absolute Error: {mae}")
    # print(f"R-squared: {r2}")
    
    model_results.append({
        'Model': f'Polynomial Regression_{grado}',
        'Best_model':gs.best_estimator_,
        'Best_params':gs.best_params_,
        'Best_score':gs.best_score_,
        'MSE': mse,
        'MAE': mae,
        'R-squared': r2
    })
    
    # Convertir los resultados en DataFrame y concatenar con el anterior
    new_results_df = pd.DataFrame(model_results)
    results_df = pd.concat([results_df, new_results_df], ignore_index=True)
    
    # # Mostrar los resultados
    # print(results_df)
    
    return best_model, results_df

# --------------------------------------------

def train_and_evaluate_decision_tree_model(ruta_guardar, X_train, X_test, y_train, y_test,results_df):
    # Crear una lista vacía para almacenar los resultados
    model_results = []
    ruta_con_png = os.path.splitext(ruta_guardar)[0] + '.png'
    
    # Crear el pipeline con los pasos de preprocesamiento y modelo
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),  # Escalado opcional
        ("classifier", DecisionTreeRegressor())  # Árbol de Decisión
    ])
    
    # Definir los parámetros de búsqueda para el GridSearch
    tree_params = {
        'scaler': [StandardScaler(), None],
        # 'scaler': [StandardScaler(), MinMaxScaler(), None],
        'classifier__max_depth': [None, 5, 10, 15],  # Profundidad máxima del árbol
        'classifier__min_samples_split': [5, 10],  # Número mínimo de muestras para dividir
        'classifier__min_samples_leaf': [2, 5]  # Número mínimo de muestras en una hoja
    }
    
    # Definir el espacio de búsqueda
    search_space = [
        tree_params
    ]
    
    # Configurar GridSearchCV
    gs = GridSearchCV(estimator=pipe,
                      param_grid=search_space,
                      cv=10,
                      scoring='neg_mean_absolute_error',
                      verbose=2,
                      n_jobs=-1)
    
    # Entrenar el modelo con GridSearchCV
    gs.fit(X_train, y_train)

    # Obtener el modelo entrenado con los mejores parámetros
    best_model = gs.best_estimator_
    
    # Obtener la importancia de las características del modelo
    importances = best_model.named_steps['classifier'].feature_importances_

    # Crear un DataFrame para ordenar y visualizar las importancias
    features = X_train.columns  # Si X_train es un DataFrame de pandas
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })

    # Ordenar el DataFrame por la importancia de las características
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Mostrar las importancias
    # print(importance_df)

    # Graficar las importancias
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importances')

    # Guardar la imagen sin mostrarla
    plt.savefig(ruta_con_png)
    plt.close()  # Cierra la figura para que no se muestre
    
    # # Guardar el mejor modelo en un archivo .pkl
    # best_model = gs.best_estimator_
    # with open(ruta_guardar, 'wb') as file:
    #     pickle.dump(best_model, file)
    best_model = gs.best_estimator_
    gs.best_estimator_
    best_scaler = gs.best_estimator_.named_steps['scaler']
    # with open(ruta_guardar, 'wb') as file:
    #     pickle.dump(best_model, file)
    with open(ruta_guardar, 'wb') as file:
        pickle.dump({
            'model': best_model,
            'scaler': best_scaler
        }, file)
    
    # Evaluar el modelo
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # # Imprimir los resultados
    # print(f"Mean Squared Error: {mse}")
    # print(f"Mean Absolute Error: {mae}")
    # print(f"R-squared: {r2}")
    
    model_results.append({
        'Model': 'Decision Tree',
        'Best_model':gs.best_estimator_,
        'Best_params':gs.best_params_,
        'Best_score':gs.best_score_,
        'MSE': mse,
        'MAE': mae,
        'R-squared': r2
    })
    
    # Convertir los resultados en DataFrame y concatenar con el anterior
    new_results_df = pd.DataFrame(model_results)
    results_df = pd.concat([results_df, new_results_df], ignore_index=True)
    
    # Mostrar los resultados
    # print(results_df)
    
    return best_model, results_df

# --------------------------------------------

def train_and_evaluate_random_forest_model(ruta_guardar, X_train, X_test, y_train, y_test, results_df):
    # Crear una lista vacía para almacenar los resultados
    model_results = []

    match = re.search(r'pca(\d+)', ruta_guardar)

    if match:
        num_pca = int(match.group(1))  # Extraer el número capturado
        print(f"El número después de 'pca' es: {num_pca}")
    else:
        num_pca = len(X_train)

    ruta_con_png = os.path.splitext(ruta_guardar)[0] + '.png'

    # Crear el pipeline con los pasos de preprocesamiento y modelo
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),  # Escalado opcional
        ("pca", PCA()),  # Reducción de dimensionalidad opcional
        ("classifier", RandomForestRegressor(random_state=42))  # Random Forest
    ])

    # Definir los parámetros de búsqueda para el GridSearch
    forest_params = {
        'scaler': [StandardScaler(), None],
        # "pca__n_components": [num_pca],  # Dimensionalidad reducida
        # "pca__n_components": [5, 10, 0.95],  # Dimensionalidad reducida
        'classifier__n_estimators': [100],  # Número de árboles en el bosque
        'classifier__max_depth': [None, 5, 10],  # Profundidad máxima de cada árbol
        'classifier__min_samples_split': [5, 10],  # Número mínimo de muestras para dividir un nodo
        'classifier__min_samples_leaf': [2, 5]  # Número mínimo de muestras en una hoja
    }

    # Configurar GridSearchCV
    gs = GridSearchCV(estimator=pipe,
                      param_grid=forest_params,
                      cv=10,
                      scoring='neg_mean_absolute_error',
                      verbose=2,
                      n_jobs=-1)

    # Entrenar el modelo con GridSearchCV
    gs.fit(X_train, y_train)

    # Obtener el modelo entrenado con los mejores parámetros
    best_model = gs.best_estimator_

    # Manejo de características dependiendo del uso de PCA
    if 'pca' in best_model.named_steps and best_model.named_steps['pca'] is not None:
        n_components = best_model.named_steps['pca'].n_components_
        features = [f'PC{i + 1}' for i in range(n_components)]  # Nombres de componentes principales
    else:
        features = X_train.columns  # Usar las columnas originales

    # Obtener la importancia de las características del modelo
    importances = best_model.named_steps['classifier'].feature_importances_

    # Crear un DataFrame para ordenar y visualizar las importancias
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })

    # Ordenar el DataFrame por la importancia de las características
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Mostrar las importancias
    # print(importance_df)

    # Graficar las importancias
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importances')

    # Guardar la imagen sin mostrarla
    plt.savefig(ruta_con_png)
    plt.close()  # Cierra la figura para que no se muestre

    # # Guardar el mejor modelo en un archivo .pkl
    # with open(ruta_guardar, 'wb') as file:
    #     pickle.dump(best_model, file)
    best_model = gs.best_estimator_
    gs.best_estimator_
    best_scaler = gs.best_estimator_.named_steps['scaler']
    # with open(ruta_guardar, 'wb') as file:
    #     pickle.dump(best_model, file)
    with open(ruta_guardar, 'wb') as file:
        pickle.dump({
            'model': best_model,
            'scaler': best_scaler
        }, file)

    # Evaluar el modelo
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    model_results.append({
        'Model': 'Random Forest',
        'Best_model': gs.best_estimator_,
        'Best_params': gs.best_params_,
        'Best_score': gs.best_score_,
        'MSE': mse,
        'MAE': mae,
        'R-squared': r2
    })

    # Convertir los resultados en DataFrame y concatenar con el anterior
    new_results_df = pd.DataFrame(model_results)
    results_df = pd.concat([results_df, new_results_df], ignore_index=True)

    return best_model, results_df
   
# --------------------------------------------

def entrenar_xgboost_pipeline(ruta_guardar,X_train,X_test,y_train,y_test,results_df):
    model_results = []

    match = re.search(r'pca(\d+)', ruta_guardar)

    if match:
        num_pca = int(match.group(1))  # Extraer el número capturado
        print(f"El número después de 'pca' es: {num_pca}")
    else:
        num_pca = len(X_train)

    ruta_con_png = os.path.splitext(ruta_guardar)[0] + '.png'
    
    # Pipeline para XGBoost
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),  # Escalado de características
        ("pca", PCA()),  # Reducción de dimensionalidad (opcional)
        ("classifier", XGBRegressor(random_state=42, objective='reg:squarederror'))  # XGBoost para regresión
    ])

    # Espacio de búsqueda para el GridSearch
    xgb_params = {
        'scaler': [StandardScaler(), MinMaxScaler(), None],
        # "pca__n_components": [num_pca],
        # "pca__n_components": [5, 10, 0.95],
        'classifier__n_estimators': [50,100,200],  # Número de árboles
        # 'classifier__max_depth': [3, 6, 10],  # Profundidad máxima
        'classifier__max_depth': [2,3,4,5],  # Profundidad máxima
        # 'classifier__learning_rate': [0.01, 0.1],  # Tasa de aprendizaje
        'classifier__learning_rate': [0.05, 0.2],  # Tasa de aprendizaje
        'classifier__subsample': [0.8],  # Proporción de muestras utilizadas
        'classifier__colsample_bytree': [0.6,0.8]  # Proporción de características utilizadas
    }

    # Configurar GridSearchCV
    gs = GridSearchCV(estimator=pipe,
                      param_grid=xgb_params,
                      cv=3,
                      scoring='neg_mean_absolute_error',
                      verbose=2,
                      n_jobs=-1)

    # Entrenar el modelo
    gs.fit(X_train, y_train)

    # Obtener el modelo entrenado con los mejores parámetros
    best_model = gs.best_estimator_
    
    # Manejo de características dependiendo del uso de PCA
    if 'pca' in best_model.named_steps and best_model.named_steps['pca'] is not None:
        n_components = best_model.named_steps['pca'].n_components_
        features = [f'PC{i + 1}' for i in range(n_components)]  # Nombres de componentes principales
    else:
        features = X_train.columns  # Usar las columnas originales

    # Obtener la importancia de las características del modelo
    importances = best_model.named_steps['classifier'].feature_importances_

    # Crear un DataFrame para ordenar y visualizar las importancias
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })

    # Ordenar el DataFrame por la importancia de las características
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Mostrar las importancias
    # print(importance_df)

    # Graficar las importancias
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importances')

    # Guardar la imagen sin mostrarla
    plt.savefig(ruta_con_png)
    plt.close()  # Cierra la figura para que no se muestre


    # # Guardar el mejor modelo
    # with open(ruta_guardar, 'wb') as file:
    #     pickle.dump(best_model, file)
    best_model = gs.best_estimator_
    gs.best_estimator_
    best_scaler = gs.best_estimator_.named_steps['scaler']
    # with open(ruta_guardar, 'wb') as file:
    #     pickle.dump(best_model, file)
    with open(ruta_guardar, 'wb') as file:
        pickle.dump({
            'model': best_model,
            'scaler': best_scaler
        }, file)

    # Evaluación del modelo
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # # Mostrar los resultados
    # print(f"Mean Squared Error: {mse}")
    # print(f"Mean Absolute Error: {mae}")
    # print(f"R-squared: {r2}")

    model_results.append({
        'Model': 'XGBoost (Pipeline)',
        'Best_model':gs.best_estimator_,
        'Best_params':gs.best_params_,
        'Best_score':gs.best_score_,
        'MSE': mse,
        'MAE': mae,
        'R-squared': r2
    })

    # Convertir los resultados en DataFrame y concatenar con el anterior
    new_results_df = pd.DataFrame(model_results)
    results_df = pd.concat([results_df, new_results_df], ignore_index=True)

    # Mostrar los resultados
    # print(results_df)

    return best_model, results_df

# --------------------------------------------

def entrenar_lightgbm_pipeline(ruta_guardar,X_train, X_test,y_train,y_test,results_df):
    model_results = []
        
    match = re.search(r'pca(\d+)', ruta_guardar)

    if match:
        num_pca = int(match.group(1))  # Extraer el número capturado
        print(f"El número después de 'pca' es: {num_pca}")
    else:
        num_pca = len(X_train)

    ruta_con_png = os.path.splitext(ruta_guardar)[0] + '.png'

    # Pipeline para LightGBM
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),  # Escalado de características
        ("pca", PCA()),  # Reducción de dimensionalidad (opcional)
        ("classifier", lgb.LGBMRegressor(random_state=42))  # LightGBM para regresión
    ])

    # Espacio de búsqueda para el GridSearch
    lgb_params = {
        'scaler': [StandardScaler(), MinMaxScaler(), None],
        # "pca__n_components": [num_pca],
        # "pca__n_components": [5, 10, 0.95],
        'classifier__n_estimators': [100],  # Número de árboles
        'classifier__max_depth': [3, 6, 10],  # Profundidad máxima
        'classifier__learning_rate': [0.01, 0.1],  # Tasa de aprendizaje
        'classifier__subsample': [0.8],  # Proporción de muestras utilizadas
        'classifier__colsample_bytree': [0.6,0.8] , # Proporción de características utilizadas
        'classifier__num_leaves': [31, 50, 100]  # Número de hojas
    }

    # Configurar GridSearchCV
    gs = GridSearchCV(estimator=pipe,
                      param_grid=lgb_params,
                      cv=3,
                      scoring='neg_mean_absolute_error',
                      verbose=2,
                      n_jobs=-1)

    # Entrenar el modelo
    gs.fit(X_train, y_train)

    # Obtener el modelo entrenado con los mejores parámetros
    best_model = gs.best_estimator_
    
    # Manejo de características dependiendo del uso de PCA
    if 'pca' in best_model.named_steps and best_model.named_steps['pca'] is not None:
        n_components = best_model.named_steps['pca'].n_components_
        features = [f'PC{i + 1}' for i in range(n_components)]  # Nombres de componentes principales
    else:
        features = X_train.columns  # Usar las columnas originales

    # Obtener la importancia de las características del modelo
    importances = best_model.named_steps['classifier'].feature_importances_

    # Crear un DataFrame para ordenar y visualizar las importancias
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })

    # Ordenar el DataFrame por la importancia de las características
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Mostrar las importancias
    # print(importance_df)

    # Graficar las importancias
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importances')

    # Guardar la imagen sin mostrarla
    plt.savefig(ruta_con_png)
    plt.close()  # Cierra la figura para que no se muestre


    # # Guardar el mejor modelo
    # with open(ruta_guardar, 'wb') as file:
    #     pickle.dump(best_model, file)
    best_model = gs.best_estimator_
    gs.best_estimator_
    best_scaler = gs.best_estimator_.named_steps['scaler']
    # with open(ruta_guardar, 'wb') as file:
    #     pickle.dump(best_model, file)
    with open(ruta_guardar, 'wb') as file:
        pickle.dump({
            'model': best_model,
            'scaler': best_scaler
        }, file)

    # Evaluación del modelo
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # # Mostrar los resultados
    # print(f"Mean Squared Error: {mse}")
    # print(f"Mean Absolute Error: {mae}")
    # print(f"R-squared: {r2}")


    model_results.append({
        'Model': 'LightGBM (Pipeline)',
        'Best_model':gs.best_estimator_,
        'Best_params':gs.best_params_,
        'Best_score':gs.best_score_,
        'MSE': mse,
        'MAE': mae,
        'R-squared': r2
    })

    # Convertir los resultados en DataFrame y concatenar con el anterior
    new_results_df = pd.DataFrame(model_results)
    results_df = pd.concat([results_df, new_results_df], ignore_index=True)

    # Mostrar los resultados
    # print(results_df)

    return best_model, results_df

# --------------------------------------------

def entrenar_red_neuronal(ruta_guardar, X_train, X_test, y_train, y_test,results_df):
    model_results = []

    # Verificar si la ruta contiene 'log' con una regex
    if re.search(r"log", ruta_guardar):
        print("Se detectó 'log' en la ruta. Aplicando transformación logarítmica a la variable objetivo.")
        y_train = np.log1p(y_train)  # Transformar la variable objetivo (log(1 + y))
        y_test = np.log1p(y_test)    # Transformar el conjunto de prueba

    # Si también tienes un conjunto de validación, hacer un split adicional
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # Aplicar el StandardScaler
    scaler = StandardScaler()

    # Ajustar el escalador con los datos de entrenamiento y transformar X_train
    X_train_scaled = scaler.fit_transform(X_train)

    # Transformar X_test y X_valid con el mismo escalador
    X_test_scaled = scaler.transform(X_test)
    X_valid_scaled = scaler.transform(X_valid)

    # Definir la arquitectura del modelo de red neuronal
    model = keras.models.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=X_train_scaled.shape[1:]),  # Capa densa de 64 neuronas con ReLU
        keras.layers.Dense(32, activation='relu'),  # Capa densa de 32 neuronas con ReLU
        keras.layers.Dense(1)  # Capa de salida, 1 neurona para la regresión
    ])

    # Compilar el modelo
    model.compile(loss="mean_absolute_error",
                  metrics=['mean_absolute_error'],  # Usamos el error absoluto medio para regresión
                  optimizer=keras.optimizers.Adam(learning_rate=0.001))  # Optimizer Adam con tasa de aprendizaje ajustada

    # Ajuste del modelo a los datos escalados
    history = model.fit(X_train_scaled, y_train,  # Entrenamos con los datos de entrenamiento escalados
                        epochs=50,  # Aumentamos las épocas para un mejor ajuste
                        batch_size=32,  # Tamaño de batch más grande para entrenamiento
                        validation_data=(X_valid_scaled, y_valid))  # Validación con los datos de validación escalados

    # # Guardar el mejor modelo en un archivo .pkl
    # with open(ruta_guardar, 'wb') as file:
    #     pickle.dump(model, file)
    # Guardar el mejor modelo en un archivo .pkl
    with open(ruta_guardar, 'wb') as file:
        pickle.dump({
            'model': model,
            'scaler': scaler
        }, file)

    # Evaluación del modelo
    y_pred_nn = model.predict(X_test_scaled)

    # Si aplicaste logarítmica, destransformar para comparar correctamente
    if re.search(r"log", ruta_guardar):
        y_pred_nn = np.expm1(y_pred_nn)  # Invertir la transformación logarítmica
        y_test = np.expm1(y_test)       # Invertir la transformación logarítmica

    mse_nn = mean_squared_error(y_test, y_pred_nn)
    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    r2_nn = r2_score(y_test, y_pred_nn)

    model_results.append({
        'Model': 'Neural Network',
        'Best_model': '-',
        'Best_params': '-',
        'Best_score': '-',
        'MSE': mse_nn,
        'MAE': mae_nn,
        'R-squared': r2_nn
    })

    # Convertir los resultados en DataFrame y concatenar con el anterior
    new_results_df = pd.DataFrame(model_results)
    results_df = pd.concat([results_df, new_results_df], ignore_index=True)

    # Devolver los resultados actualizados
    return results_df