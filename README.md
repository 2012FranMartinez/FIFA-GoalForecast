<h1 align="center" style="font-size: 50px;">>>⚽️🥅 OctoBet-GoalForecast <<</h1>

Este repositorio contiene un proyecto de predicción de goles basado en datos reales de jugadores de fútbol del año 2022. Utilizando técnicas de Machine Learning y análisis estadístico, el objetivo es construir modelos que puedan anticipar la cantidad de goles de un jugador en un año.

## 🎯 Objetivos del Proyecto

Este proyecto tiene como objetivo predecir goles en partidos de fútbol utilizando Machine Learning.

Limpiar y explorar los datos de estadísticas de jugadores de 2022 para identificar características clave en la probabilidad de marcar goles.
Desarrollar y entrenar modelos de predicción de goles con algoritmos de Machine Learning.
Evaluar y comparar el rendimiento de los modelos para seleccionar el más preciso.
Crear visualizaciones y métricas que ayuden a interpretar los resultados.
Explora los notebooks y ejecuta el código para entrenar y evaluar los modelos.

## 🧠 Estructura del Repositorio

**app_streamlit/**: Contiene la visualización final para el cliente del modelo ganador entrenado.
Ejecutar en terminal-> streamlit run streamlit_final_app.py

**chromedriver/**: Contiene el chromedriver utilizado para llevar a cabo el web scraping del proyecto. Actualiza el tuyo si vas a hacer pruebas con esta parte.

**data/**: Contiene los conjuntos de datos con estadísticas reales de los jugadores del año 2022.
Y los diferentes dataframes utilizados con datos raw, procesados, y el train/test

**docs/**: Contiene las presentaciones tanto técnica como para negocio y las imágenes utilizadas para este fin.

**models/**: Modelos entrenados guardados para su reutilización y evaluación.

**notebooks/**: Notebooks de Jupyter con el análisis exploratorio de datos (EDA), la creación de modelos y el ajuste de parámetros.

**src/**: Scripts de Python con el código modular para la preparación de datos, creación de modelos y predicción.

README.md: Información general sobre el proyecto.

## 💻 Tecnologías Utilizadas

**Python**: Para procesamiento y modelado de datos.
**Pandas y NumPy**: Para manipulación de datos.
**scikit-learn**: Para entrenamiento de modelos de predicción.
**Seaborn y Matplotlib**: Para visualización de datos y resultados.

## Requisitos

-   **Python 3.x** (asegúrate de tenerlo instalado en tu sistema)

## 💾 Cómo Empezar

1. Clona el repositorio en tu máquina local:
    ```bash
    git clone https://github.com/2012FranMartinez/FIFA-GoalForecast.git
    ```
2. Navega a la carpeta del proyecto:
    ```bash
    cd tu_repositorio
    ```
3. Crea un entorno virtual (opcional pero recomendado:

    ```bash
    (Linux/macOS)-> python3 -m venv venv

    (Windows)-> python -m venv venv
    ```

4. Activa el entorno virtual:

    ```bash
    (Linux/macOS)-> source venv/bin/activate


    (Windows)-> .\venv\Scripts\activate
    ```

5. Instala las dependencias del proyecto:
    ```bash
    pip install -r requirements.txt
    ```

# Contribuciones

¡Las contribuciones son bienvenidas! Si tienes sugerencias o mejoras, abre un Issue o envía un Pull Request.
