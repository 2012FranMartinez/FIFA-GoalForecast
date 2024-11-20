import pandas as pd
import glob
import os
import re
from functools import reduce

def combinar_archivos_dinamico(ruta_carpeta):
    # Búsqueda de archivos CSV en la carpeta
    patron_archivos = os.path.join(ruta_carpeta, "*.csv")
    
    # Diccionario para almacenar los tipos encontrados (Offensive, xG, etc.) y sus respectivos dataframes por año
    tipos_archivo = {}

    # Expresión regular para extraer el tipo de archivo (como "Offensive", "xG", etc.) y el año
    regex = r'^[A-Z]\w{2}_\w+_(\d{4})\.csv'  # Busca los archivos que siguen el patrón

    # Itera sobre cada archivo CSV en la carpeta
    for archivo in glob.glob(patron_archivos):
        # Extrae el nombre base del archivo
        nombre_archivo = os.path.basename(archivo)

        # Usamos la regex para extraer el tipo y el año del nombre del archivo
        match = re.match(regex, nombre_archivo)
        if match:
            # El tipo es la parte del nombre entre la liga (Ejemplo: Esp, Eng) y el año
            tipo = nombre_archivo.split('_')[1]  # Obtiene el tipo (Offensive, xG, etc.)
            año = match.group(1)  # Año (ej. "2022")

            # Lee el archivo CSV en un DataFrame
            df = pd.read_csv(archivo)

            # Extrae la liga del nombre del archivo (los primeros 3 caracteres antes del "_")
            liga = nombre_archivo.split('_')[0]

            # Agrega las columnas de año y liga
            df['Año'] = año
            df['Liga'] = liga

            # Si aún no hemos encontrado el tipo, inicializamos una lista para ese tipo
            if tipo not in tipos_archivo:
                tipos_archivo[tipo] = {}

            # Si no hemos encontrado el año para este tipo, inicializamos una lista para ese año
            if año not in tipos_archivo[tipo]:
                tipos_archivo[tipo][año] = []

            # Añadimos el DataFrame a la lista correspondiente (por tipo y año)
            tipos_archivo[tipo][año].append(df)

    # Ahora guardamos los archivos CSV para cada tipo y año encontrados
    for tipo, años in tipos_archivo.items():
        for año, dfs in años.items():
            # Concatena todos los DataFrames de este tipo y año
            df_final = pd.concat(dfs, ignore_index=True)

            # Guarda el resultado en un archivo CSV con el formato deseado
            df_final.to_csv(f"All_{tipo}_{año}.csv", index=False)
            print(f"Archivos combinados guardados en: All_{tipo}_{año}.csv")

def procesar_datos(ruta_archivo):
    # Cargar el archivo CSV en un DataFrame
    df = pd.read_csv(ruta_archivo)
    
    # Eliminar las columnas innecesarias
    df.drop(columns=['Unnamed: 0', 'Unnamed: 2'], inplace=True)
    
    # Separar la columna 'Player' por saltos de línea
    df[['num', 'nom_jugad', 'equipo_ns_posicion']] = df['Player'].str.split('\n', expand=True, n=2)
    
    # Separar 'equipo_ns_posicion' en equipo, ns y posición
    equipo_ns_posicion = df['equipo_ns_posicion'].str.split(',', expand=True)
    
    # Ajustar según la cantidad de columnas generadas
    df['equipo'] = equipo_ns_posicion[0].str.strip()
    df['ns'] = equipo_ns_posicion[1].str.strip()
    # Unimos el resto de las columnas en caso de posiciones múltiples
    df['posicion'] = equipo_ns_posicion.iloc[:, 2:].apply(lambda x: ','.join(x.dropna().str.strip()), axis=1)
    
    # Convertir 'ns' a tipo numérico si aplica
    df['ns'] = pd.to_numeric(df['ns'], errors='coerce')
    
    # Eliminar la columna auxiliar 'equipo_ns_posicion'
    df.drop(columns=['equipo_ns_posicion'], inplace=True)

    print(df.head())
    
    # Guardar el DataFrame procesado en un nuevo archivo CSV
    df.to_csv(f"procesado_{os.path.basename(ruta_archivo)}", index=False)

    # Retornar el DataFrame procesado
    return df

def combinar_archivos_por_jugador(*rutas_archivo):
    # Verificar que se han pasado al menos dos archivos
    if len(rutas_archivo) < 2:
        raise ValueError("Se deben proporcionar al menos dos rutas de archivo.")

    # Extraer el año del primer archivo coincidente
    anio = None
    for ruta in rutas_archivo:
        match = re.search(r'\d{4}', ruta)
        if match:
            anio = match.group(0)
            break
    if not anio:
        raise ValueError("No se encontró un año en los nombres de archivo proporcionados.")

    # Cargar los archivos en DataFrames
    dataframes = [pd.read_csv(ruta) for ruta in rutas_archivo]

    # Unir todos los DataFrames en uno solo usando 'Player' como clave
    df_combinado = reduce(lambda left, right: pd.merge(left, right, on='nom_jugad', how='outer'), dataframes)
    # # Unir todos los DataFrames en uno solo usando 'Player' como clave y realizando un "outer join"
    # df_combinado = reduce(lambda left, right: pd.merge(left, right, on='Player', how='outer'), dataframes)

    # Guardar el archivo combinado con el nombre solicitado
    nombre_salida = f"All_stats_{anio}.csv"
    df_combinado.to_csv(nombre_salida, index=False)
    
    return df_combinado, nombre_salida

# # 1 Juntar todos los del año en el mismo:
# ruta_carpeta = r"C:\Users\Francisco\Desktop\Proyectos\Proyecto_ml_fifa_goals_forecast\FIFA-GoalForecast"
# combinar_archivos_dinamico(ruta_carpeta)


# # 2 Procesar los datos:
# ruta_archivo = r"C:\Users\Francisco\Desktop\Proyectos\Proyecto_ml_fifa_goals_forecast\FIFA-GoalForecast\data\All_Offensive_2022.csv"
# ruta_archivo = r"C:\Users\Francisco\Desktop\Proyectos\Proyecto_ml_fifa_goals_forecast\FIFA-GoalForecast\data\All_xG_2022.csv"
# procesar_datos(ruta_archivo)
# print("hola")


# 3 Juntar por jugador las Stats:
rutas_archivo = ['procesado_All_Offensive_2022.csv', 'procesado_All_xG_2022.csv']
df_combinado, nombre_salida = combinar_archivos_por_jugador(*rutas_archivo)
print(f"Archivo combinado guardado como: {nombre_salida}")
print(df_combinado.head())
print("ajja")