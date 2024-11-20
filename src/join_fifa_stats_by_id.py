"""import pandas as pd
from fuzzywuzzy import fuzz
from rapidfuzz import fuzz




def asignar_id_fifa_optimizado(df1, df2):
    # Asegúrate de que las columnas están en mayúsculas para la comparación
    df1['nom_jugad_upper'] = df1['nom_jugad'].str.upper()
    df2['short_name_upper'] = df2['short_name'].str.upper()
    df2['long_name_upper'] = df2['long_name'].str.upper()

    # Hacer merge de las primeras comprobaciones en base a 'nom_jugad' y 'short_name'
    df1 = pd.merge(df1, df2[['id_fifa', 'short_name_upper']], 
                   left_on='nom_jugad_upper', 
                   right_on='short_name_upper', 
                   how='left')

    # Si no hay match en short_name, hacer la segunda comprobación con long_name
    df1 = pd.merge(df1, df2[['id_fifa', 'long_name_upper']], 
                   left_on='nom_jugad_upper', 
                   right_on='long_name_upper', 
                   how='left', suffixes=('', '_long'))

    # Llenar la columna 'id_fifa' donde sea 0 con el id_fifa de las otras columnas
    df1['id_fifa'] = df1['id_fifa'].fillna(df1['id_fifa_long']).fillna(0).astype(int)

    # Eliminar las columnas auxiliares
    df1.drop(columns=['short_name_upper', 'long_name_upper', 'id_fifa_long'], inplace=True)

    # Fuzzy matching para las filas que aún no tienen id_fifa (por si se dieron casos sin match exacto)
    def fuzzy_match(row, df2):
        if row['id_fifa'] != 0:  # Si ya tiene id_fifa, no hacer más comprobaciones
            return row

        nom_jugad = row['nom_jugad_upper']
        mejor_ratio = 0
        mejor_match = None

        # Fuzzy matching con 'short_name'
        for _, row2 in df2.iterrows():
            ratio = fuzz.ratio(nom_jugad, row2['short_name_upper'])
            if ratio > mejor_ratio:
                mejor_ratio = ratio
                mejor_match = row2

        # Si no hay match en 'short_name', intentar con 'long_name'
        if mejor_ratio < 80:
            for _, row2 in df2.iterrows():
                ratio = fuzz.ratio(nom_jugad, row2['long_name_upper'])
                if ratio > mejor_ratio:
                    mejor_ratio = ratio
                    mejor_match = row2

        # Si se encontró una coincidencia suficiente, asignar el id_fifa
        if mejor_ratio >= 80 and mejor_match is not None:
            row['id_fifa'] = mejor_match['id_fifa']
        return row

    # Aplicar fuzzy matching solo a las filas donde 'id_fifa' sigue siendo 0
    df1 = df1.apply(fuzzy_match, axis=1, df2=df2)

    return df1

# Uso de la función
df1_actualizado = asignar_id_fifa_optimizado(df_reales, df_fifa)

# Puedes guardar el resultado si quieres: 
df1_actualizado.to_csv("../data/All_stats_2022_id.csv", index=False)

print("holi")"""

import pandas as pd
import re
from unidecode import unidecode
from rapidfuzz import fuzz, process

# Función de preprocesamiento para normalizar nombres
def limpiar_nombre(nombre):
    nombre = unidecode(nombre).upper()  # Quitar acentos y convertir a mayúsculas
    nombre = re.sub(r'\W+', '', nombre)  # Eliminar caracteres no alfanuméricos (espacios y símbolos)
    return nombre

# Función 1: Limpieza completa y merge inicial
def asignar_id_fifa_con_merge(df1, df2):
    # Aplicar la función de limpieza a las columnas de nombres en ambos DataFrames
    df1['nom_jugad_limpio'] = df1['nom_jugad'].apply(limpiar_nombre)
    df2['short_name_limpio'] = df2['short_name'].apply(limpiar_nombre)
    df2['long_name_limpio'] = df2['long_name'].apply(limpiar_nombre)
    
    # Merge con short_name
    df1 = pd.merge(df1, df2[['id_fifa', 'short_name_limpio']], 
                   left_on='nom_jugad_limpio', 
                   right_on='short_name_limpio', 
                   how='left')

    # Merge con long_name si no hubo match en short_name
    df1 = pd.merge(df1, df2[['id_fifa', 'long_name_limpio']], 
                   left_on='nom_jugad_limpio', 
                   right_on='long_name_limpio', 
                   how='left', suffixes=('', '_long'))

    # Rellenar con id_fifa
    df1['id_fifa'] = df1['id_fifa'].fillna(df1['id_fifa_long']).fillna(0).astype(int)

    # Eliminar columnas auxiliares
    df1.drop(columns=['nom_jugad_limpio', 'short_name_limpio', 'long_name_limpio', 'id_fifa_long'], inplace=True)
    return df1

# Función 2: Comprobaciones adicionales solo en registros sin id_fifa
def completar_id_fifa(df1, df2):
    # Seleccionar solo los registros que aún no tienen id_fifa asignado
    df_sin_id = df1[df1['id_fifa'] == 0].copy()
    
    # Realizar las comprobaciones de coincidencias adicionales
    for index, row in df_sin_id.iterrows():
        nom_jugad_limpio = limpiar_nombre(row['nom_jugad'])
        # Comparar con short_name y long_name en df2
        match_short = df2[df2['short_name_limpio'] == nom_jugad_limpio]
        match_long = df2[df2['long_name_limpio'] == nom_jugad_limpio]
        
        # Si se encuentra coincidencia en short_name
        if not match_short.empty:
            df1.at[index, 'id_fifa'] = match_short['id_fifa'].values[0]
        # Si se encuentra coincidencia en long_name
        elif not match_long.empty:
            df1.at[index, 'id_fifa'] = match_long['id_fifa'].values[0]
        else:
            # Dividir nom_jugad en dos y comparar por partes
            partes = nom_jugad_limpio.split()
            if len(partes) >= 2:
                match_partial_short = df2[
                    (df2['short_name_limpio'] == partes[0]) &
                    (df2['long_name_limpio'] == partes[1])
                ]
                if not match_partial_short.empty:
                    df1.at[index, 'id_fifa'] = match_partial_short['id_fifa'].values[0]
    
    return df1

from fuzzywuzzy import process, fuzz
import re

# Función principal para completar el ID FIFA
def completar_id_fifa_fuzzy(df1, df2, threshold=85, threshold_final=50):
    # Seleccionar solo los registros que aún no tienen id_fifa asignado
    df_sin_id = df1[df1['id_fifa'] == 0].copy()
    
    # Convertir los nombres a listas para fuzzy matching
    nombres_fifa = df2[['id_fifa', 'short_name_limpio', 'long_name_limpio']].values.tolist()
    
    for index, row in df_sin_id.iterrows():
        nom_jugad_limpio = limpiar_nombre(row['nom_jugad'])
        
        # Fuzzy matching con short_name
        match_short = process.extractOne(nom_jugad_limpio, 
                                         [n[1] for n in nombres_fifa], 
                                         scorer=fuzz.ratio)
        if match_short and match_short[1] >= threshold:
            id_fifa = next((n[0] for n in nombres_fifa if n[1] == match_short[0]), None)
            if id_fifa:
                df1.at[index, 'id_fifa'] = id_fifa
                continue
        
        # Fuzzy matching con long_name
        match_long = process.extractOne(nom_jugad_limpio, 
                                        [n[2] for n in nombres_fifa if n[2] is not None], 
                                        scorer=fuzz.ratio)
        if match_long and match_long[1] >= threshold:
            id_fifa = next((n[0] for n in nombres_fifa if n[2] == match_long[0]), None)
            if id_fifa:
                df1.at[index, 'id_fifa'] = id_fifa
                continue
        
        # Comprobación con apellido y nombre (si el nombre tiene más de una palabra)
        nombres_split = nom_jugad_limpio.split()
        if len(nombres_split) >= 2:
            first_name = nombres_split[0]
            last_name = nombres_split[-1]  # Suponemos que el último es el apellido
            
            # Fuzzy matching con el apellido
            match_last = process.extractOne(last_name, 
                                            [n[1].split()[-1] for n in nombres_fifa], 
                                            scorer=fuzz.ratio)
            if match_last and match_last[1] >= threshold:
                id_fifa = next((n[0] for n in nombres_fifa if n[1].split()[-1] == match_last[0]), None)
                if id_fifa:
                    df1.at[index, 'id_fifa'] = id_fifa
                    continue
            
            # Fuzzy matching con el nombre
            match_first = process.extractOne(first_name, 
                                             [n[1].split()[0] for n in nombres_fifa], 
                                             scorer=fuzz.ratio)
            if match_first and match_first[1] >= threshold:
                id_fifa = next((n[0] for n in nombres_fifa if n[1].split()[0] == match_first[0]), None)
                if id_fifa:
                    df1.at[index, 'id_fifa'] = id_fifa
                    continue
        
        # Fuzzy matching con un umbral mucho más laxo (final con threshold bajo)
        match_final = process.extractOne(nom_jugad_limpio, 
                                         [n[1] for n in nombres_fifa], 
                                         scorer=fuzz.ratio)
        if match_final and match_final[1] >= threshold_final:
            id_fifa = next((n[0] for n in nombres_fifa if n[1] == match_final[0]), None)
            if id_fifa:
                df1.at[index, 'id_fifa'] = id_fifa
                continue
        
        # Último intento con long_name si no se ha encontrado nada
        match_final_long = process.extractOne(nom_jugad_limpio, 
                                              [n[2] for n in nombres_fifa], 
                                              scorer=fuzz.ratio)
        if match_final_long and match_final_long[1] >= threshold_final:
            id_fifa = next((n[0] for n in nombres_fifa if n[2] == match_final_long[0]), None)
            if id_fifa:
                df1.at[index, 'id_fifa'] = id_fifa

    return df1

# Cargar los datos
# df_reales = pd.read_csv("./data/DatosReales_15_10_2024.csv")
df_reales = pd.read_csv("../data/All_stats_2022.csv")
df_fifa = pd.read_csv("../data/players_22.csv")

# Filtrar las columnas necesarias del dataframe FIFA
df_fifa = df_fifa[df_fifa["league_name"].isin(["Italian Serie A", "French Ligue 2","English Premier League",
                                               "Spain Primera Division","German 1. Bundesliga","Portuguese Liga ZON SAGRES",
                                               "Holland Eredivisie"])]

df_fifa = df_fifa[["id_fifa", "short_name", "long_name"]]

# Paso 1: Ejecuta la primera función para realizar el merge inicial y asignar id_fifa
df_reales = asignar_id_fifa_con_merge(df_reales, df_fifa)
print(df_reales.head())

# Paso 2: Completa id_fifa en los registros restantes sin coincidencias iniciales
df_reales = completar_id_fifa(df_reales, df_fifa)

# Paso 3: Aplica fuzzy matching para los registros que aún no tienen id_fifa
# df_reales = completar_id_fifa_fuzzy(df_reales, df_fifa)
df1_actualizado = completar_id_fifa_fuzzy(df_reales, df_fifa, threshold=85, threshold_final=79)

# Finalmente, guarda el resultado en un CSV
df_reales.to_csv("../data/All_stats_2022_id.csv", index=False)
