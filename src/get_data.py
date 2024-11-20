from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
from datetime import datetime
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os

from selenium import webdriver
from selenium.webdriver.chrome.service import Service


import os
from datetime import datetime

import os
from datetime import datetime

# Obtén la fecha y hora actual en el formato que desees (por ejemplo, "YYYY-MM-DD_HH-MM-SS")
fecha_hora_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define el nombre de la carpeta
nombre_carpeta = f"./src/prueba_{fecha_hora_actual}"

# Verifica si la carpeta existe y créala si no
if not os.path.exists(nombre_carpeta):
    os.makedirs("./"+nombre_carpeta)
    print(f"Carpeta creada: {nombre_carpeta}")
else:
    print(f"La carpeta '{nombre_carpeta}' ya existe.")




# Configurar chromedriver
chrome_options = webdriver.chrome.options.Options()
driver = webdriver.Chrome(options=chrome_options)
chrome_driver = "./chromedriver/chromedriver.exe"
service_to_pass = Service(executable_path=chrome_driver)
driver = webdriver.Chrome(service=service_to_pass, options=chrome_options)


# Fijar años a extraer
# años = [(i, i + 1) for i in range(2019, 2025)]  
años = [(2019,2020)]
print(años)

# Tablas a extraer
tablas_to_extract = {"xG":"statistics-table-xg",
                     "Offensive":"stage-top-player-stats-offensive",
                     "Passing":"stage-top-player-stats-passing",
                     "Summary":"statistics-table-summary"}

# Componer Url´s
url_base = "https://www.whoscored.com/Regions"
ligas = {
    'Esp': '/206/Tournaments/4/Seasons/8681/Stages/19895/PlayerStatistics/Spain-LaLiga-2019-2020',
    'Ing': '/252/Tournaments/2/Seasons/10316/Stages/23400/PlayerStatistics/England-Premier-2019-2020',
    'Ita': '/108/Tournaments/5/Seasons/10375/Stages/23490/PlayerStatistics/Italy-Serie-A-2019-2020',
    'Ger': '/81/Tournaments/3/Seasons/10365/Stages/23471/PlayerStatistics/Germany-Bundesliga-2019-2020',
    'Fra': '/74/Tournaments/22/Seasons/10329/Stages/23414/PlayerStatistics/France-Ligue-1-2019-2020',
    'Por': '/177/Tournaments/21/Seasons/10378/Stages/23494/PlayerStatistics/Portugal-Liga-2019-2020',
    'Ned': '/155/Tournaments/13/Seasons/10321/Stages/23405/PlayerStatistics/Netherlands-Eredivisie-2019-2020'    
}

"""'Ing': '/252/Tournaments/2/Seasons/10316/Stages/23400/PlayerStatistics/England-Premier-2019-2020',
    'Ita': '/108/Tournaments/5/Seasons/10375/Stages/23490/PlayerStatistics/Italy-Serie-A-2019-2020',
    'Ger': '/81/Tournaments/3/Seasons/10365/Stages/23471/PlayerStatistics/Germany-Bundesliga-2019-2020',
    'Fra': '/74/Tournaments/22/Seasons/10329/Stages/23414/PlayerStatistics/France-Ligue-1-2019-2020',
    'Por': '/177/Tournaments/21/Seasons/10378/Stages/23494/PlayerStatistics/Portugal-Liga-2019-2020',
    'Ned': '/155/Tournaments/13/Seasons/10321/Stages/23405/PlayerStatistics/Netherlands-Eredivisie-2019-2020',
    'Esp': '/206/Tournaments/4/Seasons/8681/Stages/19895/PlayerStatistics/Spain-LaLiga-2019-2020'
    
    ,"Ing","Ita","Ger","Fra","Por","Ned"
    """

# ligas_sufijo = ["Ing","Ita","Ger","Fra","Por","Ned"]
urls_finales = []
for año_inicio, año_fin  in años:
    for liga,liga_url in ligas.items():
        url = url_base+liga_url+str(año_inicio)+"-"+str(año_fin)
        # print(url)
        urls_finales.append(url)


# ----------------------------------------------------------
paises = ["Esp","Ing","Ita","Ger","Fra","Por","Ned"]

# RECORRER PAISES (LIGAS)
for pais in paises:
    print(f"RECORRER PAISES (LIGAS):{pais}")
    print(url_base+ligas[pais])
    driver.get(url_base+ligas[pais])
    # Hay cookies?
    print("Cookies")
    elementos = driver.find_elements(By.XPATH,'//button[@mode="primary"]')
    if len(elementos) != 0:
        elemento = driver.find_element(By.XPATH,'//button[@mode="primary"]')
        elemento.click()
    try:
        close_button = driver.find_element(By.XPATH, '//button[contains(@class,"webpush") and @aria-label="Close this dialog"]')
        close_button.click()  # Si se encuentra, le hace clic
        print("Elemento encontrado y clic realizado.")
    except NoSuchElementException:
        print("Elemento no encontrado. No se realizó ningún clic.")
    
    # [RECORRER POR AÑOS]
    for años_tupla in años:
        print(f"[RECORRER AÑOS]:{años_tupla}")
        print(años_tupla)
        print("Desplegable años")
        # Desplegable temporadas
        time.sleep(1)
        elemento = driver.find_element(By.XPATH, '//select[@id="seasons"]')
        driver.execute_script("arguments[0].scrollIntoView();", elemento)
        dropdown = driver.find_element(By.XPATH, '//select[@id="seasons"]')
        dropdown.click()
        time.sleep(2)
        # Clickar en el año
        opcion = driver.find_element(By.XPATH, f'//option[text()="{años_tupla[0]}/{años_tupla[1]}"]')
        opcion.click()
        print("Clickado en el desplegable")
        time.sleep(5)
        
        # Clickar en Player Statistics si o si
        click_stadistics = False
        while click_stadistics == False:
            print("Comprobando estadistics")
            esta = driver.find_elements(By.XPATH,'//a[text()="Player Statistics"]/..')
            if len(esta)>0:
                elemento = driver.find_element(By.XPATH,'//a[text()="Player Statistics"]/..')
                time.sleep(2)
                elemento.click()
                click_stadistics = True
                time.sleep(2)
            else:
                time.sleep(2)

        # [RECORRER POR TABLAS (offensive,xG,etc)]
        for clave_tablas, nom_tablas in tablas_to_extract.items():
            print(f"[RECORRER TABLAS]. Clave: {clave_tablas} Nom_tabla:{nom_tablas}")
            # Clickar en la tabla que toque
            elemento = driver.find_element(By.XPATH,f'//div[@class="ws-panel"]//a[text()="{clave_tablas}"]')
            elemento.click()
            time.sleep(5)

            # Hay resultados?
            e = driver.find_elements(By.XPATH,f'//div[@id="{nom_tablas}"]//td[text()="There are no results to display"]')
            if len(e)>0:
                results = False
                break

            
            # Descargar toda la info
            # Click en All players
            elemento = driver.find_element(By.XPATH,f'//div[@id="stage-top-player-stats-{clave_tablas.lower()}"]//a[text()="All players"]')
            elemento.click()
            time.sleep(5)

            # Muevete al header
            print(f'//div[@id="{nom_tablas}"]//thead[@id="player-table-statistics-head"]//th')
            elemento = driver.find_element(By.XPATH, f'//div[@id="{nom_tablas}"]//thead[@id="player-table-statistics-head"]//th')
            driver.execute_script("arguments[0].scrollIntoView();", elemento)
            
            # Extrae los encabezados de las columnas de thead
            encabezados = driver.find_elements(By.XPATH, f'//div[@id="{nom_tablas}"]//thead[@id="player-table-statistics-head"]//th')
            nombres_columnas = [encabezado.text for encabezado in encabezados]

            # Inicializa una lista para almacenar todos los datos
            todos_los_datos = []

            # Descargar filas mientras el botón "next" esté presente
            while True:
                # Localiza la tabla y extrae los datos
                tabla_id = nom_tablas
                tabla = driver.find_element(By.ID, tabla_id)
                filas = tabla.find_elements(By.TAG_NAME, "tr")

                # Descargar cada fila por jugador
                for fila in filas:
                    # print("[RECORRER FILAS]")
                    celdas = fila.find_elements(By.TAG_NAME, "td")
                    # print(celdas)
                    fila_datos = [celda.text for celda in celdas]
                    if fila_datos:  # Agrega solo filas con datos
                        todos_los_datos.append(fila_datos)
                    # print(todos_los_datos)
                
                print("Es el utlimo (Sigue next)?")
                # Verifica si el botón "next" está presente
                try:
                    # boton_siguiente = driver.find_element(By.XPATH, '(//a[@class="option  clickable " and @id="next"])[2]')
                    boton_siguiente = driver.find_element(By.XPATH, f'//div[@id="statistics-paging-{clave_tablas.lower()}"]//a[@class="option  clickable " and @id="next"]')
                    
                    # driver.execute_script("arguments[0].scrollIntoView();", boton_siguiente)
                    boton_siguiente.click()
                    time.sleep(2)  # Espera para cargar la siguiente página
                except NoSuchElementException:
                    # Si el botón no está presente, rompe el bucle
                    # Convierte los datos recopilados en un DataFrame de Pandas con los encabezados de columnas extraídos
                    print("Guardando df")
                    df = pd.DataFrame(todos_los_datos, columns=nombres_columnas)
                    print(df.head(50))
                    # print(df)
                    # nombre_output = f"goles_{año_fin}.csv"
                    nom_output = f"{pais}_{clave_tablas}_{años_tupla[1]}.csv"
                    ruta_archivo = os.path.join(nombre_carpeta, nom_output)
                    print(ruta_archivo)
                    print(f"Guardando en: {ruta_archivo}")
                    # df.to_csv(f"{pais}_{clave_tablas}_{años_tupla[0]}_{años_tupla[1]}.csv")
                    df.to_csv(ruta_archivo)
                    break
                            
                        
# Cierra el navegador
driver.quit()





"""
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import os
from selenium.webdriver.support.ui import Select

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Configurar chromedriver
chrome_options = webdriver.chrome.options.Options()
driver = webdriver.Chrome(options=chrome_options)
chrome_driver = "./chromedriver/chromedriver.exe"
service_to_pass = Service(executable_path=chrome_driver)
driver = webdriver.Chrome(service=service_to_pass, options=chrome_options)


# Fijar años a extraer
años = [(i, i + 1) for i in range(2019, 2025)]  
# años = [(2022,2023)]
print(años)

# Tablas a extraer
tablas_to_extract = {"xG":"statistics-table-xg",
                     "Offensive":"stage-top-player-stats-offensive",
                     "Passing":"stage-top-player-stats-passing",
                     "Summary":"statistics-table-summary"}

# Componer Url´s
url_base = "https://www.whoscored.com/Regions"
ligas = {'Ing': '/252/Tournaments/2/Seasons/10316/Stages/23400/PlayerStatistics/England-Premier-2019-2020',
        'Ita': '/108/Tournaments/5/Seasons/10375/Stages/23490/PlayerStatistics/Italy-Serie-A-2019-2020',
        'Ger': '/81/Tournaments/3/Seasons/10365/Stages/23471/PlayerStatistics/Germany-Bundesliga-2019-2020',
        'Fra': '/74/Tournaments/22/Seasons/10329/Stages/23414/PlayerStatistics/France-Ligue-1-2019-2020',
        'Por': '/177/Tournaments/21/Seasons/10378/Stages/23494/PlayerStatistics/Portugal-Liga-2019-2020',
        'Ned': '/155/Tournaments/13/Seasons/10321/Stages/23405/PlayerStatistics/Netherlands-Eredivisie-2019-2020'
}
# 'esp': '/206/Tournaments/4/Seasons/8681/Stages/19895/PlayerStatistics/Spain-LaLiga-',
ligas_sufijo = ["Ing","Ita","Ger","Fra","Por","Ned"]
urls_finales = []
for año_inicio, año_fin  in años:
    for liga,liga_url in ligas.items():
        url = url_base+liga_url+str(año_inicio)+"-"+str(año_fin)
        # print(url)
        urls_finales.append(url)


# ---------------------------------------------------------        






# Recorrer las Url [REOCRRER PAISES]
for i,url in enumerate(urls_finales):
    # Cargar página
    print(f"Cargar página{url}")
    driver.get(url)
    time.sleep(2)

    # Hay cookies?
    print("Cookies")
    elementos = driver.find_elements(By.XPATH,'//button[@mode="primary"]')
    if len(elementos) != 0:
        elemento = driver.find_element(By.XPATH,'//button[@mode="primary"]')
        elemento.click()
    try:
        close_button = driver.find_element(By.XPATH, '//button[contains(@class,"webpush") and @aria-label="Close this dialog"]')
        close_button.click()  # Si se encuentra, le hace clic
        print("Elemento encontrado y clic realizado.")
    except NoSuchElementException:
        print("Elemento no encontrado. No se realizó ningún clic.")

    # print(años[i])

    # [RECORRER POR AÑOS]
    for años_tupla in años:
        print(i)
        print("Desplegable años")
        try:
            # Desplegable temporadas
            time.sleep(1)
            dropdown = driver.find_element(By.XPATH, '//select[@id="seasons"]')
            dropdown.click()
            time.sleep(2)
            opcion = driver.find_element(By.XPATH, f'//option[text()="{años_tupla[0]}/{años_tupla[1]}"]')
            opcion.click()
            print("Clickado en el desplegable")
            time.sleep(5)
            
            # dropdown = WebDriverWait(driver, 1).until(
            #     EC.visibility_of_element_located((By.XPATH, '//select[@id="seasons"]'))
            # )

            # # Primero, haz clic en el desplegable
            # dropdown.click()
            
            # print("Haciendo click en desplegable años")

            # # Espera hasta que la opción deseada sea visible
            # opcion = WebDriverWait(driver, 5).until(
            #     EC.visibility_of_element_located((By.XPATH, f'//option[text()="{años[i][0]}/{años[i][1]}"]'))
            # )
            
            # # Luego, haz clic en la opción
            # opcion.click()

            print(f"Opción seleccionada: {años_tupla[0]}/{años_tupla[1]}")
            
        except NoSuchElementException:
            print(f'Elemento no encontrado: //option[text()="{años_tupla[0]}/{años_tupla[1]}"]. No se realizó ninguna acción.')
        except TimeoutException:
            print(f'Tiempo de espera agotado. El elemento no se encontró //option[text()="{años_tupla[0]}/{años_tupla[1]}"]')
            driver.quit()


    
        click_stadistics = False
        while click_stadistics == False:
            print("Comprobando estadistics")
            esta = driver.find_elements(By.XPATH,'//a[text()="Player Statistics"]/..')
            if len(esta)>0:
                elemento = driver.find_element(By.XPATH,'//a[text()="Player Statistics"]/..')
                time.sleep(2)
                elemento.click()
                click_stadistics = True
                time.sleep(2)
            else:
                time.sleep(2)


            # elemento = driver.find_element(By.XPATH,'//a[text()="Player Statistics"]')
            # elemento.click()
            # time.sleep(2)

        # # Espera explícita hasta que el elemento sea visible
        # try:
        #     # Espera hasta que el botón sea visible (máximo 10 segundos)
        #     elemento = WebDriverWait(driver, 5).until(
        #         EC.visibility_of_element_located((By.XPATH, '//a[text()="Player Statistics"]'))
        #     )
        #     # Una vez visible, puedes hacer clic o interactuar con el elemento
        #     elemento.click()
        #     print("Elemento encontrado: //a[text()=\"Player Statistics\"] y clic realizado.")
        # except TimeoutException:
        #     print("Tiempo de espera agotado. El elemento no se encontró.")

        # elemento = driver.find_element(By.XPATH,f'//a[text()="Player Statistics"]')
        # elemento.click()
        # time.sleep(1)

        # [RECORRER POR TABLAS (offensive,xG,etc)]
        for clave_tablas, nom_tablas in tablas_to_extract.items():
            # Clickar en la tabla que toque
            elemento = driver.find_element(By.XPATH,f'//a[text()="{clave_tablas}"]')
            elemento.click()
            time.sleep(5)

            # Hay resultados?
            e = driver.find_elements(By.XPATH,f'//div[@id="{nom_tablas}"]//td[text()="There are no results to display"]')
            if len(e)>0:
                results = False
                e=[]
            else:
                results = True

            # Descargar toda la info
            if results:
                # Click en All players
                elemento = driver.find_element(By.XPATH,f'//div[@id="stage-top-player-stats-{clave_tablas.lower()}"]//a[text()="All players"]')
                elemento.click()
                # elementos = driver.find_elementks(By.XPATH, f'//div[@id="stage-top-player-stats-{clave.lower()}"]//a[text()="All players"]')
                # for elemento in elementos:
                #     try:
                #         # Haces click en el que esté disponible del All players
                #         elemento = driver.find_element(By.XPATH,elemento)
                #         elemento.click()

                #         # # Espera hasta que el elemento sea clickeable
                #         # WebDriverWait(driver, 2).until(EC.element_to_be_clickable(elemento))
                #         # # Si el elemento es clickeable, se hace clic en él
                #         # elemento.click()
                #         # print("Elemento clickeado")
                #         break  # Si se hace clic, salimos del bucle
                #     except:
                #         print("Elemento no clickeable, intentando con el siguiente")
            


                time.sleep(5)
                # Muevete al header
                print(f'//div[@id="{nom_tablas}"]//thead[@id="player-table-statistics-head"]//th')
                elemento = driver.find_element(By.XPATH, f'//div[@id="{nom_tablas}"]//thead[@id="player-table-statistics-head"]//th')
                driver.execute_script("arguments[0].scrollIntoView();", elemento)
                
                # Extrae los encabezados de las columnas de thead
                encabezados = driver.find_elements(By.XPATH, f'//div[@id="{nom_tablas}"]//thead[@id="player-table-statistics-head"]//th')
                nombres_columnas = [encabezado.text for encabezado in encabezados]

                # Inicializa una lista para almacenar todos los datos
                todos_los_datos = []

                # Descargar filas mientras el botón "next" esté presente
                while True:

                    # Localiza la tabla y extrae los datos
                    tabla_id = nom_tablas
                    tabla = driver.find_element(By.ID, tabla_id)
                    filas = tabla.find_elements(By.TAG_NAME, "tr")

                    # Descargar cada fila por jugador
                    for fila in filas:
                        celdas = fila.find_elements(By.TAG_NAME, "td")
                        print("desacargando datos")
                        # print(celdas)
                        fila_datos = [celda.text for celda in celdas]
                        if fila_datos:  # Agrega solo filas con datos
                            todos_los_datos.append(fila_datos)
                        # print(todos_los_datos)


                    print("Es el utlimo(Sigue next)?")
                    # Verifica si el botón "next" está presente
                    try:
                        # boton_siguiente = driver.find_element(By.XPATH, '(//a[@class="option  clickable " and @id="next"])[2]')
                        boton_siguiente = driver.find_element(By.XPATH, f'//div[@id="statistics-paging-{clave_tablas.lower()}"]//a[@class="option  clickable " and @id="next"]')
                        
                        # driver.execute_script("arguments[0].scrollIntoView();", boton_siguiente)
                        boton_siguiente.click()
                        time.sleep(2)  # Espera para cargar la siguiente página
                    except NoSuchElementException:
                        # Si el botón no está presente, rompe el bucle
                        # Convierte los datos recopilados en un DataFrame de Pandas con los encabezados de columnas extraídos
                        print("Guardando df")
                        df = pd.DataFrame(todos_los_datos, columns=nombres_columnas)
                        print(df.head(50))
                        # print(df)
                        # nombre_output = f"goles_{año_fin}.csv"
                        print(f"Guardando en: {ligas_sufijo[i]}_{clave_tablas}_{años[i][1]}.csv")
                        df.to_csv(f"{ligas_sufijo[i]}_{clave_tablas}_{años[i][1]}.csv")
                        break




# Obtén la fecha y hora actual en el formato que desees, por ejemplo, "YYYYMMDD_HHMMSS"

# Cierra el navegador
driver.quit()
   
# # Convierte los datos recopilados en un DataFrame de Pandas
# df = pd.DataFrame(todos_los_datos, columns=["Columna 1", "Columna 2"])  # Ajusta los nombres de las columnas
# print(df)

# # Cierra el navegador
# driver.quit()


"""