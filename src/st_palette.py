import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


df_fifa = pd.read_csv("../data/players_22.csv")
df_mostrar = df_fifa.loc[:,["short_name","overall","value_eur","pace","shooting"]]
df_mostrar = df_mostrar.iloc[0:30,:]
img = Image.open("octopus_fondo.png")
st.set_page_config(page_title="OctoBet/Paul_AI", page_icon="octopus.png",initial_sidebar_state="collapsed")


def main():
    st.sidebar.header("Navegación")
    st.image(img,use_container_width=False)
    st.markdown('# [OctoBet]')
    # st.title("Prueba Titulo")
    
    # st.header("Prueba Titulo")
    # st.subheader("Prueba Titulo")
    # st.text("Esto es un texto")
    # st.success("EXITO!")
    # st.warning("Cuidado!")
    # st.info("Esto es info")
    # st.error("Esto es un error")
    # st.exception("Esto es una excepcion")
    # st.write(3+2)
    # st.write('### Tambien markdown')
    # st.write('Escribir tb con write')
    # st.markdown('## DataFrame Fifa raw:')
    st.dataframe(df_fifa)
    # st.markdown('## DataFrame Fifa to be:')
    # st.dataframe(df_mostrar.style.highlight_max(axis=0))
    # st.table(df_mostrar)

    # codigo = """print("Hola mundo")"""
    # st.code(codigo,language="python")

    # # Selectbox
    # opc = st.selectbox(
    #     'Elige el mejor jugador',
    #     ["Elige una opc","Cr7","Messi", "Halland","Mbappe"]
    # )
    # if opc != "Elige una opc":
    #     if opc != 'Cr7':
    #         st.error(f'Te has equivocado, tu opcion es: {opc}, pero el mejor es Cr7')
    #     else: 
    #         st.success(f"Correcto el mejor es: {opc}")
    
    # # multiselect
    # opcs = st.multiselect(
    #     'Elige el mejor jugador',
    #     ["Cr7","Messi", "Halland","Mbappe"]
    # )
    # st.write(f"tus opciones son:",opcs)

    # # slider
    # overall_elegida = st.slider(
    #     'Selecciona el overal',
    #     min_value=40,
    #     max_value=99,
    #     value=50,
    #     step=1
    # )
    # st.write(f"tus opciones son:",overall_elegida)

    # # select_slider
    # opc_elegida = st.select_slider(
    #     'Selecciona el overal',
    #     options=["Portero","Defensa","Mediocentro", "Delantero"],
    #     value='Portero')
    
    # st.write(f"Tu opc elegida es: {opc_elegida}")

    nombre = st.text_input("Dime tu nombre:")
    st.write(nombre)

    # mensaje = st.text_area("Pon un mensaje")
    # st.write(mensaje)

    numero = st.number_input("ingresa un numero",1,99,1)
    st.write(numero)


    # cita = st.date_input("Selecciona una fecha")
    # st.write(cita)
    # hora = st.time_input("Selecciona una hora")
    # st.write(hora)
    px.density_heatmap(df_fifa)
    

    

    
    




    

if __name__== '__main__':
    main()
