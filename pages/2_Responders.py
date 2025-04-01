import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.Responders_function import analyze_responders_all_groups, get_table_download_link, get_figure_download_link
import io

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Responders",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("📊 Análisis de Responders")

# Información sobre el análisis de responders en la barra lateral
with st.sidebar:
    st.markdown("## Acerca del Análisis de Responders")
    st.markdown("""
    ### ¿Qué es el análisis de responders?
    
    El análisis de responders evalúa la respuesta individual a intervenciones, 
    reconociendo que diferentes personas pueden responder de manera distinta 
    al mismo tratamiento o programa.
    
    ### Conceptos clave:
    
    - **Error Típico (TE)**: Representa la variabilidad en las mediciones causada 
      por ruido de medición.
    
    - **Smallest Worthwhile Change (SWC)**: El cambio mínimo que se considera 
      relevante en la práctica. Puede basarse en:
      - 0.2 × desviación estándar de valores basales
      - Un valor personalizado determinado por el investigador
    
    - **Responders**: Individuos cuyo cambio debido a la intervención supera el SWC 
      con el nivel de confianza especificado.
    
    - **Intervalos de Confianza**: Rangos que indican la incertidumbre en los 
      cambios observados.
    
    ### Referencias:
    
    Esta aplicación implementa las metodologías descritas en:
    
    1. Swinton et al. (2018). "A Statistical Framework to Interpret Individual 
       Response to Intervention: Paving the Way for Personalized Nutrition and 
       Exercise Prescription"
    
    2. Bonafiglia et al. (2018). "Moving beyond threshold-based dichotomous 
       classification to improve the accuracy in classifying non-responders"
    """)
    
    st.markdown("---")
    st.markdown("Desarrollado con ❤️ usando Streamlit")

# Carga de datos
st.header("1. Cargar datos")

st.markdown("""
Sube un archivo CSV o Excel que contenga:
- Una columna que identifique los grupos (intervención/control)
- Columnas con valores pre y post intervención
""")

uploaded_file = st.file_uploader("Selecciona un archivo CSV o Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Determinar el tipo de archivo y leerlo
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            try:
                df = pd.read_excel(uploaded_file)
            except ImportError:
                st.error("No se puede leer el archivo Excel. Por favor instala la biblioteca 'openpyxl' con el comando: pip install openpyxl")
                st.stop()
        
        st.success(f"✅ Archivo cargado correctamente. Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
        
        # Mostrar los primeros registros del dataframe
        st.subheader("Vista previa de los datos:")
        st.dataframe(df.head())
        
        # Seleccionar columnas para el análisis
        st.header("2. Seleccionar variables")
        
        # Seleccionar la columna que contiene la variable grupo
        group_column = st.selectbox(
            "Selecciona la columna que indica el grupo (intervención/control):",
            df.columns.tolist(),
            help="Esta columna debe contener valores que identifiquen a qué grupo pertenece cada individuo."
        )
        
        # Identificar valores únicos en la columna grupo
        unique_groups = df[group_column].unique().tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Seleccionar el nombre para grupo intervención
            intervention_label = st.selectbox(
                "Selecciona el valor que identifica al grupo de intervención:",
                unique_groups,
                help="El grupo que recibió el tratamiento o intervención."
            )
        
        with col2:
            # Opciones para el grupo control (incluyendo una opción para "Sin grupo control")
            control_options = ["Sin grupo control"] + [x for x in unique_groups if x != intervention_label]
            
            control_select = st.selectbox(
                "Selecciona el valor que identifica al grupo control:",
                control_options,
                help="El grupo que NO recibió la intervención. Si no hay grupo control, selecciona 'Sin grupo control'."
            )
            
            # Establecer el valor del control_label
            control_label = None if control_select == "Sin grupo control" else control_select
        
        # Seleccionar las columnas para valores pre y post
        col1, col2 = st.columns(2)
        
        with col1:
            pre_column = st.selectbox(
                "Selecciona la columna con los valores PRE-intervención:",
                df.columns.tolist(),
                help="Los valores medidos antes de la intervención."
            )
        
        with col2:
            post_columns = [col for col in df.columns.tolist() if col != pre_column]
            post_column = st.selectbox(
                "Selecciona la columna con los valores POST-intervención:",
                post_columns,
                help="Los valores medidos después de la intervención."
            )
        
        # Parámetros para el análisis
        st.header("3. Configurar parámetros del análisis")
        
        # Primera fila: Nombre de variable y dirección
        col1, col2 = st.columns(2)
        
        with col1:
            # Nombre de la variable
            variable_name = st.text_input(
                "Nombre de la variable analizada:",
                value="Variable",
                help="Este nombre aparecerá en las etiquetas del gráfico."
            )
        
        with col2:
            # Dirección del cambio
            direction = st.radio(
                "Dirección esperada del cambio positivo:",
                ["positive", "negative"],
                help="Elige 'positive' si el éxito se refleja en un aumento del valor (ej. fuerza muscular). "
                     "Elige 'negative' si el éxito se refleja en una disminución del valor (ej. tiempo en carrera)."
            )
        
        # Segunda fila: Nivel de confianza y usar porcentaje
        col1, col2 = st.columns(2)
        
        with col1:
            # Nivel de confianza
            confidence_level = st.select_slider(
                "Nivel de confianza para los intervalos:",
                options=[0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95],
                value=0.75,
                help="Determina el ancho de los intervalos de confianza. Valores más altos crean intervalos más amplios. "
                     "El nivel 0.75 (75%) es recomendado por Swinton et al. (2018)."
            )
        
        with col2:
            # Usar cambios porcentuales o absolutos
            use_percent = st.checkbox(
                "Mostrar cambios como porcentaje del valor inicial", 
                value=True,
                help="Si se marca, los cambios se muestran como porcentaje del valor pre-intervención. "
                     "Si no, se muestran en las unidades originales."
            )
        
        # Tercera fila: Método para calcular SWC
        swc_method = st.radio(
            "Método para calcular el Smallest Worthwhile Change (SWC):",
            ["0.2sd", "custom"],
            help="'0.2sd' calcula el SWC como 0.2 veces la desviación estándar de los valores basales (recomendado). "
                 "'custom' permite ingresar un valor personalizado."
        )
        
        # Valor personalizado para SWC si es necesario
        swc_value = None
        if swc_method == "custom":
            swc_value = st.number_input(
                "Valor personalizado para SWC:", 
                value=1.0,
                help="Ingresa el valor mínimo de cambio que consideras relevante en la práctica."
            )
        
        # Botón para ejecutar el análisis
        if st.button("📊 Ejecutar análisis de responders", use_container_width=True):
            try:
                # Ejecutar el análisis con captura de salida
                with st.spinner("Analizando los datos..."):
                    # Redirigir la salida de print a un objeto StringIO
                    stdout_capture = io.StringIO()
                    import sys
                    original_stdout = sys.stdout
                    sys.stdout = stdout_capture
                    
                    # Ejecutar análisis
                    resultados = analyze_responders_all_groups(
                        group=df[group_column],
                        pre=df[pre_column],
                        post=df[post_column],
                        control_label=control_label,
                        confidence_level=confidence_level,
                        swc_method=swc_method,
                        swc_value=swc_value,
                        direction=direction,
                        variable_name=variable_name,
                        use_percent=use_percent,
                        show_confidence_intervals=True
                    )
                    
                    # Restaurar stdout
                    sys.stdout = original_stdout
                    
                    # Mostrar los mensajes capturados en un área de información
                    logs = stdout_capture.getvalue()
                    if logs.strip():
                        with st.expander("Detalles del procesamiento", expanded=False):
                            st.text(logs)
                
                st.success("✅ Análisis completado correctamente")
                
                # Mostrar resultados para cada grupo
                st.header("4. Resultados")
                
                # Crear pestañas para mostrar los resultados de cada grupo
                if resultados['resultados_por_grupo']:
                    tabs = st.tabs([str(key) for key in resultados['resultados_por_grupo'].keys()])
                    
                    for i, (grupo_clave, tab) in enumerate(zip(resultados['resultados_por_grupo'].keys(), tabs)):
                        with tab:
                            grupo_resultados = resultados['resultados_por_grupo'][grupo_clave]
                            fig = resultados['figuras'][grupo_clave]
                            
                            # Mostrar estadísticas del análisis
                            st.subheader(f"Estadísticas para el grupo: {grupo_clave}")
                            
                            # Crear columnas para los resultados
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Error Típico (TE)", f"{grupo_resultados['error_tipico']:.2f}")
                                
                            with col2:
                                st.metric("SWC", f"{grupo_resultados['swc']:.2f}")
                                
                            with col3:
                                st.metric("Proporción de responders", f"{grupo_resultados['proporcion_responders']:.2f} ({grupo_resultados['proporcion_responders']*100:.1f}%)")
                            
                            # Mostrar clasificación de participantes
                            st.subheader("Clasificación de participantes")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Responders", f"{grupo_resultados['n_responders']}/{grupo_resultados['n_responders'] + grupo_resultados['n_non_responders']} ({grupo_resultados['proporcion_responders']*100:.1f}%)")
                                
                            with col2:
                                st.metric("Non-Responders", f"{grupo_resultados['n_non_responders']}/{grupo_resultados['n_responders'] + grupo_resultados['n_non_responders']} ({(1-grupo_resultados['proporcion_responders'])*100:.1f}%)")
                            
                            # Mostrar el gráfico
                            st.subheader("Gráfico de responders")
                            st.pyplot(fig)
                            
                            # Enlace para descargar el gráfico
                            st.markdown(get_figure_download_link(fig, f"responders_{variable_name}_{grupo_clave}.png", "📥 Descargar gráfico"), unsafe_allow_html=True)
                            
                            # Mostrar la tabla con los resultados individuales
                            st.subheader("Resultados individuales detallados")
                            st.dataframe(grupo_resultados['resultados_individuales'])
                            
                            # Enlace para descargar los resultados
                            st.markdown(get_table_download_link(grupo_resultados['resultados_individuales'], f"resultados_{variable_name}_{grupo_clave}.csv", "📥 Descargar resultados en CSV"), unsafe_allow_html=True)
                else:
                    st.warning("No se encontraron grupos para analizar. Verifica los parámetros seleccionados.")
            
            except Exception as e:
                st.error(f"Error al ejecutar el análisis: {str(e)}")
                st.exception(e)
    
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        st.exception(e)
else:
    # Mostrar ejemplo e instrucciones cuando no hay archivo cargado
    st.info("""
    ### Instrucciones:
    1. Sube un archivo CSV o Excel con tus datos
    2. Selecciona las columnas que contienen los grupos, valores pre y post intervención
    3. Configura los parámetros del análisis
    4. Ejecuta el análisis y visualiza los resultados
    
    ### Formato de datos recomendado:
    Tu archivo debe tener un formato similar a:
    
    | ID | Grupo | Pre | Post |
    |----|-------|-----|------|
    | 1  | intervention | 10.5 | 12.6 |
    | 2  | intervention | 11.2 | 13.8 |
    | 3  | control | 10.8 | 11.0 |
    | 4  | control | 11.5 | 11.8 |
    
    Donde:
    - **Grupo**: Identifica a qué grupo pertenece cada participante (intervención o control)
    - **Pre**: Mediciones antes de la intervención
    - **Post**: Mediciones después de la intervención
    """)

    # Ejemplo visual
    st.subheader("Ejemplo de análisis de responders:")
    
    # Crear imagen explicativa
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as mpatches
    
    # Datos de ejemplo
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(10)
    responder_values = [3.5, 2.1, 0.7, 0.3, -0.3, -0.3, -1.2, -1.6, -2.6, -3.0]
    responder_colors = ['black', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white']
    
    # Barras
    bars = ax.bar(x, responder_values, color=responder_colors, edgecolor='black')
    
    # Línea SWC
    ax.axhline(y=1.0, color='blue', linestyle='-', linewidth=1.5)
    
    # Añadir barras de error
    ci_width = 1.5
    for i, val in enumerate(responder_values):
        if val >= 0:
            y_pos = val + 0.2
        else:
            y_pos = val - 0.2
        ax.text(i, y_pos, f"{val}", ha='center', va='center', fontsize=8)
        ax.errorbar(i, val, yerr=ci_width, fmt='none', ecolor='red', capsize=3)
    
    # Leyenda y etiquetas
    responder_patch = mpatches.Patch(color='black', label='Responders')
    non_responder_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='Non-Responders')
    ci_patch = mpatches.Patch(color='red', alpha=0.3, label='75% CI')
    ax.legend(handles=[responder_patch, non_responder_patch, ci_patch], loc='upper left')
    
    ax.set_xlabel('Participants')
    ax.set_ylabel('Changes in Variable (%)')
    ax.set_title('Responders: 1/10 (10.0%), Non-Responders: 9/10 (90.0%)')
    ax.set_xticks([])
    
    st.pyplot(fig)
    
    st.markdown("""
    ### Interpretación del ejemplo:
    
    - **Barras negras**: Representan a los *responders* (participantes que respondieron a la intervención)
    - **Barras blancas**: Representan a los *non-responders* (participantes que no respondieron)
    - **Línea azul**: Muestra el *Smallest Worthwhile Change* (SWC) - el cambio mínimo considerado relevante
    - **Barras de error rojas**: Muestran el intervalo de confianza para cada cambio
    
    Un participante se considera **responder** cuando el límite inferior de su intervalo de confianza supera el SWC.
    """)

        