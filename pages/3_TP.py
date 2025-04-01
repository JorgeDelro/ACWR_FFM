"""
Training Peaks Simulator
-----------------------
Esta aplicación simula las funcionalidades principales de TrainingPeaks para análisis
y seguimiento del entrenamiento deportivo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import io
import os
import base64
import sys

# Agregar la ruta de utils al path
sys.path.append("./utils")

# Importar funciones desde TrainingPeaks_functions.py
import utils.TrainingPeaks_functions as tp

# Configurar la página
st.set_page_config(
    page_title="Training Periodization Simulator",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Funciones de la interfaz

def display_header():
    """Muestra el encabezado de la aplicación"""
    col1, col2 = st.columns([1, 4])
    with col1:
        st.title("Training Periodization Simulator")
    with col2:
        st.markdown("Análisis y seguimiento de entrenamiento deportivo")
    
    st.divider()

def sidebar_navigation():
    """Navegación lateral y configuración de usuario"""
    with st.sidebar:
        st.header("Navegación")
        
        page = st.radio(
            "Selecciona una sección:",
            [
                "🏠 Dashboard",
                "📊 Análisis de Métricas",
                "📝 Registro de Actividad",
                "📈 PMC (Performance Management Chart)",
                "⚙️ Configuración",
                "📁 Gestión de Datos"
            ]
        )
        
        st.divider()
        
        # Recuperar o configurar datos del atleta
        st.subheader("Datos del Atleta")
        
        if 'athlete_data' not in st.session_state:
            st.session_state.athlete_data = {
                'name': 'Usuario',
                'ftp': 250,
                'weight': 70.0,
                'threshold_hr': 160,
                'max_hr': 190,
                'rest_hr': 60,
                'threshold_pace': 240  # 4:00 min/km
            }
        
        st.session_state.athlete_data['name'] = st.text_input("Nombre", value=st.session_state.athlete_data['name'])
        st.session_state.athlete_data['ftp'] = st.number_input("FTP (watts)", value=st.session_state.athlete_data['ftp'], min_value=50, max_value=500)
        st.session_state.athlete_data['weight'] = st.number_input("Peso (kg)", value=st.session_state.athlete_data['weight'], min_value=30.0, max_value=150.0, step=0.1)
        st.session_state.athlete_data['threshold_hr'] = st.number_input("FC Umbral", value=st.session_state.athlete_data['threshold_hr'], min_value=100, max_value=200)
        st.session_state.athlete_data['max_hr'] = st.number_input("FC Máxima", value=st.session_state.athlete_data['max_hr'], min_value=120, max_value=220)
        st.session_state.athlete_data['rest_hr'] = st.number_input("FC Reposo", value=st.session_state.athlete_data['rest_hr'], min_value=30, max_value=100)
        
        pace_str = tp.format_pace(st.session_state.athlete_data['threshold_pace'])
        threshold_pace_input = st.text_input("Ritmo Umbral (min:seg/km)", value=pace_str)
        st.session_state.athlete_data['threshold_pace'] = tp.pace_to_seconds(threshold_pace_input)
        
        return page

def load_or_generate_data():
    """Carga datos existentes o genera datos de muestra"""
    if 'activities_df' not in st.session_state:
        try:
            # Intentar cargar datos existentes
            if os.path.exists('training_data.csv'):
                df = pd.read_csv('training_data.csv')
                # Asegurarse de que la columna de fecha es datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                st.session_state.activities_df = df
                st.success("Datos cargados correctamente.")
            else:
                # Generar datos de muestra
                st.session_state.activities_df = pd.DataFrame()
        except Exception as e:
            st.error(f"Error al cargar datos: {e}")
            st.session_state.activities_df = pd.DataFrame()

def dashboard_page():
    """Página principal del dashboard"""
    st.header("Dashboard")
    
    # Mostrar métricas clave
    if not st.session_state.activities_df.empty:
        # Calcular datos PMC
        pmc_data = tp.calculate_pmc_data(st.session_state.activities_df)
        
        if not pmc_data.empty:
            # Obtener últimos valores
            latest_data = pmc_data.iloc[-1]
            
            # Métricas clave en cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Fitness (CTL)",
                    f"{latest_data['CTL']:.1f}",
                    delta=f"{latest_data['CTL'] - pmc_data.iloc[-8]['CTL']:.1f} (7 días)",
                )
            
            with col2:
                st.metric(
                    "Fatiga (ATL)",
                    f"{latest_data['ATL']:.1f}",
                    delta=f"{latest_data['ATL'] - pmc_data.iloc[-8]['ATL']:.1f} (7 días)",
                )
            
            with col3:
                st.metric(
                    "Forma (TSB)",
                    f"{latest_data['TSB']:.1f}",
                    delta=f"{latest_data['TSB'] - pmc_data.iloc[-8]['TSB']:.1f} (7 días)",
                )
            
            with col4:
                # Calcular TSS semanal
                today = pd.Timestamp(dt.datetime.now().date())
                seven_days_ago = today - pd.Timedelta(days=7)
                last_7_days = st.session_state.activities_df[st.session_state.activities_df['date'] >= seven_days_ago]
                weekly_tss = last_7_days['TSS'].sum() if not last_7_days.empty else 0
                
                # Comparar con la semana anterior
                today = pd.Timestamp(dt.datetime.now().date())
                fourteen_days_ago = today - pd.Timedelta(days=14)
                seven_days_ago = today - pd.Timedelta(days=7)

                previous_week = st.session_state.activities_df[
                    (st.session_state.activities_df['date'] >= fourteen_days_ago) &
                    (st.session_state.activities_df['date'] < seven_days_ago)
                ]
                previous_weekly_tss = previous_week['TSS'].sum() if not previous_week.empty else 0
                
                st.metric(
                    "TSS Semanal",
                    f"{weekly_tss:.0f}",
                    delta=f"{weekly_tss - previous_weekly_tss:.0f} vs anterior",
                )
        
        # Gráficos de resumen
        st.subheader("Resumen de entrenamiento")
        
        tab1, tab2, tab3 = st.tabs(["PMC", "Distribución Semanal", "Distribución por Actividad"])
        
        with tab1:
            if not pmc_data.empty:
                fig = tp.plot_pmc(pmc_data)
                st.pyplot(fig)
            else:
                st.info("No hay suficientes datos para mostrar el PMC.")
        
        with tab2:
            fig = tp.plot_weekly_summary(st.session_state.activities_df)
            st.pyplot(fig)
        
        with tab3:
            fig = tp.plot_activity_distribution(st.session_state.activities_df)
            st.pyplot(fig)
        
        # Actividades recientes
        st.subheader("Actividades recientes")
        
        recent_activities = st.session_state.activities_df.sort_values('date', ascending=False).head(5)
        
        if not recent_activities.empty:
            for idx, activity in recent_activities.iterrows():
                with st.expander(f"{activity['title']} - {activity['date'].strftime('%d/%m/%Y')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Tipo:** {activity['activity_type'].capitalize()}")
                        st.write(f"**Duración:** {dt.timedelta(seconds=int(activity['duration_seconds']))}")
                        
                        if activity['activity_type'] == 'cycling':
                            st.write(f"**Potencia media:** {activity['avg_power']:.0f} W")
                            st.write(f"**NP:** {activity.get('np', 0):.0f} W")
                            st.write(f"**IF:** {activity.get('intensity_factor', 0):.2f}")
                        
                        elif activity['activity_type'] == 'running':
                            pace_str = tp.format_pace(activity['avg_pace'])
                            st.write(f"**Ritmo medio:** {pace_str}")
                        
                        st.write(f"**FC media:** {activity['avg_hr']:.0f} ppm")
                    
                    with col2:
                        st.write(f"**TSS:** {activity['TSS']:.0f}")
                        
                        # Calcular contribución a CTL, ATL
                        ftp = st.session_state.athlete_data['ftp']
                        ctl_contribution = activity['TSS'] / tp.DEFAULT_CTL_DAYS
                        atl_contribution = activity['TSS'] / tp.DEFAULT_ATL_DAYS
                        
                        st.write(f"**Contribución a CTL:** +{ctl_contribution:.2f}")
                        st.write(f"**Contribución a ATL:** +{atl_contribution:.2f}")
                        st.write(f"**Impacto en TSB:** -{atl_contribution - ctl_contribution:.2f}")
        else:
            st.info("No hay actividades registradas.")
    else:
        st.info("No hay datos de actividades. Registra una actividad o genera datos de muestra en la sección 'Gestión de Datos'.")

def activity_log_page():
    """Página para registrar nuevas actividades"""
    st.header("Registro de Actividad")
    
    # Formulario para añadir nueva actividad
    with st.form("activity_form"):
        st.subheader("Nueva Actividad")
        
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Título", "Entrenamiento")
            date = st.date_input("Fecha", dt.datetime.now().date())
            activity_type = st.selectbox(
                "Tipo de Actividad",
                ["cycling", "running", "swimming"]
            )
            
            duration_str = st.text_input("Duración (hh:mm:ss)", "01:00:00")
            
            # Convertir duración a segundos
            try:
                h, m, s = map(int, duration_str.split(':'))
                duration_seconds = h * 3600 + m * 60 + s
            except:
                duration_seconds = 3600  # 1 hora por defecto
        
        with col2:
            # Campos específicos según el tipo de actividad
            if activity_type == "cycling":
                avg_power = st.number_input("Potencia Media (W)", value=200, min_value=0, max_value=1000)
                np_value = st.number_input("Normalized Power (W)", value=220, min_value=0, max_value=1000)
                intensity_factor = st.number_input("Intensity Factor", value=0.8, min_value=0.0, max_value=2.0, step=0.01)
                ftp = st.session_state.athlete_data['ftp']
                
                # Calcular TSS
                tss = tp.calculate_tss_from_power(duration_seconds, np_value, intensity_factor, ftp)
                
                st.metric("TSS Estimado", f"{tss:.0f}")
            
            elif activity_type == "running":
                pace_str = st.text_input("Ritmo Medio (min:seg/km)", "5:30")
                avg_pace = tp.pace_to_seconds(pace_str)
                
                threshold_pace = st.session_state.athlete_data['threshold_pace']
                grade_factor = st.number_input("Factor Desnivel", value=1.0, min_value=0.9, max_value=1.5, step=0.01)
                
                # Calcular rTSS
                rtss = tp.calculate_running_tss(duration_seconds/3600, avg_pace, threshold_pace, grade_factor)
                
                st.metric("TSS Estimado", f"{rtss:.0f}")
            
            elif activity_type == "swimming":
                intensity = st.number_input("Intensidad", value=1.0, min_value=0.5, max_value=1.5, step=0.01)
                
                # Calcular sTSS
                stss = tp.calculate_swimming_tss(duration_seconds/3600, intensity)
                
                st.metric("TSS Estimado", f"{stss:.0f}")
            
            # Frecuencia cardíaca (común a todas las actividades)
            avg_hr = st.number_input("FC Media", value=150, min_value=0, max_value=220)
        
        # Botón para enviar
        submitted = st.form_submit_button("Guardar Actividad")
        
        if submitted:
            # Crear datos de actividad
            new_activity = {
                'date': pd.to_datetime(date),
                'title': title,
                'activity_type': activity_type,
                'duration_seconds': duration_seconds,
                'avg_hr': avg_hr
            }
            
            # Añadir datos específicos según tipo
            if activity_type == "cycling":
                new_activity.update({
                    'avg_power': avg_power,
                    'np': np_value,
                    'intensity_factor': intensity_factor,
                    'ftp': st.session_state.athlete_data['ftp'],
                    'TSS': tss
                })
            elif activity_type == "running":
                new_activity.update({
                    'avg_pace': avg_pace,
                    'threshold_pace': threshold_pace,
                    'grade_factor': grade_factor,
                    'TSS': rtss
                })
            elif activity_type == "swimming":
                new_activity.update({
                    'intensity': intensity,
                    'TSS': stss
                })
            
            # Añadir a DataFrame de actividades
            if 'activities_df' not in st.session_state or st.session_state.activities_df.empty:
                st.session_state.activities_df = pd.DataFrame([new_activity])
            else:
                st.session_state.activities_df = pd.concat([
                    st.session_state.activities_df,
                    pd.DataFrame([new_activity])
                ], ignore_index=True)
            
            # Guardar en CSV
            try:
                st.session_state.activities_df.to_csv('training_data.csv', index=False)
                st.success("Actividad guardada correctamente.")
            except Exception as e:
                st.error(f"Error al guardar los datos: {e}")
    
    # Mostrar actividades recientes
    if 'activities_df' in st.session_state and not st.session_state.activities_df.empty:
        st.subheader("Actividades Recientes")
        
        recent = st.session_state.activities_df.sort_values('date', ascending=False).head(5)
        
        # Formatear DataFrame para visualización
        display_df = recent[['date', 'title', 'activity_type', 'duration_seconds', 'TSS']].copy()
        display_df['Duración'] = display_df['duration_seconds'].apply(
            lambda x: str(dt.timedelta(seconds=int(x)))
        )
        display_df = display_df.drop('duration_seconds', axis=1)
        
        # Renombrar columnas
        display_df = display_df.rename(columns={
            'date': 'Fecha',
            'title': 'Título',
            'activity_type': 'Tipo',
            'TSS': 'TSS'
        })
        
        st.dataframe(display_df)
    else:
        st.info("No hay actividades registradas.")

def pmc_page():
    """Página del Performance Management Chart (PMC)"""
    st.header("Performance Management Chart (PMC)")
    
    if 'activities_df' in st.session_state and not st.session_state.activities_df.empty:
        # Opciones de visualización
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Seleccionar rango de fechas
            date_range = st.date_input(
                "Rango de fechas:",
                [
                    st.session_state.activities_df['date'].min().date(),
                    st.session_state.activities_df['date'].max().date()
                ]
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                
                # Filtrar datos por fecha
                filtered_df = st.session_state.activities_df[
                    (st.session_state.activities_df['date'].dt.date >= start_date) &
                    (st.session_state.activities_df['date'].dt.date <= end_date)
                ]
        
        with col2:
            # Ajustes de constantes PMC
            ctl_days = st.number_input(
                "Días CTL (Fitness)",
                value=tp.DEFAULT_CTL_DAYS,
                min_value=14,
                max_value=56,
                step=1
            )
            
            atl_days = st.number_input(
                "Días ATL (Fatiga)",
                value=tp.DEFAULT_ATL_DAYS,
                min_value=3,
                max_value=14,
                step=1
            )
        
        if not filtered_df.empty:
            # Calcular datos PMC
            pmc_data = tp.calculate_pmc_data(filtered_df)
            
            if not pmc_data.empty:
                # Explicación de las métricas
                with st.expander("Explicación de métricas PMC"):
                    st.markdown("""
                    ### Métricas del Performance Management Chart
                    
                    - **CTL (Chronic Training Load)**: Representa tu "fitness" o condición física acumulada. Es la carga de entrenamiento promedio durante un período prolongado (por defecto 42 días).
                    
                    - **ATL (Acute Training Load)**: Representa tu "fatiga" reciente. Es la carga de entrenamiento promedio en un período corto (por defecto 7 días).
                    
                    - **TSB (Training Stress Balance)**: También conocido como "forma". Es la diferencia entre CTL y ATL (CTL - ATL).
                        - TSB positivo (+): Buena forma, descansado
                        - TSB negativo (-): Fatiga acumulada
                    
                    - **TSS (Training Stress Score)**: Cuantifica la carga de entrenamiento de una sesión individual.
                    
                    ### Interpretación
                    
                    - **Fitness (CTL)**: Una línea ascendente indica mejora en la condición física.
                    - **Fatiga (ATL)**: Aumenta rápidamente con entrenamientos intensos.
                    - **Forma (TSB)**: Es óptima para competir cuando es ligeramente positiva tras un período de alta carga.
                    """)
                
                # Mostrar métricas actuales
                latest_data = pmc_data.iloc[-1]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Fitness (CTL)",
                        f"{latest_data['CTL']:.1f}",
                        delta=f"{latest_data['CTL'] - pmc_data.iloc[-8]['CTL']:.1f} (7 días)",
                    )
                
                with col2:
                    st.metric(
                        "Fatiga (ATL)",
                        f"{latest_data['ATL']:.1f}",
                        delta=f"{latest_data['ATL'] - pmc_data.iloc[-8]['ATL']:.1f} (7 días)",
                    )
                
                with col3:
                    st.metric(
                        "Forma (TSB)",
                        f"{latest_data['TSB']:.1f}",
                        delta=f"{latest_data['TSB'] - pmc_data.iloc[-8]['TSB']:.1f} (7 días)",
                    )
                
                # Gráfico PMC
                fig = tp.plot_pmc(pmc_data)
                st.pyplot(fig)
                
                # Tabla de datos PMC
                with st.expander("Ver datos detallados"):
                    # Crear tabla con datos diarios
                    table_data = pmc_data[['date', 'TSS', 'CTL', 'ATL', 'TSB']].copy()
                    table_data['date'] = table_data['date'].dt.date
                    
                    # Renombrar columnas
                    table_data = table_data.rename(columns={
                        'date': 'Fecha',
                        'TSS': 'TSS',
                        'CTL': 'Fitness',
                        'ATL': 'Fatiga',
                        'TSB': 'Forma'
                    })
                    
                    st.dataframe(table_data.sort_values('Fecha', ascending=False))
                
                # Predicción de PMC
                st.subheader("Predicción de PMC")
                st.write("Simula cómo evolucionarán tus métricas de PMC en base a entrenamientos futuros.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    pred_days = st.slider("Días a predecir", 7, 28, 14)
                
                with col2:
                    weekly_tss = st.number_input(
                        "TSS semanal estimado",
                        value=int(filtered_df['TSS'].sum() / ((end_date - start_date).days / 7)),
                        min_value=0,
                        max_value=1000
                    )
                
                # Extender datos PMC con predicción
                last_date = pmc_data['date'].max()
                last_ctl = pmc_data['CTL'].iloc[-1]
                last_atl = pmc_data['ATL'].iloc[-1]
                
                # Crear fechas futuras
                future_dates = pd.date_range(start=last_date + dt.timedelta(days=1), periods=pred_days)
                
                # Distribuir TSS diario (patrón: descanso lunes, más el finde)
                daily_pattern = [0.7, 1.0, 1.2, 0.8, 0.5, 1.4, 1.9]  # Lun a Dom
                daily_tss = []
                
                for date in future_dates:
                    day_of_week = date.weekday()
                    day_tss = (weekly_tss / 7) * daily_pattern[day_of_week]
                    daily_tss.append(day_tss)
                
                # Calcular CTL, ATL, TSB futuros
                future_ctl = [last_ctl]
                future_atl = [last_atl]
                future_tsb = [last_ctl - last_atl]
                
                for tss in daily_tss:
                    next_ctl = future_ctl[-1] + (tss - future_ctl[-1]) / ctl_days
                    next_atl = future_atl[-1] + (tss - future_atl[-1]) / atl_days
                    next_tsb = next_ctl - next_atl
                    
                    future_ctl.append(next_ctl)
                    future_atl.append(next_atl)
                    future_tsb.append(next_tsb)
                
                # Quitar el primer valor (es el último de los datos reales)
                future_ctl = future_ctl[1:]
                future_atl = future_atl[1:]
                future_tsb = future_tsb[1:]
                
                # Crear DataFrame con predicción
                future_pmc = pd.DataFrame({
                    'date': future_dates,
                    'TSS': daily_tss,
                    'CTL': future_ctl,
                    'ATL': future_atl,
                    'TSB': future_tsb
                })
                
                # Combinar datos reales y predicción
                combined_pmc = pd.concat([pmc_data, future_pmc])
                
                # Crear gráfico con predicción
                fig, ax1 = plt.subplots(figsize=(12, 8))
                
                # Gráfico de CTL y ATL
                ax1.set_xlabel('Fecha')
                ax1.set_ylabel('CTL / ATL', color='black')
                
                # Datos reales
                ax1.plot(pmc_data['date'], pmc_data['CTL'], color='blue', label='Fitness (CTL)')
                ax1.plot(pmc_data['date'], pmc_data['ATL'], color='red', label='Fatiga (ATL)')
                
                # Predicción (líneas punteadas)
                ax1.plot(future_pmc['date'], future_pmc['CTL'], color='blue', linestyle='--', label='Predicción CTL')
                ax1.plot(future_pmc['date'], future_pmc['ATL'], color='red', linestyle='--', label='Predicción ATL')
                
                ax1.grid(True, linestyle='--', alpha=0.7)
                
                # Segundo eje para TSB
                ax2 = ax1.twinx()
                ax2.set_ylabel('TSB (Forma)', color='green')
                
                # Datos reales de TSB
                ax2.plot(pmc_data['date'], pmc_data['TSB'], color='green', label='Forma (TSB)')
                
                # Predicción de TSB
                ax2.plot(future_pmc['date'], future_pmc['TSB'], color='green', linestyle='--', label='Predicción TSB')
                
                # Rellenar áreas para TSB
                ax2.fill_between(
                    combined_pmc['date'], 
                    combined_pmc['TSB'], 
                    0,
                    where=combined_pmc['TSB'] >= 0, 
                    color='lightgreen', 
                    alpha=0.5
                )
                ax2.fill_between(
                    combined_pmc['date'], 
                    combined_pmc['TSB'], 
                    0,
                    where=combined_pmc['TSB'] < 0, 
                    color='lightcoral', 
                    alpha=0.5
                )
                
                # Línea vertical para separar datos reales de predicción
                ax1.axvline(x=last_date, color='gray', linestyle='-.')
                
                # Combinar leyendas
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                # Formatear fechas
                fig.autofmt_xdate()
                
                plt.title('Performance Management Chart (PMC) con Predicción')
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Recomendaciones basadas en TSB actual
                st.subheader("Recomendaciones para la planificación")
                
                current_tsb = latest_data['TSB']
                tsb_state = ""
                
                if current_tsb <= -30:
                    tsb_state = "Sobrecarga extrema"
                    recommendation = """
                    **Estado**: Sobrecarga extrema (TSB ≤ -30)
                    
                    **Recomendaciones**:
                    - Necesitas descanso urgente para evitar sobreentrenamiento.
                    - Programa 2-4 días de descanso o actividad muy ligera.
                    - Enfócate en la recuperación (nutrición, sueño, hidratación).
                    - No planifiques competiciones importantes en las próximas 2 semanas.
                    """
                elif current_tsb <= -15:
                    tsb_state = "Fatiga significativa"
                    recommendation = """
                    **Estado**: Fatiga significativa (-30 < TSB ≤ -15)
                    
                    **Recomendaciones**:
                    - Estás en fase de sobrecarga, que puede ser productiva si es controlada.
                    - Reduce el volumen en un 30-50% durante 3-5 días.
                    - Mantén algo de intensidad para no perder adaptaciones.
                    - Ideal para bloques de construcción, no para competir.
                    """
                elif current_tsb <= 0:
                    tsb_state = "Carga de entrenamiento"
                    recommendation = """
                    **Estado**: Carga de entrenamiento normal (-15 < TSB ≤ 0)
                    
                    **Recomendaciones**:
                    - Balance adecuado entre carga y recuperación.
                    - Puedes mantener este nivel para desarrollo de fitness a largo plazo.
                    - Apto para competiciones de menor importancia.
                    - Alterna días más duros con días más suaves.
                    """
                elif current_tsb <= 15:
                    tsb_state = "Forma óptima"
                    recommendation = """
                    **Estado**: Forma óptima (0 < TSB ≤ 15)
                    
                    **Recomendaciones**:
                    - Zona ideal para competiciones importantes.
                    - Mantén un volumen moderado con algunas sesiones de calidad.
                    - Esta forma es sostenible durante 1-2 semanas.
                    - Puedes aumentar ligeramente la intensidad y reducir volumen.
                    """
                elif current_tsb <= 25:
                    tsb_state = "Muy descansado"
                    recommendation = """
                    **Estado**: Muy descansado (15 < TSB ≤ 25)
                    
                    **Recomendaciones**:
                    - Estás muy descansado, pero podrías empezar a perder algo de fitness.
                    - Ideal para competiciones muy importantes o tras un periodo de carga.
                    - Puedes mantener este estado durante aproximadamente una semana.
                    - Incluye entrenamientos de calidad para mantener la forma.
                    """
                else:
                    tsb_state = "Pérdida de fitness"
                    recommendation = """
                    **Estado**: Desentrenamiento potencial (TSB > 25)
                    
                    **Recomendaciones**:
                    - TSB demasiado alto indica pérdida de fitness por descanso excesivo.
                    - Aumenta gradualmente la carga de entrenamiento.
                    - Reintroduce entrenamientos estructurados.
                    - Usa este periodo para trabajar técnica y recuperar de lesiones.
                    """
                
                # Mostrar estado y recomendaciones
                st.info(f"**Estado actual**: {tsb_state} (TSB = {current_tsb:.1f})")
                st.markdown(recommendation)
            else:
                st.info("No hay suficientes datos para generar el PMC.")
        else:
            st.info("No hay datos para el período seleccionado.")
    else:
        st.info("No hay actividades registradas para generar el PMC.")

def configuration_page():
    """Página de configuración avanzada"""
    st.header("Configuración")
    
    # Pestañas para diferentes configuraciones
    tab1, tab2, tab3 = st.tabs([
        "Zonas de Entrenamiento", 
        "Parámetros PMC", 
        "Preferencias de Visualización"
    ])
    
    with tab1:
        st.subheader("Configuración de Zonas de Entrenamiento")
        
        # Seleccionar tipo de zonas a configurar
        zone_type = st.radio(
            "Selecciona tipo de zonas a configurar:",
            ["Potencia", "Frecuencia Cardíaca", "Ritmo"],
            horizontal=True
        )
        
        if zone_type == "Potencia":
            st.write("**Zonas de potencia basadas en FTP**")
            
            # Obtener FTP actual
            current_ftp = st.session_state.athlete_data['ftp']
            
            # Permitir ajustar FTP
            new_ftp = st.number_input(
                "FTP (Functional Threshold Power)", 
                value=current_ftp,
                min_value=50,
                max_value=500
            )
            
            # Mostrar zonas calculadas
            if new_ftp != current_ftp:
                st.session_state.athlete_data['ftp'] = new_ftp
            
            # Generar y mostrar zonas
            power_zones = tp.generate_zones(ftp=new_ftp)['power']
            
            # Crear tabla de zonas
            zone_data = []
            for zone_name, (zone_min, zone_max) in power_zones.items():
                zone_data.append({
                    "Zona": zone_name,
                    "Rango (W)": f"{zone_min} - {zone_max}",
                    "% FTP": f"{int(zone_min/new_ftp*100 if zone_min > 0 else 0)}-{int(zone_max/new_ftp*100)}%"
                })
            
            st.table(pd.DataFrame(zone_data))
            
            # Explicación de zonas
            with st.expander("Explicación de zonas de potencia"):
                st.markdown("""
                ### Zonas de Potencia (Basadas en FTP)
                
                - **Zona 1 - Recuperación (< 55% FTP)**: Esfuerzo muy ligero, usado para recuperación activa. Mejora la capacidad de depurar metabolitos y prepara el cuerpo para entrenamientos futuros.
                
                - **Zona 2 - Resistencia (56-75% FTP)**: Intensidad moderada, sostenible durante horas. Mejora la eficiencia metabólica, la utilización de grasas y la capilarización.
                
                - **Zona 3 - Tempo (76-90% FTP)**: "Ritmo cómodo pero trabajado". Mejora la economía de pedaleo y eleva el umbral de lactato. Intensidad de estado estable superior.
                
                - **Zona 4 - Umbral (91-105% FTP)**: Intensidad cercana al umbral de lactato. Entrenar en esta zona mejora la capacidad para mantener esfuerzos de alta intensidad por períodos prolongados.
                
                - **Zona 5 - VO₂máx (106-120% FTP)**: Desarrolla la potencia aeróbica máxima. Intervalos típicos de 3-8 minutos. Mejora la capacidad cardíaca y la potencia aeróbica.
                
                - **Zona 6 - Capacidad anaeróbica (121-150% FTP)**: Desarrolla la resistencia a la acumulación de lactato. Intervalos típicos de 30 segundos a 3 minutos. Mejora la capacidad tampón.
                
                - **Zona 7 - Potencia neuromuscular (> 150% FTP)**: Esfuerzos máximos muy cortos. Mejora la coordinación neuromuscular y la fuerza máxima específica.
                """)
        
        elif zone_type == "Frecuencia Cardíaca":
            st.write("**Zonas de frecuencia cardíaca basadas en LTHR**")
            
            # Obtener datos actuales
            current_lthr = st.session_state.athlete_data['threshold_hr']
            current_max_hr = st.session_state.athlete_data['max_hr']
            current_rest_hr = st.session_state.athlete_data['rest_hr']
            
            # Permitir ajustar valores
            col1, col2, col3 = st.columns(3)
            
            with col1:
                new_lthr = st.number_input(
                    "LTHR (FC Umbral)", 
                    value=current_lthr,
                    min_value=100,
                    max_value=200
                )
            
            with col2:
                new_max_hr = st.number_input(
                    "FC Máxima", 
                    value=current_max_hr,
                    min_value=120,
                    max_value=220,
                    key="config_max_hr"  # Añade una clave única
                )
            
            with col3:
                new_rest_hr = st.number_input(
                    "FC Reposo", 
                    value=current_rest_hr,
                    min_value=30,
                    max_value=100
                )
            
            # Actualizar valores si cambiaron
            if new_lthr != current_lthr:
                st.session_state.athlete_data['threshold_hr'] = new_lthr
            if new_max_hr != current_max_hr:
                st.session_state.athlete_data['max_hr'] = new_max_hr
            if new_rest_hr != current_rest_hr:
                st.session_state.athlete_data['rest_hr'] = new_rest_hr
            
            # Generar y mostrar zonas
            hr_zones = tp.generate_zones(lthr=new_lthr)['heart_rate']
            
            # Crear tabla de zonas
            zone_data = []
            for zone_name, (zone_min, zone_max) in hr_zones.items():
                zone_data.append({
                    "Zona": zone_name,
                    "Rango (ppm)": f"{zone_min} - {zone_max}",
                    "% LTHR": f"{int(zone_min/new_lthr*100 if zone_min > 0 else 0)}-{int(zone_max/new_lthr*100)}%"
                })
            
            st.table(pd.DataFrame(zone_data))
            
            # Explicación de zonas
            with st.expander("Explicación de zonas de frecuencia cardíaca"):
                st.markdown("""
                ### Zonas de Frecuencia Cardíaca (Basadas en LTHR)
                
                - **Zona 1 - Recuperación (< 81% LTHR)**: Esfuerzo muy ligero, utilizado para recuperación activa. Permite circular sangre oxigenada a los músculos sin generar fatiga adicional.
                
                - **Zona 2 - Resistencia aeróbica (82-88% LTHR)**: Intensidad moderada, sostenible durante períodos prolongados. Mejora la eficiencia cardiovascular y la oxidación de grasas.
                
                - **Zona 3 - Tempo (89-93% LTHR)**: Intensidad "comfortably hard". Mejora la resistencia aeróbica y eficiencia metabólica. Típicamente usado en entrenamientos de tempo.
                
                - **Zona 4 - Umbral de lactato (94-99% LTHR)**: Intensidad cercana o en el umbral de lactato. Entrena la capacidad para procesar y eliminar lactato eficientemente.
                
                - **Zona 5A - VO₂máx (100-102% LTHR)**: Desarrolla el consumo máximo de oxígeno. Intervalos intensos que llevan al sistema cardiovascular cerca de su capacidad máxima.
                
                - **Zona 5B - Capacidad anaeróbica (103-105% LTHR)**: Mejora la tolerancia al lactato y la capacidad tampón. Intervalos de alta intensidad que generan acumulación de lactato.
                
                - **Zona 5C - Potencia anaeróbica (> 106% LTHR)**: Esfuerzos máximos que desarrollan la potencia anaeróbica. La frecuencia cardíaca a menudo no llega a su máximo por la corta duración de estos esfuerzos.
                
                **Nota**: Las zonas basadas en FC tienen cierto retraso respecto al esfuerzo real, especialmente en intervalos cortos. Para esfuerzos anaeróbicos, es preferible usar zonas de potencia.
                """)
        
        elif zone_type == "Ritmo":
            st.write("**Zonas de ritmo basadas en ritmo umbral**")
            
            # Obtener ritmo umbral actual
            current_threshold_pace = st.session_state.athlete_data['threshold_pace']
            current_pace_str = tp.format_pace(current_threshold_pace)
            
            # Permitir ajustar ritmo umbral
            new_pace_str = st.text_input(
                "Ritmo Umbral (min:seg/km)", 
                value=current_pace_str,
                key="config_threshold_pace"  # Añade una clave única
            )
            new_threshold_pace = tp.pace_to_seconds(new_pace_str)
            
            # Actualizar si cambió
            if new_threshold_pace != current_threshold_pace:
                st.session_state.athlete_data['threshold_pace'] = new_threshold_pace
            
            # Generar y mostrar zonas
            pace_zones = tp.generate_zones(threshold_pace=new_threshold_pace)['pace']
            
            # Crear tabla de zonas
            zone_data = []
            for zone_name, (zone_min, zone_max) in pace_zones.items():
                zone_data.append({
                    "Zona": zone_name,
                    "Ritmo": f"{tp.format_pace(zone_min)} - {tp.format_pace(zone_max)}",
                    "% Ritmo Umbral": f"{int(new_threshold_pace/zone_min*100)}-{int(new_threshold_pace/zone_max*100 if zone_max > 0 else 0)}%"
                })
            
            st.table(pd.DataFrame(zone_data))
            
            # Explicación de zonas
            with st.expander("Explicación de zonas de ritmo"):
                st.markdown("""
                ### Zonas de Ritmo (Basadas en Ritmo Umbral)
                
                - **Zona 1 - Recuperación (>120% del ritmo umbral)**: Ritmo muy lento usado para recuperación activa. Mejora la circulación sin añadir estrés significativo.
                
                - **Zona 2 - Resistencia aeróbica (111-119% del ritmo umbral)**: Ritmo fácil, conversacional. Desarrolla eficiencia aeróbica y capacidad para carreras largas.
                
                - **Zona 3 - Tempo (105-110% del ritmo umbral)**: Ritmo "moderadamente duro". Mejora la eficiencia metabólica y eleva gradualmente el umbral de lactato.
                
                - **Zona 4 - Umbral (99-104% del ritmo umbral)**: Ritmo cercano al umbral de lactato, sostenible durante unos 60 minutos en esfuerzo máximo. Mejora la capacidad para mantener ritmos elevados.
                
                - **Zona 5A - VO₂máx (95-98% del ritmo umbral)**: Ritmo para intervalos de 3-5 minutos. Mejora la potencia aeróbica máxima.
                
                - **Zona 5B - Capacidad anaeróbica (90-94% del ritmo umbral)**: Ritmo para intervalos de 1-3 minutos. Mejora la capacidad para tolerar y eliminar lactato.
                
                - **Zona 5C - Velocidad (<90% del ritmo umbral)**: Ritmo muy rápido para sprints y repeticiones cortas. Mejora la mecánica, economía de carrera y velocidad máxima.
                
                **Nota**: A diferencia de potencia y FC, en ritmo los porcentajes son inversos (un ritmo más rápido es un número más bajo de minutos/km).
                """)
    
    with tab2:
        st.subheader("Parámetros del Performance Management Chart (PMC)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Configurar constantes CTL y ATL
            ctl_days = st.number_input(
                "Constante de tiempo CTL (días)",
                value=tp.DEFAULT_CTL_DAYS,
                min_value=14,
                max_value=56,
                step=1,
                help="Número de días utilizados para calcular el CTL (fitness). Valores típicos: 42-56 días."
            )
        
        with col2:
            atl_days = st.number_input(
                "Constante de tiempo ATL (días)",
                value=tp.DEFAULT_ATL_DAYS,
                min_value=3,
                max_value=14,
                step=1,
                help="Número de días utilizados para calcular el ATL (fatiga). Valores típicos: 7-14 días."
            )
        
        # Explicación detallada de constantes y métricas PMC
        with st.expander("Detalles técnicos del PMC"):
            st.markdown("""
            ### Funcionamiento técnico del Performance Management Chart
            
            #### Fórmulas Básicas
            
            **1. CTL (Chronic Training Load)**
            - CTL de hoy = CTL de ayer + (TSS de hoy - CTL de ayer) / constante CTL
            - La constante CTL representa el número de días considerados para el promedio.
            - Valores típicos: 42 días (6 semanas) para ciclismo, hasta 56 días para carrera.
            - Un valor mayor produce cambios más graduales y suaves en el CTL.
            
            **2. ATL (Acute Training Load)**
            - ATL de hoy = ATL de ayer + (TSS de hoy - ATL de ayer) / constante ATL
            - La constante ATL representa el número de días para el promedio de fatiga.
            - Valores típicos: 7 días (1 semana).
            - Un valor menor produce cambios más rápidos en el ATL (respuesta más sensible).
            
            **3. TSB (Training Stress Balance)**
            - TSB = CTL - ATL
            - Representa el balance entre fitness a largo plazo y fatiga reciente.
            
            #### Interpretación Avanzada
            
            **Ratio ATL:CTL**
            - El ratio entre las constantes ATL:CTL (normalmente 1:6) determina la sensibilidad del modelo.
            - Ajustar este ratio afecta cómo responde el modelo a cambios en la carga de entrenamiento.
            
            **TSB y Rendimiento**
            - TSB negativo (-10 a -30): Fase de construcción, acumulando fitness a costa de fatiga.
            - TSB cercano a cero (-5 a +5): Balance neutro, sostenible a largo plazo.
            - TSB positivo (+5 a +25): "Forma", ideal para competiciones importantes.
            - TSB muy alto (>+25): Posible pérdida de fitness por exceso de descanso.
            
            **Rampas de CTL**
            - El incremento semanal de CTL no debería exceder 5-8 puntos para evitar sobreentrenamiento.
            - Incrementos mayores aumentan el riesgo de lesiones y fatiga excesiva.
            
            #### Estrategias de Periodización
            
            **Periodización en Bloques**
            1. Bloque de carga: Incremento deliberado de CTL con TSB negativo
            2. Bloque de recuperación: Reducción de carga permitiendo que ATL baje y TSB se vuelva positivo
            
            **Periodización para Evento**
            1. Fase de construcción: Incremento gradual de CTL durante semanas/meses
            2. Fase de especialidad: Mantener CTL alto con entrenamientos específicos
            3. Taper: Reducción estratégica de volumen manteniendo intensidad para maximizar TSB
            """)
        
        # Guardar configuración modificada
        if st.button("Guardar configuración de PMC"):
            # Actualizar constantes globales (en realidad no modifica las constantes en el módulo,
            # pero podría implementarse un sistema para guardar preferencias)
            st.success("Configuración guardada correctamente.")
    
    with tab3:
        st.subheader("Preferencias de Visualización")
        
        # Opciones de visualización del PMC
        st.write("**Configuración del PMC**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Opciones ficticias (en una implementación real estas serían funcionales)
            pmc_range = st.selectbox(
                "Rango predeterminado del PMC",
                ["Últimos 90 días", "Últimos 180 días", "Año actual", "Todo"]
            )
            
            pmc_scale = st.selectbox(
                "Escala de visualización",
                ["Automática", "0-100", "0-150", "0-200"]
            )
        
        with col2:
            show_tss = st.checkbox("Mostrar TSS en el PMC", value=True)
            logarithmic_tss = st.checkbox("Escala logarítmica para TSS", value=False)

def metrics_analysis_page():
    """Página de análisis detallado de métricas"""
    st.header("Análisis de Métricas")
    
    if not st.session_state.activities_df.empty:
        # Seleccionar tipo de análisis
        analysis_type = st.selectbox(
            "Selecciona tipo de análisis:",
            [
                "Distribución de Zonas",
                "Progresión de métricas",
                "Análisis por tipo de actividad",
                "Comparación de períodos"
            ]
        )
        
        if analysis_type == "Distribución de Zonas":
            zone_type = st.radio(
                "Tipo de zonas:",
                ["Potencia", "Frecuencia Cardíaca", "Ritmo"],
                horizontal=True
            )
            
            ftp = st.session_state.athlete_data['ftp']
            lthr = st.session_state.athlete_data['threshold_hr']
            threshold_pace = st.session_state.athlete_data['threshold_pace']
            
            if zone_type == "Potencia":
                fig = tp.plot_zone_distribution(
                    st.session_state.activities_df, 
                    zone_type='power', 
                    ftp=ftp
                )
                st.pyplot(fig)
                
                # Mostrar tabla de zonas
                zones = tp.generate_zones(ftp=ftp)
                if 'power' in zones:
                    st.subheader("Zonas de potencia")
                    zone_data = []
                    for zone_name, (zone_min, zone_max) in zones['power'].items():
                        zone_data.append({
                            "Zona": zone_name,
                            "Mínimo (W)": zone_min,
                            "Máximo (W)": zone_max,
                            "% FTP": f"{int(zone_min/ftp*100 if zone_min > 0 else 0)}-{int(zone_max/ftp*100)}%"
                        })
                    st.table(pd.DataFrame(zone_data))
            
            elif zone_type == "Frecuencia Cardíaca":
                fig = tp.plot_zone_distribution(
                    st.session_state.activities_df, 
                    zone_type='heart_rate', 
                    lthr=lthr
                )
                st.pyplot(fig)
                
                # Mostrar tabla de zonas
                zones = tp.generate_zones(lthr=lthr)
                if 'heart_rate' in zones:
                    st.subheader("Zonas de frecuencia cardíaca")
                    zone_data = []
                    for zone_name, (zone_min, zone_max) in zones['heart_rate'].items():
                        zone_data.append({
                            "Zona": zone_name,
                            "Mínimo (ppm)": zone_min,
                            "Máximo (ppm)": zone_max,
                            "% LTHR": f"{int(zone_min/lthr*100 if zone_min > 0 else 0)}-{int(zone_max/lthr*100)}%"
                        })
                    st.table(pd.DataFrame(zone_data))
            
            elif zone_type == "Ritmo":
                fig = tp.plot_zone_distribution(
                    st.session_state.activities_df, 
                    zone_type='pace', 
                    threshold_pace=threshold_pace
                )
                st.pyplot(fig)
                
                # Mostrar tabla de zonas
                zones = tp.generate_zones(threshold_pace=threshold_pace)
                if 'pace' in zones:
                    st.subheader("Zonas de ritmo")
                    zone_data = []
                    for zone_name, (zone_min, zone_max) in zones['pace'].items():
                        zone_data.append({
                            "Zona": zone_name,
                            "Ritmo Mínimo": tp.format_pace(zone_min),
                            "Ritmo Máximo": tp.format_pace(zone_max),
                            "% Ritmo Umbral": f"{int(threshold_pace/zone_min*100)}-{int(threshold_pace/zone_max*100 if zone_max > 0 else 0)}%"
                        })
                    st.table(pd.DataFrame(zone_data))
        
        elif analysis_type == "Progresión de métricas":
            # Seleccionar métricas y período
            metrics = st.multiselect(
                "Selecciona métricas a analizar:",
                ["TSS", "CTL", "ATL", "TSB", "IF", "NP", "Duración"],
                default=["TSS", "CTL"]
            )
            
            date_range = st.date_input(
                "Selecciona período a analizar:",
                [
                    st.session_state.activities_df['date'].min().date(),
                    st.session_state.activities_df['date'].max().date()
                ]
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                
                # Filtrar datos por fecha
                filtered_df = st.session_state.activities_df[
                    (st.session_state.activities_df['date'].dt.date >= start_date) &
                    (st.session_state.activities_df['date'].dt.date <= end_date)
                ]
                
                if not filtered_df.empty:
                    # Calcular datos PMC si es necesario
                    if any(metric in ["CTL", "ATL", "TSB"] for metric in metrics):
                        pmc_data = tp.calculate_pmc_data(filtered_df)
                        
                        # Fusionar con datos de actividad para análisis
                        if not pmc_data.empty:
                            # Agrupar actividades por día para análisis
                            daily_activities = filtered_df.groupby(filtered_df['date'].dt.date).agg({
                                'TSS': 'sum',
                                'duration_seconds': 'sum',
                                'np': 'mean',
                                'intensity_factor': 'mean'
                            }).reset_index()
                            daily_activities['date'] = pd.to_datetime(daily_activities['date'])
                            
                            # Fusionar con datos PMC
                            analysis_df = pd.merge(
                                pmc_data,
                                daily_activities,
                                on='date',
                                how='left'
                            )
                            
                            # Rellenar valores NaN
                            analysis_df.fillna({
                                'TSS': 0,
                                'duration_seconds': 0,
                                'np': 0,
                                'intensity_factor': 0
                            }, inplace=True)
                            
                            # Convertir duración a horas
                            analysis_df['Duration'] = analysis_df['duration_seconds'] / 3600
                            
                            # Crear gráfico
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            # Graficar métricas seleccionadas
                            for metric in metrics:
                                if metric in analysis_df.columns:
                                    ax.plot(analysis_df['date'], analysis_df[metric], label=metric)
                            
                            ax.set_xlabel('Fecha')
                            ax.set_ylabel('Valor')
                            ax.set_title('Progresión de métricas')
                            ax.legend()
                            ax.grid(True, linestyle='--', alpha=0.7)
                            
                            # Formatear fechas en el eje x
                            fig.autofmt_xdate()
                            
                            st.pyplot(fig)
                    else:
                        # Análisis simple de métricas diarias
                        daily_activities = filtered_df.groupby(filtered_df['date'].dt.date).agg({
                            'TSS': 'sum',
                            'duration_seconds': 'sum',
                            'np': 'mean',
                            'intensity_factor': 'mean'
                        }).reset_index()
                        daily_activities['date'] = pd.to_datetime(daily_activities['date'])
                        
                        # Convertir duración a horas
                        daily_activities['Duration'] = daily_activities['duration_seconds'] / 3600
                        
                        # Crear gráfico
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Graficar métricas seleccionadas
                        for metric in metrics:
                            if metric in daily_activities.columns:
                                ax.plot(daily_activities['date'], daily_activities[metric], label=metric)
                        
                        ax.set_xlabel('Fecha')
                        ax.set_ylabel('Valor')
                        ax.set_title('Progresión de métricas')
                        ax.legend()
                        ax.grid(True, linestyle='--', alpha=0.7)
                        
                        # Formatear fechas en el eje x
                        fig.autofmt_xdate()
                        
                        st.pyplot(fig)
                else:
                    st.info("No hay datos para el período seleccionado.")
        
        elif analysis_type == "Análisis por tipo de actividad":
            # Seleccionar tipo de actividad
            activity_types = st.session_state.activities_df['activity_type'].unique()
            selected_type = st.selectbox(
                "Selecciona tipo de actividad:",
                activity_types
            )
            
            # Filtrar por tipo de actividad
            filtered_df = st.session_state.activities_df[
                st.session_state.activities_df['activity_type'] == selected_type
            ]
            
            if not filtered_df.empty:
                st.subheader(f"Análisis de actividades: {selected_type.capitalize()}")
                
                # Estadísticas generales
                total_activities = len(filtered_df)
                total_duration = filtered_df['duration_seconds'].sum() / 3600  # horas
                total_tss = filtered_df['TSS'].sum()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Actividades", f"{total_activities}")
                
                with col2:
                    st.metric("Tiempo Total", f"{total_duration:.1f} h")
                
                with col3:
                    st.metric("TSS Total", f"{total_tss:.0f}")
                
                # Métricas específicas según el tipo
                if selected_type == 'cycling':
                    avg_power = filtered_df['avg_power'].mean()
                    avg_np = filtered_df['np'].mean() if 'np' in filtered_df.columns else 0
                    avg_if = filtered_df['intensity_factor'].mean() if 'intensity_factor' in filtered_df.columns else 0
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Potencia Media", f"{avg_power:.0f} W")
                    
                    with col2:
                        st.metric("NP Media", f"{avg_np:.0f} W")
                    
                    with col3:
                        st.metric("IF Medio", f"{avg_if:.2f}")
                
                elif selected_type == 'running':
                    avg_pace = filtered_df['avg_pace'].mean() if 'avg_pace' in filtered_df.columns else 0
                    
                    st.metric("Ritmo Medio", tp.format_pace(avg_pace))
                
                # Gráfico de progresión
                st.subheader("Progresión de rendimiento")
                
                # Ordenar por fecha
                progress_df = filtered_df.sort_values('date')
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                if selected_type == 'cycling':
                    # Progresión de potencia
                    ax.plot(progress_df['date'], progress_df['avg_power'], 'b-', label='Potencia Media')
                    if 'np' in progress_df.columns:
                        ax.plot(progress_df['date'], progress_df['np'], 'r-', label='NP')
                    
                    ax.set_ylabel('Potencia (W)')
                elif selected_type == 'running':
                    # Progresión de ritmo (invertido para que mejor sea arriba)
                    if 'avg_pace' in progress_df.columns:
                        # Convertir pace a min/km para visualización
                        pace_minutes = progress_df['avg_pace'] / 60
                        ax.plot(progress_df['date'], pace_minutes, 'g-', label='Ritmo')
                        ax.invert_yaxis()  # Invertir eje Y para que mejor ritmo esté arriba
                        ax.set_ylabel('Ritmo (min/km)')
                
                ax.set_xlabel('Fecha')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                
                fig.autofmt_xdate()
                
                st.pyplot(fig)
                
                # Tabla de actividades
                st.subheader("Actividades")
                
                # Mostrar tabla con las métricas más relevantes según tipo
                if selected_type == 'cycling':
                    display_cols = ['date', 'title', 'duration_seconds', 'avg_power', 'np', 'intensity_factor', 'TSS']
                elif selected_type == 'running':
                    display_cols = ['date', 'title', 'duration_seconds', 'avg_pace', 'TSS']
                elif selected_type == 'swimming':
                    display_cols = ['date', 'title', 'duration_seconds', 'intensity', 'TSS']
                else:
                    display_cols = ['date', 'title', 'duration_seconds', 'TSS']
                
                display_df = filtered_df[display_cols].sort_values('date', ascending=False).copy()
                
                # Formatear columnas para mejor visualización
                if 'duration_seconds' in display_df.columns:
                    display_df['Duración'] = display_df['duration_seconds'].apply(
                        lambda x: str(dt.timedelta(seconds=int(x)))
                    )
                    display_df = display_df.drop('duration_seconds', axis=1)
                
                if 'avg_pace' in display_df.columns:
                    display_df['Ritmo'] = display_df['avg_pace'].apply(tp.format_pace)
                    display_df = display_df.drop('avg_pace', axis=1)
                
                # Renombrar columnas para mejor visualización
                display_df = display_df.rename(columns={
                    'date': 'Fecha',
                    'title': 'Título',
                    'avg_power': 'Potencia Media',
                    'np': 'NP',
                    'intensity_factor': 'IF',
                    'intensity': 'Intensidad'
                })
                
                st.dataframe(display_df)
            else:
                st.info(f"No hay actividades de tipo {selected_type}.")
        
        elif analysis_type == "Comparación de períodos":
            st.subheader("Comparación de períodos de entrenamiento")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Período 1**")
                period1_start, period1_end = st.date_input(
                    "Rango de fechas (Período 1):",
                    [
                        st.session_state.activities_df['date'].min().date(),
                        (st.session_state.activities_df['date'].min() + dt.timedelta(days=28)).date()
                    ],
                    key="period1"
                )
            
            with col2:
                st.write("**Período 2**")
                period2_start, period2_end = st.date_input(
                    "Rango de fechas (Período 2):",
                    [
                        (st.session_state.activities_df['date'].min() + dt.timedelta(days=29)).date(),
                        st.session_state.activities_df['date'].max().date()
                    ],
                    key="period2"
                )
            
            # Filtrar datos para ambos períodos
            period1_df = st.session_state.activities_df[
                (st.session_state.activities_df['date'].dt.date >= period1_start) &
                (st.session_state.activities_df['date'].dt.date <= period1_end)
            ]
            
            period2_df = st.session_state.activities_df[
                (st.session_state.activities_df['date'].dt.date >= period2_start) &
                (st.session_state.activities_df['date'].dt.date <= period2_end)
            ]
            
            if not period1_df.empty and not period2_df.empty:
                # Duración de los períodos en días
                period1_days = (period1_end - period1_start).days + 1
                period2_days = (period2_end - period2_start).days + 1
                
                # Calculamos resumen de métricas para cada período
                period1_metrics = {
                    'Días': period1_days,
                    'Actividades': len(period1_df),
                    'Actividades/semana': len(period1_df) / (period1_days/7),
                    'TSS Total': period1_df['TSS'].sum(),
                    'TSS/semana': period1_df['TSS'].sum() / (period1_days/7),
                    'Duración Total (h)': period1_df['duration_seconds'].sum() / 3600,
                    'Duración/semana (h)': (period1_df['duration_seconds'].sum() / 3600) / (period1_days/7)
                }
                
                period2_metrics = {
                    'Días': period2_days,
                    'Actividades': len(period2_df),
                    'Actividades/semana': len(period2_df) / (period2_days/7),
                    'TSS Total': period2_df['TSS'].sum(),
                    'TSS/semana': period2_df['TSS'].sum() / (period2_days/7),
                    'Duración Total (h)': period2_df['duration_seconds'].sum() / 3600,
                    'Duración/semana (h)': (period2_df['duration_seconds'].sum() / 3600) / (period2_days/7)
                }
                
                # Crear DataFrame para visualización
                metrics_df = pd.DataFrame({
                    'Métrica': list(period1_metrics.keys()),
                    'Período 1': list(period1_metrics.values()),
                    'Período 2': list(period2_metrics.values()),
                    'Diferencia (%)': [
                        round((period2_metrics[key] - period1_metrics[key]) / period1_metrics[key] * 100 if period1_metrics[key] != 0 else 0, 1)
                        for key in period1_metrics.keys()
                    ]
                })
                
                # Formatear valores numéricos
                for col in ['Período 1', 'Período 2']:
                    metrics_df[col] = metrics_df[col].apply(
                        lambda x: round(x, 1) if isinstance(x, float) else x
                    )
                
                # Mostrar tabla comparativa
                st.table(metrics_df)
                
                # Distribución por tipo de actividad
                st.subheader("Distribución por tipo de actividad")
                
                # Calcular distribución para ambos períodos
                period1_types = period1_df['activity_type'].value_counts()
                period2_types = period2_df['activity_type'].value_counts()
                
                # Crear gráfico de barras comparativas
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Obtener todos los tipos únicos
                all_types = list(set(list(period1_types.index) + list(period2_types.index)))
                x = np.arange(len(all_types))
                width = 0.35
                
                # Valores para cada período
                period1_values = [period1_types.get(t, 0) for t in all_types]
                period2_values = [period2_types.get(t, 0) for t in all_types]
                
                # Crear barras
                ax.bar(x - width/2, period1_values, width, label=f'Período 1 ({period1_start} a {period1_end})')
                ax.bar(x + width/2, period2_values, width, label=f'Período 2 ({period2_start} a {period2_end})')
                
                ax.set_xticks(x)
                ax.set_xticklabels([t.capitalize() for t in all_types])
                ax.legend()
                ax.set_ylabel('Número de actividades')
                ax.set_title('Comparación de actividades por tipo')
                
                st.pyplot(fig)
                
                # Progresión de CTL en ambos períodos
                st.subheader("Progresión de Fitness (CTL)")
                
                # Calcular PMC para cada período
                pmc1 = tp.calculate_pmc_data(period1_df)
                pmc2 = tp.calculate_pmc_data(period2_df)
                
                if not pmc1.empty and not pmc2.empty:
                    # Normalizar días para comparación (0 a n días)
                    pmc1['day'] = (pmc1['date'] - pmc1['date'].min()).dt.days
                    pmc2['day'] = (pmc2['date'] - pmc2['date'].min()).dt.days
                    
                    # Crear gráfico
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    ax.plot(pmc1['day'], pmc1['CTL'], 'b-', label=f'Período 1')
                    ax.plot(pmc2['day'], pmc2['CTL'], 'r-', label=f'Período 2')
                    
                    ax.set_xlabel('Días desde inicio del período')
                    ax.set_ylabel('CTL (Fitness)')
                    ax.set_title('Comparación de progresión de Fitness')
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    st.pyplot(fig)
            else:
                st.info("No hay suficientes datos para comparar los períodos seleccionados.")
    else:
        st.info("No hay datos de actividades. Registra una actividad o genera datos de muestra en la sección 'Gestión de Datos'.")

def data_management_page():
    """Página de gestión de datos"""
    st.header("Gestión de Datos")
    
    # Pestañas para diferentes funciones de gestión de datos
    tab1, tab2, tab3 = st.tabs(["Generar Datos de Muestra", "Importar/Exportar", "Ayuda"])
    
    with tab1:
        st.subheader("Generar Datos de Muestra")
        st.write("Genera datos de entrenamiento de muestra para explorar las funcionalidades de la aplicación.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Configurar opciones para la generación de datos
            num_days = st.slider(
                "Período a generar (días)",
                min_value=30,
                max_value=365,
                value=90,
                step=30
            )
            
            ftp = st.session_state.athlete_data['ftp']
            
            activities_per_week = st.slider(
                "Promedio de actividades por semana",
                min_value=1,
                max_value=14,
                value=5
            )
        
        with col2:
            # Previsualización de datos a generar
            st.write(f"Se generarán aproximadamente {int(num_days * activities_per_week / 7)} actividades")
            st.write(f"Período: Últimos {num_days} días")
            st.write(f"FTP para cálculos: {ftp} W")
            
            # Advertencia de sobrescritura
            st.warning("La generación de datos nuevos sobrescribirá los datos existentes.")
        
        # Botón para generar datos
        if st.button("Generar Datos de Muestra"):
            with st.spinner("Generando datos..."):
                # Generar datos de muestra
                sample_data = tp.generate_sample_data(
                    num_days=num_days,
                    ftp=ftp,
                    activities_per_week=activities_per_week
                )
                
                # Guardar en sesión y en CSV
                st.session_state.activities_df = sample_data
                try:
                    sample_data.to_csv('training_data.csv', index=False)
                    st.success(f"Se han generado {len(sample_data)} actividades de muestra.")
                except Exception as e:
                    st.error(f"Error al guardar datos: {e}")
    
    with tab2:
        st.subheader("Importar/Exportar Datos")
        
        # Sección de carga de archivos
        st.write("**Importar Datos**")
        
        uploaded_file = st.file_uploader(
            "Cargar archivo CSV con datos de actividades",
            type=["csv"]
        )
        
        if uploaded_file is not None:
            try:
                # Procesar archivo cargado
                imported_df = tp.csv_to_dataframe(uploaded_file)
                
                if not imported_df.empty:
                    st.session_state.activities_df = imported_df
                    
                    # Mostrar vista previa
                    st.write(f"Datos cargados: {len(imported_df)} actividades")
                    st.dataframe(imported_df.head())
                    
                    # Opción para guardar
                    if st.button("Guardar Datos Importados"):
                        imported_df.to_csv('training_data.csv', index=False)
                        st.success("Datos guardados correctamente.")
                else:
                    st.error("El archivo no contiene datos válidos.")
            except Exception as e:
                st.error(f"Error al procesar el archivo: {e}")
        
        # Sección de exportación
        st.write("**Exportar Datos**")
        
        if 'activities_df' in st.session_state and not st.session_state.activities_df.empty:
            # Crear botón de descarga
            csv = st.session_state.activities_df.to_csv(index=False)
            
            # Función para crear enlace de descarga
            def get_download_link(csv_data, filename="training_data.csv"):
                b64 = base64.b64encode(csv_data.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Descargar CSV</a>'
                return href
            
            st.markdown(get_download_link(csv), unsafe_allow_html=True)
            
            # Opción para borrar datos
            if st.button("Borrar Todos los Datos"):
                confirm = st.checkbox("Confirmar eliminación")
                
                if confirm:
                    st.session_state.activities_df = pd.DataFrame()
                    try:
                        if os.path.exists('training_data.csv'):
                            os.remove('training_data.csv')
                        st.success("Datos eliminados correctamente.")
                    except Exception as e:
                        st.error(f"Error al eliminar archivo: {e}")
        else:
            st.info("No hay datos para exportar.")
    
    with tab3:
        st.subheader("Ayuda y Documentación")
        
        with st.expander("Estructura de Datos"):
            st.markdown("""
            ### Estructura del CSV de Actividades
            
            Para importar correctamente tus datos, el archivo CSV debe contener las siguientes columnas:
            
            #### Columnas obligatorias:
            
            - **date**: Fecha de la actividad (formato YYYY-MM-DD o DD/MM/YYYY)
            - **activity_type**: Tipo de actividad (cycling, running, swimming)
            - **duration_seconds**: Duración en segundos
            - **TSS**: Training Stress Score
            
            #### Columnas específicas por actividad:
            
            **Ciclismo (cycling)**:
            - **avg_power**: Potencia media
            - **np**: Normalized Power
            - **intensity_factor**: Intensity Factor
            - **ftp**: FTP usado para los cálculos
            
            **Carrera (running)**:
            - **avg_pace**: Ritmo medio en segundos por km
            - **threshold_pace**: Ritmo umbral en segundos por km
            - **grade_factor**: Factor de corrección por desnivel
            
            **Natación (swimming)**:
            - **intensity**: Intensidad relativa de la sesión
            
            #### Otras columnas útiles:
            
            - **title**: Título de la actividad
            - **avg_hr**: Frecuencia cardíaca media
            
            ### Ejemplo de formato:
            
            ```
            date,title,activity_type,duration_seconds,avg_power,np,intensity_factor,TSS,avg_hr
            2023-01-01,Entrenamiento en bicicleta,cycling,3600,200,220,0.8,64,150
            2023-01-03,Carrera suave,running,1800,,,,,145
            ```
            """)
        
        with st.expander("Métricas principales"):
            st.markdown("""
            ### Métricas principales de TrainingPeaks
            
            #### TSS (Training Stress Score)
            Cuantifica la carga de entrenamiento de una sesión combinando intensidad y duración en un único valor. 100 TSS equivale a una hora a intensidad de umbral (FTP).
            
            - **Ciclismo**: TSS = (segundos × NP × IF) ÷ (FTP × 3600) × 100
            - **Carrera**: TSS = (duración en horas) × (ritmo NGP/ritmo umbral)² × 100
            - **FC**: TSS = (duración en minutos) × (FC media/FC umbral)² × 100 / 60
            
            #### NP (Normalized Power)
            Estima el costo metabólico de una sesión de ciclismo considerando las variaciones de potencia, no solo el promedio.
            
            #### IF (Intensity Factor)
            Intensidad relativa de un entrenamiento comparado con tu capacidad máxima sostenible (FTP).
            - IF = NP / FTP
            
            #### CTL (Chronic Training Load)
            Representa tu "fitness" o condición física acumulada a largo plazo. Es un promedio exponencialmente ponderado de TSS de los últimos 42 días.
            
            #### ATL (Acute Training Load)
            Representa tu fatiga a corto plazo. Es un promedio exponencialmente ponderado de TSS de los últimos 7 días.
            
            #### TSB (Training Stress Balance)
            Representa tu "forma" o "frescura". Se calcula como CTL - ATL.
            - TSB positivo: Descansado, potencialmente en forma
            - TSB negativo: Fatigado, potencialmente en fase de construcción
            
            #### VI (Variability Index)
            Ratio entre Normalized Power y potencia media. Indica qué tan constante o variable fue el esfuerzo.
            - VI cercano a 1.0: Esfuerzo constante
            - VI alto (>1.1): Esfuerzo muy variable
            """)
        
        with st.expander("Principios de planificación"):
            st.markdown("""
            ### Principios de Planificación con TrainingPeaks
            
            #### Periodización utilizando CTL, ATL y TSB
            
            1. **Fase de Construcción**
               - Incremento gradual de CTL (1-5 puntos por semana)
               - TSB negativo (-10 a -30)
               - Duración: 4-12 semanas
               - Objetivo: Aumentar fitness sacrificando frescura
            
            2. **Fase de Especialidad**
               - Mantener CTL alto
               - Alternar bloques de carga (TSB más negativo) y recuperación (TSB cercano a cero)
               - Duración: 4-8 semanas
               - Objetivo: Desarrollar capacidades específicas para el evento objetivo
            
            3. **Taper (Afinamiento)**
               - Reducción gradual de CTL (reducción de volumen, mantenimiento de intensidad)
               - Aumento de TSB hacia valores positivos (+5 a +15)
               - Duración: 1-3 semanas
               - Objetivo: Llegar al evento objetivo con buena forma (TSB positivo)
            
            4. **Recuperación/Transición**
               - Descanso activo
               - Permitir que CTL baje
               - TSB muy positivo (>+15)
               - Duración: 1-4 semanas
               - Objetivo: Recuperación física y mental
            
            #### Recomendaciones generales
            
            - **Ramp Rate**: El incremento de CTL no debería exceder 5-8 puntos por semana para evitar sobreentrenamiento
            - **TSB para competir**: +5 a +15 (ideal para eventos prioritarios)
            - **TSB para entrenar**: -10 a +5 (rango ideal para entrenamientos de calidad)
            - **TSB muy negativo** (<-25): Riesgo de sobreentrenamiento, necesaria recuperación
            - **Distribución de intensidad**: Seguir el modelo polarizado (80% baja intensidad, 20% alta intensidad)
            
            #### Aplicación práctica
            
            1. Establece una meta de CTL realista según tu nivel y disponibilidad
            2. Planifica incrementos graduales de CTL
            3. Programa descansos estratégicos (días o semanas)
            4. Sincroniza el TSB positivo con eventos importantes
            5. Utiliza las predicciones para ajustar la carga futura
            """)

# Función principal
def main():
    """Función principal para ejecutar la aplicación"""
    # Mostrar encabezado
    display_header()
    
    # Mostrar navegación lateral y obtener página seleccionada
    page = sidebar_navigation()
    
    # Cargar datos
    load_or_generate_data()
    
    # Mostrar página correspondiente
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "📊 Análisis de Métricas":
        metrics_analysis_page()
    elif page == "📝 Registro de Actividad":
        activity_log_page()
    elif page == "📈 PMC (Performance Management Chart)":
        pmc_page()
    elif page == "⚙️ Configuración":
        configuration_page()
    elif page == "📁 Gestión de Datos":
        data_management_page()

# Ejecutar la aplicación
if __name__ == "__main__":
    main()