"""
Training Peaks Metrics Calculator Functions
------------------------------------------
Este módulo contiene todas las funciones necesarias para calcular las métricas
utilizadas en TrainingPeaks para la planificación y seguimiento del entrenamiento deportivo.
"""

import numpy as np
import pandas as pd
import datetime as dt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import matplotlib.dates as mdates
import random

# Constantes
DEFAULT_CTL_DAYS = 42  # Días para el cálculo de CTL (fitness)
DEFAULT_ATL_DAYS = 7   # Días para el cálculo de ATL (fatiga)

def calculate_normalized_power(power_array, window_size=30):
    """
    Calcula el Normalized Power (NP) a partir de un array de datos de potencia.
    
    Args:
        power_array: Array de datos de potencia (en watts)
        window_size: Tamaño de la ventana móvil en segundos (por defecto 30s)
    
    Returns:
        float: Valor de Normalized Power
    """
    if len(power_array) == 0:
        return 0
    
    # Convertir a array de numpy para un procesamiento más rápido
    power = np.array(power_array)
    
    # Aplicar promedio móvil de 30 segundos
    weights = np.ones(window_size) / window_size
    power_rolling = np.convolve(power, weights, mode='valid')
    
    # Elevar a la cuarta potencia
    power_rolling_4 = np.power(power_rolling, 4)
    
    # Calcular el promedio
    avg_power_4 = np.mean(power_rolling_4)
    
    # Raíz cuarta del resultado
    np_value = np.power(avg_power_4, 0.25)
    
    return round(np_value, 1)

def calculate_intensity_factor(np_value, ftp):
    """
    Calcula el Intensity Factor (IF) a partir del NP y el FTP.
    
    Args:
        np_value: Valor de Normalized Power
        ftp: Functional Threshold Power del atleta
    
    Returns:
        float: Valor del Intensity Factor
    """
    if ftp == 0:
        return 0
    
    return round(np_value / ftp, 2)

def calculate_variability_index(np_value, avg_power):
    """
    Calcula el Variability Index (VI) a partir del NP y la potencia media.
    
    Args:
        np_value: Valor de Normalized Power
        avg_power: Potencia media de la sesión
    
    Returns:
        float: Valor del Variability Index
    """
    if avg_power == 0:
        return 0
    
    return round(np_value / avg_power, 2)

def calculate_tss_from_power(duration_seconds, np_value, intensity_factor, ftp):
    """
    Calcula el Training Stress Score (TSS) a partir de los parámetros de potencia.
    
    Args:
        duration_seconds: Duración del entrenamiento en segundos
        np_value: Valor de Normalized Power
        intensity_factor: Intensity Factor
        ftp: Functional Threshold Power del atleta
    
    Returns:
        float: Valor del TSS
    """
    if ftp == 0 or duration_seconds == 0:
        return 0
    
    tss = (duration_seconds * np_value * intensity_factor) / (ftp * 3600) * 100
    return round(tss, 0)

def calculate_tss_from_hr(duration_minutes, avg_hr, threshold_hr):
    """
    Calcula el Training Stress Score (TSS) utilizando datos de frecuencia cardíaca.
    
    Args:
        duration_minutes: Duración del entrenamiento en minutos
        avg_hr: Frecuencia cardíaca media durante el entrenamiento
        threshold_hr: Frecuencia cardíaca en el umbral del atleta
    
    Returns:
        float: Valor del TSS basado en frecuencia cardíaca
    """
    if threshold_hr == 0 or duration_minutes == 0:
        return 0
    
    hr_ratio = avg_hr / threshold_hr
    tss = duration_minutes * hr_ratio * hr_ratio * 100 / 60
    return round(tss, 0)

def calculate_running_tss(duration_hours, avg_pace, threshold_pace, grade_factor=1.0):
    """
    Calcula el Running Training Stress Score (rTSS).
    
    Args:
        duration_hours: Duración del entrenamiento en horas
        avg_pace: Ritmo medio (en segundos por km)
        threshold_pace: Ritmo umbral (en segundos por km)
        grade_factor: Factor de corrección por desnivel
    
    Returns:
        float: Valor del rTSS
    """
    if threshold_pace == 0 or duration_hours == 0:
        return 0
    
    # El ritmo más bajo (más rápido) tiene un valor más alto, así que invertimos la relación
    pace_ratio = threshold_pace / (avg_pace * grade_factor)
    rtss = duration_hours * (pace_ratio * pace_ratio) * 100
    return round(rtss, 0)

def calculate_swimming_tss(duration_hours, intensity):
    """
    Calcula el Swimming Training Stress Score (sTSS).
    
    Args:
        duration_hours: Duración del entrenamiento en horas
        intensity: Intensidad relativa de la sesión (0.5-1.5)
    
    Returns:
        float: Valor del sTSS
    """
    if duration_hours == 0:
        return 0
    
    stss = duration_hours * (intensity * intensity) * 100
    return round(stss, 0)

def calculate_trimp(duration_minutes, avg_hr, rest_hr, max_hr, gender='M'):
    """
    Calcula el Training Impulse (TRIMP) basado en frecuencia cardíaca.
    
    Args:
        duration_minutes: Duración del entrenamiento en minutos
        avg_hr: Frecuencia cardíaca media
        rest_hr: Frecuencia cardíaca en reposo
        max_hr: Frecuencia cardíaca máxima
        gender: Género ('M' para masculino, 'F' para femenino)
    
    Returns:
        float: Valor de TRIMP
    """
    if max_hr <= rest_hr or duration_minutes == 0:
        return 0
    
    # Calcular el ratio de reserva de frecuencia cardíaca
    hr_ratio = (avg_hr - rest_hr) / (max_hr - rest_hr)
    
    # Coeficientes diferentes según género
    if gender.upper() == 'F':
        y = 1.67 * hr_ratio * np.exp(1.92 * hr_ratio)
    else:
        y = 0.64 * hr_ratio * np.exp(1.92 * hr_ratio)
    
    trimp = duration_minutes * y
    return round(trimp, 0)

def calculate_ctl(tss_array, previous_ctl=None, time_constant=DEFAULT_CTL_DAYS):
    """
    Calcula el Chronic Training Load (CTL) o "fitness".
    
    Args:
        tss_array: Array con los valores de TSS diarios
        previous_ctl: Valor CTL anterior (opcional)
        time_constant: Constante de tiempo para el cálculo (por defecto 42 días)
    
    Returns:
        float: Valor de CTL
    """
    if not tss_array:
        return previous_ctl if previous_ctl is not None else 0
    
    # Si hay un valor previo de CTL, lo usamos como punto de partida
    ctl = previous_ctl if previous_ctl is not None else tss_array[0]
    
    # Factor de decaimiento diario
    decay_factor = 1 - (1 / time_constant)
    
    # Aplicamos la fórmula exponencialmente ponderada
    for tss in tss_array:
        ctl = (decay_factor * ctl) + ((1 - decay_factor) * tss)
    
    return round(ctl, 1)

def calculate_atl(tss_array, previous_atl=None, time_constant=DEFAULT_ATL_DAYS):
    """
    Calcula el Acute Training Load (ATL) o "fatiga".
    
    Args:
        tss_array: Array con los valores de TSS diarios
        previous_atl: Valor ATL anterior (opcional)
        time_constant: Constante de tiempo para el cálculo (por defecto 7 días)
    
    Returns:
        float: Valor de ATL
    """
    if not tss_array:
        return previous_atl if previous_atl is not None else 0
    
    # Si hay un valor previo de ATL, lo usamos como punto de partida
    atl = previous_atl if previous_atl is not None else tss_array[0]
    
    # Factor de decaimiento diario
    decay_factor = 1 - (1 / time_constant)
    
    # Aplicamos la fórmula exponencialmente ponderada
    for tss in tss_array:
        atl = (decay_factor * atl) + ((1 - decay_factor) * tss)
    
    return round(atl, 1)

def calculate_tsb(ctl, atl):
    """
    Calcula el Training Stress Balance (TSB) o "forma".
    
    Args:
        ctl: Chronic Training Load
        atl: Acute Training Load
    
    Returns:
        float: Valor de TSB
    """
    return round(ctl - atl, 1)

def calculate_metrics_from_session(session_data):
    """
    Calcula todas las métricas relevantes a partir de los datos de una sesión.
    
    Args:
        session_data: Diccionario con los datos de la sesión
    
    Returns:
        dict: Diccionario con todas las métricas calculadas
    """
    metrics = {}
    
    # Extraer datos básicos
    activity_type = session_data.get('activity_type', 'cycling')
    duration_seconds = session_data.get('duration_seconds', 0)
    duration_hours = duration_seconds / 3600
    duration_minutes = duration_seconds / 60
    
    # Datos según el tipo de actividad
    if activity_type == 'cycling':
        # Datos de potencia
        power_array = session_data.get('power_data', [])
        avg_power = np.mean(power_array) if len(power_array) > 0 else session_data.get('avg_power', 0)
        ftp = session_data.get('ftp', 200)
        
        # Calcular NP, IF, VI
        np_value = calculate_normalized_power(power_array) if len(power_array) > 0 else session_data.get('np', 0)
        intensity_factor = calculate_intensity_factor(np_value, ftp)
        vi = calculate_variability_index(np_value, avg_power)
        
        # Calcular TSS basado en potencia
        tss = calculate_tss_from_power(duration_seconds, np_value, intensity_factor, ftp)
        
        metrics.update({
            'NP': np_value,
            'IF': intensity_factor,
            'VI': vi,
            'AP': round(avg_power, 1),
            'TSS': tss
        })
    
    elif activity_type == 'running':
        # Datos de carrera
        avg_pace = session_data.get('avg_pace', 300)  # en segundos por km
        threshold_pace = session_data.get('threshold_pace', 240)  # en segundos por km
        grade_factor = session_data.get('grade_factor', 1.0)
        
        # Calcular rTSS
        rtss = calculate_running_tss(duration_hours, avg_pace, threshold_pace, grade_factor)
        
        metrics.update({
            'Pace': avg_pace,
            'NGP': round(avg_pace / grade_factor, 1),
            'rTSS': rtss,
            'TSS': rtss  # Usar rTSS como TSS para el PMC
        })
    
    elif activity_type == 'swimming':
        # Datos de natación
        intensity = session_data.get('intensity', 1.0)
        
        # Calcular sTSS
        stss = calculate_swimming_tss(duration_hours, intensity)
        
        metrics.update({
            'Intensity': intensity,
            'sTSS': stss,
            'TSS': stss  # Usar sTSS como TSS para el PMC
        })
    
    # Datos de frecuencia cardíaca (comunes a todas las actividades)
    avg_hr = session_data.get('avg_hr', 0)
    rest_hr = session_data.get('rest_hr', 60)
    max_hr = session_data.get('max_hr', 180)
    threshold_hr = session_data.get('threshold_hr', 160)
    gender = session_data.get('gender', 'M')
    
    # Calcular TRIMP y TSS basado en HR (si no hay datos de potencia)
    trimp = calculate_trimp(duration_minutes, avg_hr, rest_hr, max_hr, gender)
    hr_tss = calculate_tss_from_hr(duration_minutes, avg_hr, threshold_hr)
    
    # Si no hay un TSS calculado por potencia o ritmo, usamos el basado en HR
    if 'TSS' not in metrics or metrics['TSS'] == 0:
        metrics['TSS'] = hr_tss
    
    metrics.update({
        'TRIMP': trimp,
        'HR_TSS': hr_tss,
        'Duration': duration_seconds,
        'Duration_h': round(duration_hours, 2),
        'Avg_HR': avg_hr
    })
    
    return metrics

def calculate_pmc_data(activities_df):
    """
    Calcula los datos para el Performance Management Chart (PMC).
    
    Args:
        activities_df: DataFrame con las actividades
    
    Returns:
        DataFrame: DataFrame con datos diarios de CTL, ATL y TSB
    """
    if activities_df.empty:
        return pd.DataFrame()
    
    # Asegurarse de que la fecha es datetime y ordenar
    activities_df['date'] = pd.to_datetime(activities_df['date'])
    activities_df = activities_df.sort_values('date')
    
    # Crear un DataFrame con fechas continuas
    start_date = activities_df['date'].min()
    end_date = activities_df['date'].max()
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Crear DataFrame diario
    daily_df = pd.DataFrame({'date': date_range})
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    # Agregar TSS por día (puede haber múltiples actividades por día)
    tss_by_day = activities_df.groupby(activities_df['date'].dt.date)['TSS'].sum().reset_index()
    tss_by_day['date'] = pd.to_datetime(tss_by_day['date'])
    
    # Fusionar con el DataFrame diario
    daily_df = pd.merge(daily_df, tss_by_day, on='date', how='left')
    daily_df['TSS'].fillna(0, inplace=True)
    
    # Calcular CTL y ATL acumulativos
    ctl_values = []
    atl_values = []
    tsb_values = []
    
    ctl = 0
    atl = 0
    
    for i, row in daily_df.iterrows():
        tss = row['TSS']
        
        # Actualizar CTL y ATL usando fórmulas de promedio exponencialmente ponderado
        ctl = ctl + (tss - ctl) / DEFAULT_CTL_DAYS
        atl = atl + (tss - atl) / DEFAULT_ATL_DAYS
        tsb = ctl - atl
        
        ctl_values.append(round(ctl, 1))
        atl_values.append(round(atl, 1))
        tsb_values.append(round(tsb, 1))
    
    daily_df['CTL'] = ctl_values
    daily_df['ATL'] = atl_values
    daily_df['TSB'] = tsb_values
    
    return daily_df

def plot_pmc(pmc_data, figsize=(12, 8)):
    """
    Genera el gráfico del Performance Management Chart (PMC).
    
    Args:
        pmc_data: DataFrame con los datos del PMC (fecha, CTL, ATL, TSB, TSS)
        figsize: Tamaño del gráfico
    
    Returns:
        fig: Objeto de figura de matplotlib
    """
    if pmc_data.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No hay datos suficientes para generar el PMC", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return fig
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Configurar eje para CTL y ATL
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('CTL / ATL', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Graficar CTL (azul) y ATL (rojo)
    ax1.plot(pmc_data['date'], pmc_data['CTL'], color='blue', label='Fitness (CTL)')
    ax1.plot(pmc_data['date'], pmc_data['ATL'], color='red', label='Fatiga (ATL)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Crear un segundo eje Y para TSB
    ax2 = ax1.twinx()
    ax2.set_ylabel('TSB (Forma)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Graficar TSB como área
    ax2.fill_between(pmc_data['date'], pmc_data['TSB'], 0, 
                     where=pmc_data['TSB'] >= 0, color='lightgreen', alpha=0.5)
    ax2.fill_between(pmc_data['date'], pmc_data['TSB'], 0, 
                     where=pmc_data['TSB'] < 0, color='lightcoral', alpha=0.5)
    ax2.plot(pmc_data['date'], pmc_data['TSB'], color='green', label='Forma (TSB)')
    
    # Graficar TSS como barras en el fondo
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Mover el segundo eje a la derecha
    ax3.set_ylabel('TSS', color='gray')
    ax3.tick_params(axis='y', labelcolor='gray')
    ax3.bar(pmc_data['date'], pmc_data['TSS'], color='gray', alpha=0.3, width=0.8)
    ax3.set_ylim(0, max(pmc_data['TSS']) * 2 if max(pmc_data['TSS']) > 0 else 100)
    
    # Combinar leyendas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
    
    # Formatear fechas en el eje x
    fig.autofmt_xdate()
    
    plt.title('Performance Management Chart (PMC)')
    plt.tight_layout()
    
    return fig

def format_pace(seconds_per_km):
    """
    Formatea un ritmo en segundos por km a formato min:seg por km.
    
    Args:
        seconds_per_km: Ritmo en segundos por kilómetro
    
    Returns:
        str: Ritmo formateado (min:seg/km)
    """
    if seconds_per_km == 0:
        return "0:00/km"
    
    minutes = int(seconds_per_km // 60)
    seconds = int(seconds_per_km % 60)
    return f"{minutes}:{seconds:02d}/km"

def pace_to_seconds(pace_str):
    """
    Convierte un ritmo formato min:seg/km a segundos por km.
    
    Args:
        pace_str: Ritmo en formato "min:seg/km"
    
    Returns:
        float: Ritmo en segundos por kilómetro
    """
    try:
        if ":" in pace_str:
            minutes, seconds = pace_str.split(":")
            seconds = seconds.split("/")[0]  # Quitar "/km" si está presente
            return int(minutes) * 60 + int(seconds)
        else:
            return float(pace_str)
    except:
        return 0

def plot_weekly_summary(activities_df, figsize=(12, 8)):
    """
    Genera un gráfico de resumen semanal de carga de entrenamiento.
    
    Args:
        activities_df: DataFrame con las actividades
        figsize: Tamaño del gráfico
    
    Returns:
        fig: Objeto de figura de matplotlib
    """
    if activities_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No hay datos suficientes para generar el resumen semanal", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return fig
    
    # Asegurarse de que la fecha es datetime
    activities_df['date'] = pd.to_datetime(activities_df['date'])
    
    # Agregar columna de semana y año
    activities_df['year_week'] = activities_df['date'].dt.isocalendar().year.astype(str) + '-' + \
                               activities_df['date'].dt.isocalendar().week.astype(str).str.zfill(2)
    
    # Agregar actividades por semana y tipo
    weekly_summary = activities_df.groupby(['year_week', 'activity_type'])['TSS'].sum().unstack().fillna(0)
    
    # Preparar datos para el gráfico
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colores por tipo de actividad
    colors = {'cycling': 'blue', 'running': 'green', 'swimming': 'orange', 'other': 'gray'}
    
    # Crear barras apiladas
    bottom = np.zeros(len(weekly_summary))
    
    for activity_type in weekly_summary.columns:
        color = colors.get(activity_type, 'gray')
        ax.bar(weekly_summary.index, weekly_summary[activity_type], bottom=bottom, label=activity_type.title(), color=color)
        bottom += weekly_summary[activity_type].values
    
    # Calcular TSS total por semana
    weekly_total = weekly_summary.sum(axis=1)
    
    # Superponer línea de total
    ax2 = ax.twinx()
    ax2.plot(weekly_summary.index, weekly_total, 'r--', label='TSS Total')
    ax2.set_ylabel('TSS Total', color='red')
    
    # Añadir etiquetas
    ax.set_xlabel('Semana')
    ax.set_ylabel('TSS por tipo de actividad')
    ax.set_title('Resumen semanal de carga de entrenamiento')
    
    # Rotar etiquetas del eje x
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Mostrar leyendas
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    return fig

def generate_sample_data(num_days=90, ftp=250, activities_per_week=4):
    """
    Genera datos de muestra para demostración.
    
    Args:
        num_days: Número de días a generar
        ftp: FTP del atleta
        activities_per_week: Promedio de actividades por semana
    
    Returns:
        DataFrame: DataFrame con actividades de muestra
    """
    today = dt.datetime.now().date()
    start_date = today - dt.timedelta(days=num_days)
    
    data = []
    activity_types = ['cycling', 'running', 'swimming']
    
    # Probabilidad de entrenamiento cada día (mayor en fin de semana)
    day_probabilities = [0.4, 0.4, 0.5, 0.4, 0.5, 0.7, 0.8]  # Lun a Dom
    
    # Crear un periodización simple
    # Fase de base (4 semanas), construcción (4 semanas), especialidad (4 semanas), pico (1 semana), recuperación (1 semana)
    phase_intensity = {
        'base': 0.7,
        'build': 0.85,
        'specialty': 1.0,
        'peak': 1.1,
        'recovery': 0.5
    }
    
    # Asignar fase a cada semana
    current_date = start_date
    current_week = 0
    
    while current_date <= today:
        week_in_cycle = current_week % 14
        
        if week_in_cycle < 4:
            phase = 'base'
        elif week_in_cycle < 8:
            phase = 'build'
        elif week_in_cycle < 12:
            phase = 'specialty'
        elif week_in_cycle < 13:
            phase = 'peak'
        else:
            phase = 'recovery'
        
        # Intensidad base para la semana según la fase
        base_intensity = phase_intensity[phase]
        
        # Generar actividades para cada día de la semana
        for day_offset in range(7):
            day_date = current_date + dt.timedelta(days=day_offset)
            
            if day_date > today:
                break
                
            # Decidir si hay entrenamiento este día
            day_of_week = day_date.weekday()
            if random.random() < day_probabilities[day_of_week] * (activities_per_week / 7):
                # Seleccionar tipo de actividad
                # Más probable que sea ciclismo, luego carrera, luego natación
                activity_type = random.choices(
                    activity_types, 
                    weights=[0.6, 0.3, 0.1], 
                    k=1
                )[0]
                
                # Duración base según el tipo (en minutos)
                if activity_type == 'cycling':
                    base_duration = random.randint(45, 180)
                elif activity_type == 'running':
                    base_duration = random.randint(30, 90)
                else:  # swimming
                    base_duration = random.randint(30, 60)
                
                # Ajustar duración según la fase
                duration_minutes = base_duration * (0.8 + base_intensity * 0.4)
                
                # Generar datos según el tipo de actividad
                if activity_type == 'cycling':
                    # Potencia media como porcentaje del FTP ajustado por fase e intensidad diaria
                    daily_intensity = base_intensity * random.uniform(0.9, 1.1)
                    avg_power = ftp * daily_intensity * random.uniform(0.7, 0.85)
                    
                    # Variabilidad según el tipo de entrenamiento
                    variability = random.uniform(1.0, 1.1)
                    np_value = avg_power * variability
                    
                    # Calcular IF
                    intensity_factor = np_value / ftp
                    
                    # Calcular TSS
                    tss = (duration_minutes * 60 * np_value * intensity_factor) / (ftp * 3600) * 100
                    
                    activity_data = {
                        'date': day_date,
                        'activity_type': activity_type,
                        'duration_seconds': duration_minutes * 60,
                        'avg_power': avg_power,
                        'np': np_value,
                        'intensity_factor': intensity_factor,
                        'TSS': tss,
                        'ftp': ftp,
                        'avg_hr': 110 + random.randint(20, 60),
                        'title': f"Entrenamiento en bicicleta - {day_date.strftime('%d/%m/%Y')}"
                    }
                    
                elif activity_type == 'running':
                    # Ritmo en minutos por km (menor es más rápido)
                    # Asumimos un ritmo umbral de 4:30 min/km (270 segundos)
                    threshold_pace = 270
                    daily_intensity = base_intensity * random.uniform(0.9, 1.1)
                    avg_pace = threshold_pace / daily_intensity
                    
                    # Factor de corrección por desnivel
                    grade_factor = random.uniform(0.95, 1.05)
                    
                    # Calcular rTSS
                    rtss = calculate_running_tss(duration_minutes/60, avg_pace, threshold_pace, grade_factor)
                    
                    activity_data = {
                        'date': day_date,
                        'activity_type': activity_type,
                        'duration_seconds': duration_minutes * 60,
                        'avg_pace': avg_pace,
                        'threshold_pace': threshold_pace,
                        'grade_factor': grade_factor,
                        'TSS': rtss,
                        'avg_hr': 120 + random.randint(20, 50),
                        'title': f"Entrenamiento de carrera - {day_date.strftime('%d/%m/%Y')}"
                    }
                    
                else:  # swimming
                    # Intensidad de natación
                    intensity = base_intensity * random.uniform(0.9, 1.1)
                    
                    # Calcular sTSS
                    stss = calculate_swimming_tss(duration_minutes/60, intensity)
                    
                    activity_data = {
                        'date': day_date,
                        'activity_type': activity_type,
                        'duration_seconds': duration_minutes * 60,
                        'intensity': intensity,
                        'TSS': stss,
                        'avg_hr': 110 + random.randint(20, 40),
                        'title': f"Entrenamiento de natación - {day_date.strftime('%d/%m/%Y')}"
                    }
                
                data.append(activity_data)
        
        current_date += dt.timedelta(days=7)
        current_week += 1
    
    return pd.DataFrame(data)

def generate_zones(ftp=None, lthr=None, threshold_pace=None):
    """
    Genera zonas de entrenamiento basadas en FTP, LTHR o ritmo umbral.
    
    Args:
        ftp: Functional Threshold Power
        lthr: Lactate Threshold Heart Rate
        threshold_pace: Ritmo umbral en segundos por km
    
    Returns:
        dict: Diccionario con zonas de entrenamiento
    """
    zones = {}
    
    # Zonas de potencia (ciclismo)
    if ftp:
        power_zones = {
            'Zone 1 - Recuperación': (0, int(0.55 * ftp)),
            'Zone 2 - Resistencia': (int(0.56 * ftp), int(0.75 * ftp)),
            'Zone 3 - Tempo': (int(0.76 * ftp), int(0.90 * ftp)),
            'Zone 4 - Umbral': (int(0.91 * ftp), int(1.05 * ftp)),
            'Zone 5 - VO2max': (int(1.06 * ftp), int(1.20 * ftp)),
            'Zone 6 - Capacidad anaeróbica': (int(1.21 * ftp), int(1.50 * ftp)),
            'Zone 7 - Potencia neuromuscular': (int(1.51 * ftp), int(2.00 * ftp))
        }
        zones['power'] = power_zones
    
    # Zonas de frecuencia cardíaca
    if lthr:
        hr_zones = {
            'Zone 1 - Recuperación': (0, int(0.81 * lthr)),
            'Zone 2 - Resistencia aeróbica': (int(0.82 * lthr), int(0.88 * lthr)),
            'Zone 3 - Tempo': (int(0.89 * lthr), int(0.93 * lthr)),
            'Zone 4 - Umbral de lactato': (int(0.94 * lthr), int(0.99 * lthr)),
            'Zone 5A - VO2max': (int(1.00 * lthr), int(1.02 * lthr)),
            'Zone 5B - Capacidad anaeróbica': (int(1.03 * lthr), int(1.05 * lthr)),
            'Zone 5C - Potencia anaeróbica': (int(1.06 * lthr), int(1.10 * lthr))
        }
        zones['heart_rate'] = hr_zones
    
    # Zonas de ritmo (carrera)
    if threshold_pace:
        pace_zones = {
            'Zone 1 - Recuperación': (int(threshold_pace * 1.33), int(threshold_pace * 1.20)),
            'Zone 2 - Resistencia aeróbica': (int(threshold_pace * 1.19), int(threshold_pace * 1.11)),
            'Zone 3 - Tempo': (int(threshold_pace * 1.10), int(threshold_pace * 1.05)),
            'Zone 4 - Umbral': (int(threshold_pace * 1.04), int(threshold_pace * 0.99)),
            'Zone 5A - VO2max': (int(threshold_pace * 0.98), int(threshold_pace * 0.95)),
            'Zone 5B - Capacidad anaeróbica': (int(threshold_pace * 0.94), int(threshold_pace * 0.90)),
            'Zone 5C - Velocidad': (int(threshold_pace * 0.89), int(threshold_pace * 0.80))
        }
        zones['pace'] = pace_zones
    
    return zones

def csv_to_dataframe(csv_file):
    """
    Convierte un archivo CSV a DataFrame de pandas.
    
    Args:
        csv_file: Archivo CSV cargado
    
    Returns:
        DataFrame: DataFrame con los datos del CSV
    """
    try:
        if isinstance(csv_file, str):
            # Si es una ruta de archivo
            df = pd.read_csv(csv_file)
        else:
            # Si es un objeto de archivo (por ejemplo, de st.file_uploader)
            df = pd.read_csv(csv_file)
        
        # Convertir columnas de fecha
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Convertir columnas numéricas
        numeric_cols = ['duration_seconds', 'avg_power', 'np', 'intensity_factor', 
                        'TSS', 'ftp', 'avg_hr', 'avg_pace', 'threshold_pace', 
                        'grade_factor', 'intensity']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    except Exception as e:
        print(f"Error al procesar el CSV: {e}")
        return pd.DataFrame()

def export_to_csv(df, filename="training_data.csv"):
    """
    Exporta un DataFrame a un archivo CSV.
    
    Args:
        df: DataFrame a exportar
        filename: Nombre del archivo CSV
    
    Returns:
        str: Ruta del archivo CSV generado
    """
    try:
        df.to_csv(filename, index=False)
        return filename
    except Exception as e:
        print(f"Error al exportar a CSV: {e}")
        return None

def plot_zone_distribution(activities_df, zone_type='power', ftp=None, lthr=None, threshold_pace=None, figsize=(12, 8)):
    """
    Genera un gráfico de distribución de tiempo en zonas.
    
    Args:
        activities_df: DataFrame con las actividades
        zone_type: Tipo de zonas ('power', 'heart_rate', 'pace')
        ftp: FTP del atleta (para zonas de potencia)
        lthr: LTHR del atleta (para zonas de frecuencia cardíaca)
        threshold_pace: Ritmo umbral (para zonas de ritmo)
        figsize: Tamaño del gráfico
    
    Returns:
        fig: Objeto de figura de matplotlib
    """
    if activities_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No hay datos suficientes para analizar zonas", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return fig
    
    # Generar zonas
    zones = generate_zones(ftp, lthr, threshold_pace)
    
    if zone_type not in zones:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No hay datos suficientes para zonas de {zone_type}", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return fig
    
    # Filtrar actividades según el tipo de zona
    if zone_type == 'power':
        filtered_df = activities_df[activities_df['activity_type'] == 'cycling'].copy()
        value_col = 'avg_power'
    elif zone_type == 'heart_rate':
        filtered_df = activities_df.copy()  # Se puede aplicar a todas las actividades
        value_col = 'avg_hr'
    elif zone_type == 'pace':
        filtered_df = activities_df[activities_df['activity_type'] == 'running'].copy()
        value_col = 'avg_pace'
    
    if filtered_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No hay actividades para analizar zonas de {zone_type}", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return fig
    
    # Asignar zona a cada actividad
    zone_assignments = []
    current_zones = zones[zone_type]
    
    for index, row in filtered_df.iterrows():
        value = row[value_col]
        assigned_zone = None
        
        for zone_name, (zone_min, zone_max) in current_zones.items():
            if zone_min <= value <= zone_max:
                assigned_zone = zone_name
                break
        
        if assigned_zone:
            zone_assignments.append(assigned_zone)
        else:
            # Si no cae en ninguna zona, asignamos a la más cercana
            closest_zone = min(current_zones.items(), 
                             key=lambda x: min(abs(value - x[1][0]), abs(value - x[1][1])))
            zone_assignments.append(closest_zone[0])
    
    filtered_df['zone'] = zone_assignments
    
    # Calcular tiempo en cada zona
    zone_time = filtered_df.groupby('zone')['duration_seconds'].sum()
    
    # Ordenar zonas
    zone_order = list(current_zones.keys())
    zone_time = zone_time.reindex(zone_order)
    
    # Convertir a horas
    zone_time = zone_time / 3600
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colores según tipo de zona
    if zone_type == 'power':
        colors = ['gray', 'blue', 'green', 'yellow', 'orange', 'red', 'purple']
    elif zone_type == 'heart_rate':
        colors = ['gray', 'blue', 'green', 'yellow', 'orange', 'red', 'purple']
    elif zone_type == 'pace':
        colors = ['gray', 'blue', 'green', 'yellow', 'orange', 'red', 'purple']
    
    # Crear gráfico de barras
    bars = ax.bar(zone_time.index, zone_time.values, color=colors[:len(zone_time)])
    
    # Añadir etiquetas con horas y porcentaje
    total_time = zone_time.sum()
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_time) * 100 if total_time > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f"{height:.1f}h\n({percentage:.1f}%)",
                ha='center', va='bottom', rotation=0, fontsize=9)
    
    # Ajustar etiquetas y título
    ax.set_xlabel('Zonas de entrenamiento')
    ax.set_ylabel('Tiempo (horas)')
    
    if zone_type == 'power':
        title = 'Distribución del tiempo en zonas de potencia'
    elif zone_type == 'heart_rate':
        title = 'Distribución del tiempo en zonas de frecuencia cardíaca'
    else:
        title = 'Distribución del tiempo en zonas de ritmo'
    
    ax.set_title(title)
    
    # Rotar etiquetas del eje x
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig

def plot_activity_distribution(activities_df, figsize=(10, 6)):
    """
    Genera un gráfico de distribución de actividades por tipo.
    
    Args:
        activities_df: DataFrame con las actividades
        figsize: Tamaño del gráfico
    
    Returns:
        fig: Objeto de figura de matplotlib
    """
    if activities_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No hay datos suficientes para analizar tipos de actividad", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return fig
    
    # Contar actividades por tipo
    activity_counts = activities_df['activity_type'].value_counts()
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colores para los tipos de actividad
    colors = {'cycling': 'blue', 'running': 'green', 'swimming': 'orange'}
    
    # Crear gráfico de pastel
    wedges, texts, autotexts = ax.pie(
        activity_counts.values, 
        labels=activity_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=[colors.get(act, 'gray') for act in activity_counts.index]
    )
    
    # Personalizar texto
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
    
    ax.axis('equal')  # Asegurar que el pastel es circular
    
    plt.title('Distribución de actividades por tipo')
    
    return fig