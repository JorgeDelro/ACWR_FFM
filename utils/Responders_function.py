import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def analyze_responders_all_groups(group, pre, post, control_label='control',
                       confidence_level=0.75, swc_method='0.2sd', swc_value=None, 
                       direction='positive', variable_name=None, use_percent=False,
                       show_confidence_intervals=True):
    """
    Analiza responders y non-responders para todos los grupos en los datos.
    
    Parámetros:
    -----------
    group : array-like
        Vector que indica el grupo al que pertenece cada individuo
    pre : array-like
        Valores pre-intervención de todos los individuos
    post : array-like
        Valores post-intervención de todos los individuos
    control_label : str, opcional (default='control')
        Etiqueta utilizada para identificar al grupo control (para estimar TE)
    confidence_level : float, opcional (default=0.75)
        Nivel de confianza para los intervalos (por ejemplo: 0.5, 0.75, 0.9, 0.95)
    swc_method : str, opcional (default='0.2sd')
        Método para calcular el Smallest Worthwhile Change (SWC):
        '0.2sd' - 0.2 x desviación estándar baseline
        'custom' - valor personalizado proporcionado en swc_value
    swc_value : float, opcional
        Valor personalizado del SWC (requerido si swc_method='custom')
    direction : str, opcional (default='positive')
        Dirección esperada del cambio positivo: 'positive' o 'negative'
    variable_name : str, opcional
        Nombre de la variable analizada para etiquetas en el gráfico
    use_percent : bool, opcional (default=False)
        Si True, muestra los cambios como porcentaje del valor inicial.
        Si False, muestra los cambios en las unidades originales.
    show_confidence_intervals : bool, opcional (default=True)
        Si True, muestra los intervalos de confianza en el gráfico.
        
    Retorna:
    --------
    dict
        Diccionario con los resultados del análisis para cada grupo
    """
    # Convertir inputs a arrays numpy
    group = np.array(group, dtype=object)
    pre = np.array(pre, dtype=float)
    post = np.array(post, dtype=float)
    
    # Obtener grupos únicos
    grupos_unicos = np.unique(group)
    
    # Calcular el Error Típico (TE) si hay grupo control
    te_estimado = None
    if control_label is not None and control_label in grupos_unicos:
        mask_control = np.array([str(x) == str(control_label) for x in group])
        con_pre = pre[mask_control]
        con_post = post[mask_control]
        con_changes = con_post - con_pre
        
        if use_percent:
            con_changes_percent = ((con_post - con_pre) / con_pre) * 100
            sd_diff = np.std(con_changes_percent, ddof=1)
        else:
            sd_diff = np.std(con_changes, ddof=1)
            
        te_estimado = sd_diff / np.sqrt(2)
        print(f"Error Típico (TE) estimado del grupo control: {te_estimado:.2f}")
    
    # Diccionario para almacenar los resultados por grupo
    resultados_por_grupo = {}
    figuras_por_grupo = {}
    
    # Procesar cada grupo
    for grupo_actual in grupos_unicos:
        print(f"\nAnalizando grupo: {grupo_actual}")
        
        # Filtrar datos del grupo actual
        mask_grupo = (group == grupo_actual)
        grupo_pre = pre[mask_grupo]
        grupo_post = post[mask_grupo]
        
        # Calcular cambios absolutos y porcentuales
        grupo_changes = grupo_post - grupo_pre
        grupo_changes_percent = ((grupo_post - grupo_pre) / grupo_pre) * 100
        
        # Usar el TE estimado del grupo control o calcular uno específico para este grupo
        if te_estimado is not None and str(grupo_actual) != str(control_label):
            te = te_estimado
            print(f"Usando TE del grupo control: {te:.2f}")
        else:
            # Si no hay TE disponible, estimar uno
            if use_percent:
                te = 5.0  # Valor por defecto para porcentajes
            else:
                te = np.mean(grupo_pre) * 0.05  # 5% del valor medio
                
            print(f"Estimando TE para este grupo: {te:.2f}")
        
        # Calcular el Smallest Worthwhile Change (SWC)
        if swc_method == '0.2sd':
            if use_percent:
                # Para cambios porcentuales, calcular SWC como 0.2 * SD de los valores pre convertidos a porcentaje
                swc_percent = 0.2 * np.std(grupo_pre, ddof=1) / np.mean(grupo_pre) * 100
                swc = swc_percent
            else:
                swc = 0.2 * np.std(grupo_pre, ddof=1)
        elif swc_method == 'custom' and swc_value is not None:
            if use_percent and not isinstance(swc_value, str):
                # Convertir el valor personalizado a porcentaje si es necesario
                swc = swc_value / np.mean(grupo_pre) * 100 if not isinstance(swc_value, str) else swc_value
            else:
                swc = swc_value
        else:
            raise ValueError("swc_method debe ser '0.2sd' o 'custom'. Si es 'custom', swc_value debe proporcionarse.")
        
        # Ajustar dirección del SWC
        if direction == 'negative':
            swc = -abs(swc)
        else:
            swc = abs(swc)
        
        # Determinar el multiplicador para el intervalo de confianza
        if 0.95 <= confidence_level < 0.99:
            ci_multiplier = 1.96  # Aproximadamente un 95% CI
        elif 0.9 <= confidence_level < 0.95:
            ci_multiplier = 1.64  # Aproximadamente un 90% CI
        elif 0.8 <= confidence_level < 0.9:
            ci_multiplier = 1.28  # Aproximadamente un 80% CI
        elif 0.75 <= confidence_level < 0.8:
            ci_multiplier = 1.15  # Aproximadamente un 75% CI
        elif 0.7 <= confidence_level < 0.75:
            ci_multiplier = 1.04  # Aproximadamente un 70% CI
        elif 0.6 <= confidence_level < 0.7:
            ci_multiplier = 0.84  # Aproximadamente un 60% CI
        elif 0.5 <= confidence_level < 0.6:
            ci_multiplier = 0.67  # Aproximadamente un 50% CI
        else:
            raise ValueError("confidence_level debe estar entre 0.5 y 0.99")
        
        # Seleccionar los valores de cambio según la opción use_percent
        values_change = grupo_changes_percent if use_percent else grupo_changes
        
        # Calcular intervalos de confianza para los cambios (CI)
        ci_half_width = ci_multiplier * np.sqrt(2) * te
        change_ci_lower = values_change - ci_half_width
        change_ci_upper = values_change + ci_half_width
        
        # Clasificar a los individuos como responders o non-responders
        if direction == 'positive':
            classification = []
            for lower, upper in zip(change_ci_lower, change_ci_upper):
                if lower > swc:
                    classification.append('Responder')
                elif upper < 0:
                    classification.append('Adverse Responder')
                elif upper < swc:
                    classification.append('Non-Responder')
                else:
                    classification.append('Uncertain')
        else:  # direction == 'negative'
            classification = []
            for lower, upper in zip(change_ci_lower, change_ci_upper):
                if upper < swc:
                    classification.append('Responder')
                elif lower > 0:
                    classification.append('Adverse Responder')
                elif lower > swc:
                    classification.append('Non-Responder')
                else:
                    classification.append('Uncertain')
        
        # Simplificar clasificación para el gráfico (solo Responders y Non-Responders)
        classification_simple = []
        for cat in classification:
            if cat == 'Responder':
                classification_simple.append('Responder')
            else:
                classification_simple.append('Non-Responder')
        
        # Calcular proporción de responders
        prop_responders = classification.count('Responder') / len(classification)
        n_responders = classification.count('Responder')
        n_non_responders = len(classification) - n_responders
        
        # Crear un DataFrame con los resultados individuales
        results_df = pd.DataFrame({
            'Pre': grupo_pre,
            'Post': grupo_post,
            'Change': grupo_changes,
            'Change_Percent': grupo_changes_percent,
            'CI_Lower': change_ci_lower,
            'CI_Upper': change_ci_upper,
            'Classification': classification,
            'Classification_Simple': classification_simple
        })
        
        # Ordenar el DataFrame por cambio según la dirección esperada
        sort_column = 'Change_Percent' if use_percent else 'Change'
        if direction == 'positive':
            results_df = results_df.sort_values(sort_column)
        else:
            results_df = results_df.sort_values(sort_column, ascending=False)
        
        # Crear la figura similar a la mostrada en la imagen
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Configurar el eje X con los participantes
        n_participants = len(results_df)
        x_pos = np.arange(n_participants)
        
        # Valores a graficar (absolutos o porcentuales)
        values_to_plot = results_df['Change_Percent'] if use_percent else results_df['Change']
        
        # Colorear barras según la clasificación
        colors = []
        for cat in results_df['Classification_Simple']:
            if cat == 'Responder':
                colors.append('black')
            else:
                colors.append('white')
        
        # Crear barras
        bars = ax.bar(x_pos, values_to_plot, color=colors, edgecolor='black')
        
        # Añadir barras de error o áreas sombreadas para intervalos de confianza
        if show_confidence_intervals:
            # Usar barras de error en rojo
            ax.errorbar(
                x_pos, 
                values_to_plot, 
                yerr=ci_half_width, 
                fmt='none', 
                ecolor='red', 
                elinewidth=1, 
                capsize=3,
                capthick=1
            )
        
        # Añadir valores encima de las barras
        for i, bar in enumerate(bars):
            value = values_to_plot.iloc[i]
            if value >= 0:
                va = 'bottom'  # Alineación vertical para valores positivos
                y_pos = value + 0.5  # Posición Y ajustada para valores positivos
            else:
                va = 'top'  # Alineación vertical para valores negativos
                y_pos = value - 0.5  # Posición Y ajustada para valores negativos
            
            # Formato diferente para valores absolutos y porcentuales
            if use_percent:
                ax.text(bar.get_x() + bar.get_width()/2, y_pos, 
                       f'{value:.1f}', ha='center', va=va, fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, y_pos, 
                       f'{value:.2f}', ha='center', va=va, fontsize=8)
        
        # Añadir línea horizontal para el SWC
        ax.axhline(y=swc, color='blue', linestyle='-', linewidth=1.5)
        
        # Añadir etiquetas y título
        variable_label = variable_name if variable_name else "Variable"
        ax.set_xlabel('Participants')
        
        if use_percent:
            ax.set_ylabel(f'Changes in {variable_label} (%)')
        else:
            ax.set_ylabel(f'Changes in {variable_label}')
        
        # Título con número y porcentaje de responders y non-responders
        title = f"Responders: {n_responders}/{n_participants} ({n_responders/n_participants*100:.1f}%), "
        title += f"Non-Responders: {n_non_responders}/{n_participants} ({n_non_responders/n_participants*100:.1f}%)"
        ax.set_title(title)
        
        # Leyenda
        responder_patch = mpatches.Patch(color='black', label='Responders')
        non_responder_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='Non-Responders')
        
        legend_handles = [responder_patch, non_responder_patch]
        if show_confidence_intervals:
            ci_patch = mpatches.Patch(color='red', alpha=0.3, label=f'{int(confidence_level*100)}% CI')
            legend_handles.append(ci_patch)
        
        ax.legend(handles=legend_handles, loc='upper left')
        
        # Quitar etiquetas del eje X
        ax.set_xticks([])
        
        # Ajustar el tamaño de la figura
        plt.tight_layout()
        
        # Guardar resultados para este grupo
        resultados_por_grupo[grupo_actual] = {
            'error_tipico': te,
            'swc': swc,
            'resultados_individuales': results_df,
            'proporcion_responders': prop_responders,
            'n_responders': n_responders,
            'n_non_responders': n_non_responders,
            'use_percent': use_percent,
            'confidence_level': confidence_level,
            'ci_half_width': ci_half_width
        }
        
        # Guardar la figura
        figuras_por_grupo[grupo_actual] = fig
    
    return {
        'resultados_por_grupo': resultados_por_grupo,
        'figuras': figuras_por_grupo
    }

def get_table_download_link(df, filename="data.csv", link_text="Descargar resultados en CSV"):
    """
    Genera un enlace para descargar un DataFrame como archivo CSV
    """
    import base64
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def get_figure_download_link(fig, filename="figure.png", link_text="Descargar gráfico"):
    """
    Genera un enlace para descargar una figura como imagen PNG
    """
    import base64
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{link_text}</a>'
    return href