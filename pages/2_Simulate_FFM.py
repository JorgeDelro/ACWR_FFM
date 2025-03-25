# IMPORTS
import streamlit as st
import pandas as pd
# import numpy as np
import plotly.graph_objs as go
import utils.ffm_functions

# App header
st.markdown("<div class='main-header'>Simulate Data based on Fitness-Fatigue Model</div>", unsafe_allow_html=True)

# LOAD TRAINING LOAD FOR SIMULATION
@st.cache_data
def load_data(path: str):
    data = pd.read_csv(path, sep = ";", decimal=",")
    return data

# PLOT FITTED MODEL
def plot_model(model_fitted, loads):
    # Subset para reducir el número de puntos de datos (cada 3 días)
    model_fitted_subset = pd.DataFrame({
        "day": model_fitted['day'][::3],
        "fitness": model_fitted['fitness'][::3],
        "fatigue": model_fitted['fatigue'][::3],
        "performance": model_fitted['performance'][::3]
    })

    # Trazo de performance
    performance_trace = go.Scatter(
        x=model_fitted_subset['day'],
        y=model_fitted_subset['performance'],
        mode='lines+markers',
        name="Performance [a.u]",
        line=dict(color='#ff79c6', width=3),
        marker=dict(size=8, color='#ff79c6', line=dict(width=1, color='white'))
    )

    # Trazo de carga de entrenamiento
    load_trace = go.Bar(
        x=loads['day'],
        y=loads['load'],
        name="Training load [a.u]",
        marker_color='orange',
        opacity=0.6,
        yaxis="y2"  # Eje secundario
    )

    # Trazo de fatigue con color azul claro
    fatigue_trace = go.Scatter(
        x=model_fitted_subset['day'],
        y=model_fitted_subset['fatigue'],
        mode='lines',
        name="Fatigue",
        line=dict(color='#8be9fd', width=2, dash='dash'),  # Azul claro
        yaxis="y2"
    )

    # Trazo de fitness con color verde claro
    fitness_trace = go.Scatter(
        x=model_fitted_subset['day'],
        y=model_fitted_subset['fitness'],
        mode='lines',
        name="Fitness",
        line=dict(color='#50fa7b', width=2, dash='dot'),  # Verde claro
        yaxis="y2"
    )

    # Configuración de diseño
    layout = go.Layout(
        xaxis=dict(
            title="Day",
            color="#f8f8f2",
            gridcolor="#44475a",
            zerolinecolor="#44475a",
            tickmode="linear",  
            dtick=10
        ),
        yaxis=dict(
            title="Performance [a.u]",
            color="#f8f8f2",
            gridcolor="#44475a",
            zerolinecolor="#44475a",
        ),
        yaxis2=dict(
            title="Training load [a.u]",
            overlaying='y',
            side='right',
            color="#f8f8f2",
            gridcolor="#44475a",
            zerolinecolor="#44475a",
        ),
        legend=dict(
            x=0.1,  # Mover la leyenda
            y=1.25,
            bgcolor="#44475a",  # Fondo oscuro para contraste
            bordercolor="#f8f8f2",
            font=dict(color="#f8f8f2"),
        ),
        plot_bgcolor="#282a36",  # Fondo del gráfico
        paper_bgcolor="#282a36",  # Fondo alrededor del gráfico
    )

    # Crear las figuras
    fig_1 = go.Figure(data=[performance_trace, load_trace], layout=layout)
    fig_2 = go.Figure(data=[performance_trace, fatigue_trace, fitness_trace], layout=layout)

    return [fig_1, fig_2]

model_parameters = None
model_plot = None

# APP
with st.sidebar:

    st.header("Upload Training load")

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is None:
        st.stop()

    df = load_data(uploaded_file)

    st.divider()

    # st.header("Options")
    
    option_model = st.selectbox(
    'Select Model',
    ['Standard', 'Fitness-delay', 'Variable-dose'])

        # Parameters
        # Standard
        # p* = p_init = baseline fitness - 100
        # K_g = magnitude of fitness - 1
        # K_h = magnitude of fatigue - 22.5
        # T_g = decay of fitness - 1.2
        # T_h = decay of fatigue - 8

    if 'Standard' in option_model:
        
        p_init = st.number_input(
            label="Baseline Fitness",
            value=100,
            help="help_txt")

        K_g = st.number_input(
            label="Magnitude Fitness",
            value=1,
            help="help_txt")
        
        T_g = st.number_input(
            label="Decay Fitness",
            value=22.5,
            help="help_txt")
        
        K_h = st.number_input(
            label="Magnitude Fatigue",
            value=1.2,
            help="help_txt")
        
        T_h = st.number_input(
            label="Decay Fatigue",
            value=8,
            help="help_txt")

        if st.button("Plot"):
            model_parameters = utils.ffm_functions.simulate_standard(
                pars=[float(p_init), float(K_g), float(T_g), float(K_h), float(T_h)],
                loads=df,
                returnObject="all")
            model_plot = plot_model(
                model_fitted=model_parameters,
                loads=df
            )

    if 'Fitness-delay' in option_model:
        
        p_init = st.number_input(
            label="Baseline Fitness",
            value=100,
            help="help_txt")

        K_g = st.number_input(
            label="Magnitude Fitness",
            value=1,
            help="help_txt")
        
        T_g = st.number_input(
            label="Decay Fitness",
            value=30,
            help="help_txt")
        
        T_g2 = st.number_input(
            label="Fitness delay",
            value=8,
            help="help_txt")
        
        K_h = st.number_input(
            label="Magnitude Fatigue",
            value=1,
            help="help_txt")
        
        T_h = st.number_input(
            label="Decay Fatigue",
            value=12,
            help="help_txt")

        if st.button("Plot"):
            model_parameters = utils.ffm_functions.simulate_fitness_delay(
                pars=[float(p_init), float(K_g), float(T_g), float(T_g2), float(K_h), float(T_h)],
                loads=df,
                returnObject="all")
            model_plot = plot_model(
                model_fitted=model_parameters,
                loads=df
            )

    if 'Variable-dose' in option_model:
        
        p_init = st.number_input(
            label="Baseline Fitness",
            value=90,
            help="help_txt")

        K_g = st.number_input(
            label="Magnitude Fitness",
            value=0.8,
            help="help_txt")
        
        T_g = st.number_input(
            label="Decay Fitness",
            value=26,
            help="help_txt")
        
        K_h = st.number_input(
            label="Magnitude Fatigue",
            value=1.5,
            help="help_txt")
        
        T_h = st.number_input(
            label="Decay Fatigue",
            value=11,
            help="help_txt")
        
        T_h2 = st.number_input(
            label="Decay gain fatigue",
            value=0.65,
            help="help_txt")

        if st.button("Plot"):
            model_parameters = utils.ffm_functions.simulate_fitness_delay(
                pars=[float(p_init), float(K_g), float(T_g), float(K_h), float(T_h), float(T_h2)],
                loads=df,
                returnObject="all")
            model_plot = plot_model(
                model_fitted=model_parameters,
                loads=df
            )


with st.expander("Data preview"):
    st.dataframe(df)

if model_parameters is not None:
    with st.expander("Model preview"):
        st.dataframe(model_parameters)
        
        # Convertir el dataframe a CSV para descarga
        csv = model_parameters.to_csv(index=False)
        
        # Añadir botón de descarga
        st.download_button(
            label="Download Model Parameters",
            data=csv,
            file_name="model_parameters.csv",
            mime="text/csv",
        )

if model_plot is not None:
    st.plotly_chart(model_plot[0], use_container_width=True)

if model_plot is not None:
    st.plotly_chart(model_plot[1], use_container_width=True)
