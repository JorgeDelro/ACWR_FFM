import streamlit as st
import pandas as pd
import scipy as sp
import time
from scipy.optimize import minimize
import utils.ffm_functions

st.markdown("<div class='main-header'>Fit Fitness-Fatigue Models</div>", unsafe_allow_html=True)

# LOAD TRAINING LOAD FOR SIMULATION
@st.cache_data
def load_data(path: str):
    data = pd.read_csv(path, sep=";", decimal=",")
    return data

# Función de ajuste del modelo usando la optimización BFGS
def fit_standard_model(start_vals, loads, performances):
    result = minimize(utils.ffm_functions.standard_objective_ss,
                    x0=start_vals,
                    args=(loads, performances),
                    method='BFGS')
    return result

def fit_delay_model(start_vals, loads, performances):
    result = minimize(utils.ffm_functions.fitness_delay_objective_ss,
                    x0=start_vals,
                    args=(loads, performances),
                    method='BFGS')
    return result

def fit_vdr_model(start_vals, loads, performances):
    result = minimize(utils.ffm_functions.vdr_objective_ss,
                    x0=start_vals,
                    args=(loads, performances),
                    method='BFGS')
    return result

def display_results_with_dataframe(parameters, model_name):
    if model_name == "Standard":
        param_labels = ["Fitness Baseline", "Fitness Magnitude", "Fitness Decay", "Fatigue Magnitude", "Fatigue Decay"]
    elif model_name == "Fitness-delay":
        param_labels = ["Fitness Baseline", "Fitness Magnitude", "Fitness Decay", "Fitness Delay", "Fatigue Magnitude", "Fatigue Decay"]
    elif model_name == "Variable dose-response":
        param_labels = ["Fitness Baseline", "Fitness Magnitude", "Fitness Decay", "Fatigue Magnitude", "Fatigue Decay", "Variable Gain"]
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if len(parameters) != len(param_labels):
        raise ValueError(f"Mismatch between parameters ({len(parameters)}) and labels ({len(param_labels)}).")

    df = pd.DataFrame({
        "Parameter": param_labels,
        "Value": [round(v, 4) for v in parameters]
    })
    st.dataframe(df.style.set_properties(**{
        'background-color': '#44475a',
        'color': '#f8f8f2',
        'text-align': 'center'
    }))

# Inicialización del estado global
if "fitting" not in st.session_state:
    st.session_state.fitting = {
        "df_loads": None,
        "df_performances": None,
        "selected_model": "Standard",
        "results": {
            "Standard": None,
            "Fitness-delay": None,
            "Variable dose-response": None
        }
    }

# Sidebar
with st.sidebar:
    st.header("Upload Training Loads")
    loads_file = st.file_uploader("Choose a file", key="loads")

    if loads_file:
        st.session_state.fitting["df_loads"] = load_data(loads_file)

    st.divider()

    st.header("Upload Performances")
    performances_file = st.file_uploader("Choose a file", key="performances")

    if performances_file:
        st.session_state.fitting["df_performances"] = load_data(performances_file)

    st.divider()

    option_model = st.selectbox(
        "Select Model",
        ["Standard", "Fitness-delay", "Variable dose-response"],
        index=["Standard", "Fitness-delay", "Variable dose-response"].index(st.session_state.fitting["selected_model"])
    )
    st.session_state.fitting["selected_model"] = option_model

    st.divider()

    # Input de parámetros iniciales
    st.header("Initial Values")
    if option_model == "Standard":
        p_init = st.number_input("Fitness Baseline", value=90.0)
        K_g = st.number_input("Fitness Magnitude", value=0.8)
        T_g = st.number_input("Fitness Decay", value=26.0)
        K_h = st.number_input("Fatigue Magnitude", value=1.5)
        T_h = st.number_input("Fatigue Decay", value=11.0)

        if st.button("Fit Standard Model"):
            with st.spinner("Fitting the Standard model, please wait..."):
                time.sleep(5)
                st.session_state.fitting["results"]["Standard"] = fit_standard_model(
                    [p_init, K_g, T_g, K_h, T_h],
                    st.session_state.fitting["df_loads"],
                    st.session_state.fitting["df_performances"]
                )

    elif option_model == "Fitness-delay":
        p_init = st.number_input("Fitness Baseline", value=90.0)
        K_g = st.number_input("Fitness Magnitude", value=0.8)
        T_g = st.number_input("Fitness Decay", value=26.0)
        T_g2 = st.number_input("Fitness Delay", value=8.0)
        K_h = st.number_input("Fatigue Magnitude", value=1.5)
        T_h = st.number_input("Fatigue Decay", value=11.0)

        if st.button("Fit Fitness-delay Model"):
            with st.spinner("Fitting the Fitness-delay model, please wait..."):
                time.sleep(5)
                st.session_state.fitting["results"]["Fitness-delay"] = fit_delay_model(
                    [p_init, K_g, T_g, T_g2, K_h, T_h],
                    st.session_state.fitting["df_loads"],
                    st.session_state.fitting["df_performances"]
                )

    elif option_model == "Variable dose-response":
        p_init = st.number_input("Fitness Baseline", value=90.0)
        K_g = st.number_input("Fitness Magnitude", value=0.8)
        T_g = st.number_input("Fitness Decay", value=26.0)
        K_h = st.number_input("Fatigue Magnitude", value=1.5)
        T_h = st.number_input("Fatigue Decay", value=11.0)
        T_h2 = st.number_input("Variable Gain", value=0.65)

        if st.button("Fit Variable Dose-Response Model"):
            with st.spinner("Fitting the Variable dose-response model, please wait..."):
                time.sleep(5)
                st.session_state.fitting["results"]["Variable dose-response"] = fit_vdr_model(
                    [p_init, K_g, T_g, K_h, T_h, T_h2],
                    st.session_state.fitting["df_loads"],
                    st.session_state.fitting["df_performances"]
                )

# Main content
st.header("Model Fitting")

# Mostrar datos cargados
if st.session_state.fitting["df_loads"] is not None:
    st.subheader("Training Loads")
    st.dataframe(st.session_state.fitting["df_loads"])

if st.session_state.fitting["df_performances"] is not None:
    st.subheader("Performances")
    st.dataframe(st.session_state.fitting["df_performances"])

# Mostrar resultados
results = st.session_state.fitting["results"][option_model]
if results is not None:
    st.success(f"Model fitting for {option_model} completed!")
    display_results_with_dataframe(results.x, model_name=option_model)