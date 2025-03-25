
#########################################################
#########################################################
import pandas as pd
import numpy as np

#########################################################
#########################################################
"""
List of Parameters:
    
STANDARD:
Pos.     [1] [2] [3] [4] [5] [6] [7]
pars = c(p*, kg, Tg, kh, Th)          (if initial = FALSE)
pars = c(p*, kg, Tg, kh, Th, qg, qh)  (if initial = TRUE)

FITNESS-DELAY
        [1] [2] [3] [4] [5] [6] [7] [8]
Pars: c(p*, kg, Tg, Tg2, kh, Th)          initial = FALSE
Pars: c(p*, kg, Tg, Tg2, kh, Th, qg, qh)  initial = TRUE

VARIABLE DOSE-RESPONSE
        [1] [2] [3] [4] [5] [6]  [7] [8]
Pars: c(p*, kg, Tg, kh, Th, Th2)          initial = FALSE
Pars: c(p*, kg, Tg, kh, Th, Th2, qg, qh)  initial = TRUE

"""

# FFM Standard - Banister 1975
def standard_objective_ss(pars, loads, perfVals, initial=False, maximise=False):
    
    # Número de mediciones de rendimiento
    nMeasurements = len(perfVals['performance'])

    # Vector de residuales cuadrados
    squared_residuals = np.zeros(nMeasurements)

    # Calcular el residual cuadrado para cada medición
    for n in range(nMeasurements):
        dayT = perfVals['day'][n]  # Día de la medición de rendimiento
        measured = perfVals['performance'][n]  # Valor medido del rendimiento en dayT

        # Subconjunto de cargas hasta el día dayT
        inputSubset = loads.iloc[:dayT]

        if initial:
            # Incluir efectos residuales (componentes iniciales)
            init_fitness = pars[5] * np.exp(-dayT / pars[2])
            init_fatigue = pars[6] * np.exp(-dayT / pars[4])
        else:
            init_fitness = 0
            init_fatigue = 0

        # Calcular el rendimiento modelado en el día dayT
        model = (pars[0] + init_fitness - init_fatigue +
                 pars[1] * np.sum(inputSubset['load'] * np.exp(-(dayT - inputSubset['day']) / pars[2])) -
                 pars[3] * np.sum(inputSubset['load'] * np.exp(-(dayT - inputSubset['day']) / pars[4])))

        # Calcular el residual cuadrado (modelado - medido)^2
        squared_residuals[n] = (model - measured) ** 2

    # Retornar la suma de los residuales cuadrados
    if maximise:
        return -np.sum(squared_residuals)  # Para algoritmos que maximizan por defecto
    else:
        return np.sum(squared_residuals)

def simulate_standard(pars, loads, initialPars=(0, 0), returnObject="all"):
    
    # Longitud de la serie temporal (último día de la columna 'day' de loads)
    seriesLength = loads['day'].iloc[-1]
    
    # Inicialización de vectores en cero para los resultados
    performance = np.zeros(seriesLength)      # Rendimiento modelado
    fitness = np.zeros(seriesLength)          # Fitness modelado
    fatigue = np.zeros(seriesLength)          # Fatiga modelada
    initialFitness = np.zeros(seriesLength)   # Efectos residuales iniciales (fitness)
    initialFatigue = np.zeros(seriesLength)   # Efectos residuales iniciales (fatiga)
    
    # Calcular fitness g(t), fatiga h(t), y performance p(t) para t = 1:seriesLength
    for t in range(1, seriesLength + 1):
        
        # Aislar los datos de carga necesarios para calcular p(t)
        inputSubset = loads[loads['day'] < t]
        
        # Efectos residuales de los componentes iniciales en el tiempo t
        initialFitness[t - 1] = initialPars[0] * np.exp(-(t) / pars[2])
        initialFatigue[t - 1] = initialPars[1] * np.exp(-(t) / pars[3])
        
        # Calcular g(t), h(t), p(t) para el t actual
        fitness[t - 1] = pars[1] * np.sum(inputSubset['load'] * np.exp(- (t - inputSubset['day']) / pars[2]))
        fatigue[t - 1] = pars[3] * np.sum(inputSubset['load'] * np.exp(- (t - inputSubset['day']) / pars[4]))
        performance[t - 1] = pars[0] + fitness[t - 1] - fatigue[t - 1] + initialFitness[t - 1] - initialFatigue[t - 1]
    
    # Salida
    if returnObject == "performance":
        return performance
    elif returnObject == "fitness":
        return fitness
    elif returnObject == "fatigue":
        return fatigue
    elif returnObject == "all":
        return pd.DataFrame({
            "day": np.arange(1, seriesLength + 1),
            "initial_fitness": initialFitness,
            "initial_fatigue": initialFatigue,
            "fitness": fitness,
            "fatigue": fatigue,
            "performance": performance
        })
    
#########################################################
#########################################################

def fitness_delay_objective_ss(pars, loads, perf_vals, initial=False, maximise=False):

    # Número de mediciones de rendimiento
    n_measurements = len(perf_vals['performance'])
    
    # Vector de residuos al cuadrado de longitud igual al número de mediciones de rendimiento
    squared_residuals = np.zeros(n_measurements)
    
    # Para cada medición de rendimiento, calcula (modelado - medido)^2 bajo pars
    for n in range(n_measurements):
        
        day_t = perf_vals['day'][n]               # Día de medición de rendimiento
        measured = perf_vals['performance'][n]    # Valor de rendimiento medido en el día dayT
        
        # Aislamos los datos de carga necesarios para calcular el modelo hasta dayT
        input_subset = loads[loads['day'] <= day_t]
        
        # Efectos residuales de los componentes iniciales
        if initial:
            init_fitness = pars[6] * np.exp(-day_t / pars[2])
            init_fatigue = pars[7] * np.exp(-day_t / pars[5])
        else:
            init_fitness = 0
            init_fatigue = 0
        
        # Modelo p^(dayT) = p* + g(dayT) - h(dayT)
        model = (pars[0] + init_fitness - init_fatigue +
                 pars[1] * np.sum(input_subset['load'] * 
                                (np.exp(-(day_t - input_subset['day']) / pars[2]) -
                                np.exp(-(day_t - input_subset['day']) / pars[3]))) - 
                 pars[4] * np.sum(input_subset['load'] * 
                                np.exp(-(day_t - input_subset['day']) / pars[5])))
        
        # Calcula el valor residual al cuadrado (modelado - medido)^2
        squared_residuals[n] = (model - measured) ** 2
    
    # Devuelve la suma de los residuos al cuadrado
    if maximise:
        return -np.sum(squared_residuals)
    else:
        return np.sum(squared_residuals)

def simulate_fitness_delay(pars, loads, initialPars=(0, 0), returnObject="all"):
    # Definir la longitud de la serie
    seriesLength = loads['day'].iloc[-1]  # Último día en loads

    # Inicializar los vectores en ceros
    performance = np.zeros(seriesLength)     # Rendimiento del modelo
    fitness = np.zeros(seriesLength)         # Condición física del modelo
    fatigue = np.zeros(seriesLength)         # Fatiga del modelo
    initialFitness = np.zeros(seriesLength)  # Efectos residuales (componente inicial)
    initialFatigue = np.zeros(seriesLength)

    # Calcular g(t), h(t) y p(t) para t = 1:seriesLength
    for t in range(1, seriesLength + 1):
        # Aislar los datos de carga requeridos para calcular p(t)
        # (cargas desde el día 0 hasta el día t-1)
        inputSubset = loads[loads['day'] < t]

        # Efectos residuales de los componentes iniciales en el punto temporal t
        initialFitness[t-1] = initialPars[0] * np.exp(-(t) / pars[2])
        initialFatigue[t-1] = initialPars[1] * np.exp(-(t) / pars[5])

        # Calcular g(t), h(t), p(t) para el tiempo actual t
        fitness[t-1] = pars[1] * np.sum(inputSubset['load'] * (np.exp(-(t - inputSubset['day']) / pars[2]) -
                                                            np.exp(-(t - inputSubset['day']) / pars[3])))
        fatigue[t-1] = pars[4] * np.sum(inputSubset['load'] * np.exp(-(t - inputSubset['day']) / pars[5]))
        performance[t-1] = pars[0] + fitness[t-1] - fatigue[t-1] + initialFitness[t-1] - initialFatigue[t-1]

    # Salida
    if returnObject == "performance":
        return performance
    elif returnObject == "fitness":
        return fitness
    elif returnObject == "fatigue":
        return fatigue
    elif returnObject == "all":
        # Devolver DataFrame con toda la información
        return pd.DataFrame({
            "day": np.arange(1, seriesLength + 1),
            "initial_fitness": initialFitness,
            "initial_fatigue": initialFatigue,
            "fitness": fitness,
            "fatigue": fatigue,
            "performance": performance
        })
    
# Ejemplo de uso:
# mockParameters = [100, 1, 22.5, 1.2, 1.5, 8]
# loads = pd.DataFrame({"day": range(0, 101), "load": [...]})
# result = simulate_fitness_delay(mockParameters, loads, returnObject="all")

#########################################################
#########################################################

def vdr_objective_ss(pars, loads, perf_vals, initial=False, maximise=False):
    
    # Number of performance measurements
    n_measurements = len(perf_vals['performance'])
    
    # Zeroed vector of squared residuals
    squared_residuals = np.zeros(n_measurements)
    
    # Loop through each performance measurement
    for n in range(n_measurements):
        day_t = perf_vals['day'].iloc[n]  # Day of measured performance
        measured = perf_vals['performance'].iloc[n]  # Measured performance value on day_t
        
        # Isolate the required load data up to day_t (t=0 to day_t-1)
        input_subset = loads[loads['day'] <= day_t]
        
        # Initial components
        if initial:
            init_fitness = pars[6] * np.exp(-day_t / pars[2])  # qg * exp(-day_t / Tg)
            init_fatigue = pars[7] * np.exp(-day_t / pars[5])  # qh * exp(-day_t / Th2)
        else:
            init_fitness = 0
            init_fatigue = 0
        
        # Zeroed vector for variable gain term kh2
        kh2 = np.zeros(len(input_subset))
        
        # Calculate the variable gain term kh2(i) for i=0,1,2,...,day_t-1
        for i in range(len(input_subset)):
            kh2[i] = np.sum(input_subset['load'][:i+1].values *
                            np.exp(-(input_subset['day'].iloc[i] - input_subset['day'][:i+1].values) / pars[5]))
        
        # Compute modelled performance on day_t under pars
        model = (pars[0] + init_fitness - init_fatigue +
                 pars[1] * np.sum(input_subset['load'].values * 
                                np.exp(-(day_t - input_subset['day'].values) / pars[2])) -
                 pars[3] * np.sum(kh2 * input_subset['load'].values *
                                np.exp(-(day_t - input_subset['day'].values) / pars[4])))
        
        # Compute the squared residual value (model - measured)^2
        squared_residuals[n] = (model - measured) ** 2
    
    # Return the result based on maximise flag
    if not maximise:
        return np.sum(squared_residuals)
    else:
        return -np.sum(squared_residuals)


def simulate_vdr(pars, loads, initialPars=(0, 0), returnObject="all"):
    # Definir la longitud de la serie
    seriesLength = loads['day'].iloc[-1]  # Último día en loads

    # Inicializar vectores de ceros
    performance = np.zeros(seriesLength)     # Rendimiento del modelo
    fitness = np.zeros(seriesLength)         # Condición física del modelo
    fatigue = np.zeros(seriesLength)         # Fatiga del modelo
    initialFitness = np.zeros(seriesLength)  # Efectos residuales (componente inicial)
    initialFatigue = np.zeros(seriesLength)
    kh2Dat = np.full((seriesLength, seriesLength), np.nan)  # Matriz para almacenar valores kh2

    # Calcular g(t), h(t) y p(t) para t = 1:seriesLength
    for t in range(1, seriesLength + 1):
        # Aislar los datos de carga requeridos para calcular p(t)
        inputSubset = loads[loads['day'] < t]

        # Efectos residuales de los componentes iniciales en el punto temporal t
        initialFitness[t-1] = initialPars[0] * np.exp(-(t) / pars[2])
        initialFatigue[t-1] = initialPars[1] * np.exp(-(t) / pars[3])

        # Inicializar el vector kh2 en ceros para el término de ganancia variable
        kh2 = np.zeros(t)

        # Calcular el término de ganancia variable kh2(i) para i=0,1,...,t-1 (Recursivo)
        for i in range(1, t+1):
            kh2[i-1] = np.sum(inputSubset['load'].iloc[:i] * 
                            np.exp(-((inputSubset['day'].iloc[i-1] - inputSubset['day'].iloc[:i]) / pars[5])))

        # Guardar los valores kh2(i) para i = 0 a t-1 en kh2Dat
        kh2Dat[:t, t-1] = kh2

        # Calcular g(t), h(t), p(t) para el tiempo actual t
        fitness[t-1] = pars[1] * np.sum(inputSubset['load'] * np.exp(-(t - inputSubset['day']) / pars[2]))
        fatigue[t-1] = pars[3] * np.sum(kh2 * np.exp(-(t - inputSubset['day']) / pars[4]))
        performance[t-1] = pars[0] + fitness[t-1] - fatigue[t-1] + initialFitness[t-1] - initialFatigue[t-1]

    # Salida
    if returnObject == "performance":
        return performance
    elif returnObject == "fitness":
        return fitness
    elif returnObject == "fatigue":
        return fatigue
    elif returnObject == "all":
        return pd.DataFrame({
            "day": np.arange(1, seriesLength + 1),
            "initial_fitness": initialFitness,
            "initial_fatigue": initialFatigue,
            "fitness": fitness,
            "fatigue": fatigue,
            "kh2_t": kh2Dat[:, seriesLength-1],
            "performance": performance
        })

# Ejemplo de uso:
# mockParameters = [100, 1, 22.5, 1.2, 1.5, 8]
# loads = pd.DataFrame({"day": range(0, 101), "load": [...]})
# result = simulate_vdr(mockParameters, loads, returnObject="all")
