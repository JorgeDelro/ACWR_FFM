import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
from datetime import datetime

def training_blocks(training_dates, actual_TL, diff_dates):
    """
    Calculate training blocks based on calendar days
    
    Parameters:
    -----------
    training_dates : list or pd.Series
        Dates of training sessions
    actual_TL : int
        Current position/index in the training list
    diff_dates : int
        Number of days to look back
    
    Returns:
    --------
    dict
        previous_TL: index of the first session in the block
        n_sessions: number of sessions in the block
    """
    # Get current date
    current_date = training_dates[actual_TL-1]
    
    # Calculate date that is diff_dates days before
    target_date = current_date - pd.Timedelta(days=diff_dates)
    
    # Find the first session date that is >= target_date
    previous_sessions = [i for i, date in enumerate(training_dates[:actual_TL], 1) 
                         if date >= target_date]
    
    if previous_sessions:
        previous_TL = min(previous_sessions)
    else:
        previous_TL = 1
    
    # Number of sessions in the block
    n_sessions = actual_TL - previous_TL + 1
    
    return {"previous_TL": previous_TL, "n_sessions": n_sessions}

def RAC(TL, weeks, training_dates):
    """
    Calculate Rolling Average Coupled (RAC), Acute, and Acute:Chronic Workload Ratio
    
    Parameters:
    -----------
    TL : list or array
        Training Load values
    weeks : list or array
        Week numbers for each training session
    training_dates : list or array
        Dates of each training session (should be pandas datetime objects)
    
    Returns:
    --------
    dict
        RAC_acute: Acute rolling average of training load
        RAC_chronic: Chronic rolling average of training load
        RAC_ACWR: Acute:Chronic Workload Ratio
    """
    # Convert inputs to appropriate types if needed
    TL = np.array(TL)
    weeks = np.array(weeks)
    training_dates = pd.to_datetime(training_dates) if not isinstance(training_dates[0], pd.Timestamp) else training_dates
    
    # Count number of sessions per week
    sessions_week = pd.Series(Counter(weeks)).reset_index()
    sessions_week.columns = ['week', 'Freq']
    
    # Initialize variables
    RAC_chronic = []
    RAC_acute = []
    RAC_ACWR = []
    
    # Initialize number of training sessions
    n_sessions_total = 0
    
    # Loop over the total weeks of training
    for i in sorted(set(weeks)):
        # First training week: RAC_chronic = RAC_acute
        if i == 1:
            # Loop over number of sessions in this week
            for j in range(1, sessions_week.loc[sessions_week['week'] == i, 'Freq'].iloc[0] + 1):
                # First training day: RAC_chronic = TL / RAC_acute = TL
                if j == 1:
                    # Count number of training sessions
                    n_sessions_total += 1
                    RAC_chronic.append(TL[n_sessions_total-1])
                    RAC_acute.append(TL[n_sessions_total-1])
                # Rest of the week
                elif j >= 2:
                    # Count number of training sessions
                    n_sessions_total += 1
                    RAC_chronic.append(np.sum(TL[:n_sessions_total]) / n_sessions_total)
                    RAC_acute.append(np.sum(TL[:n_sessions_total]) / n_sessions_total)
        
        # From second week to end of first month
        elif 2 <= i < 5:
            # Loop over number of sessions in this week
            for j in range(1, sessions_week.loc[sessions_week['week'] == i, 'Freq'].iloc[0] + 1):
                # Count number of training sessions
                n_sessions_total += 1
                RAC_chronic.append(np.sum(TL[:n_sessions_total]) / n_sessions_total)
                
                # RAC acute each 7 CALENDAR days
                # Calculate 7 days training blocks
                acute_TB = training_blocks(
                    training_dates=training_dates,
                    actual_TL=n_sessions_total,
                    diff_dates=6
                )
                RAC_acute.append(np.sum(TL[acute_TB['previous_TL']-1:n_sessions_total]) / acute_TB['n_sessions'])
        
        # From second month to end of data
        elif i >= 5:
            # Loop over number of sessions in this week
            for j in range(1, sessions_week.loc[sessions_week['week'] == i, 'Freq'].iloc[0] + 1):
                # Count number of training sessions
                n_sessions_total += 1
                
                # RAC chronic each 28 CALENDAR days
                # Calculate 28 days training blocks
                chronic_TB = training_blocks(
                    training_dates=training_dates,
                    actual_TL=n_sessions_total,
                    diff_dates=27
                )
                RAC_chronic.append(np.sum(TL[chronic_TB['previous_TL']-1:n_sessions_total]) / chronic_TB['n_sessions'])
                
                # RAC acute each 7 CALENDAR days
                # Calculate 7 days training blocks
                acute_TB = training_blocks(
                    training_dates=training_dates,
                    actual_TL=n_sessions_total,
                    diff_dates=6
                )
                RAC_acute.append(np.sum(TL[acute_TB['previous_TL']-1:n_sessions_total]) / acute_TB['n_sessions'])
    
    # Calculate ACWR
    RAC_ACWR = np.array(RAC_acute) / np.array(RAC_chronic)
    
    return {
        "RAC_acute": np.round(RAC_acute, 2),
        "RAC_chronic": np.round(RAC_chronic, 2),
        "RAC_ACWR": np.round(RAC_ACWR, 2)
    }

def RAU(TL, weeks, training_dates):
    """
    Calculate Rolling Average Uncoupled (RAU), Acute, and Acute:Chronic Workload Ratio
    
    Parameters:
    -----------
    TL : list or array
        Training Load values
    weeks : list or array
        Week numbers for each training session
    training_dates : list or array
        Dates of each training session (should be pandas datetime objects)
    
    Returns:
    --------
    dict
        RAU_acute: Acute rolling average of training load
        RAU_chronic: Chronic rolling average of training load
        RAU_ACWR: Acute:Chronic Workload Ratio
    """
    # Convert inputs to appropriate types if needed
    TL = np.array(TL)
    weeks = np.array(weeks)
    training_dates = pd.to_datetime(training_dates) if not isinstance(training_dates[0], pd.Timestamp) else training_dates
    
    # Count number of sessions per week
    sessions_week = pd.Series(Counter(weeks)).reset_index()
    sessions_week.columns = ['week', 'Freq']
    
    # Initialize variables
    RAU_chronic = np.full(len(TL), np.nan)  # Initialize with NaN values
    RAU_acute = np.empty(len(TL))
    RAU_ACWR = np.full(len(TL), np.nan)  # Initialize with NaN values
    
    # Initialize number of training sessions
    n_sessions_total = 0
    # We also need a new counter for the number of training sessions
    n_sessions_chronic = 1
    
    # Loop over the total weeks of training
    for i in sorted(set(weeks)):
        # First training week: RAU_chronic = NA / RAU_acute = Training load
        if i == 1:
            # Loop over number of sessions in this week
            for j in range(1, sessions_week.loc[sessions_week['week'] == i, 'Freq'].iloc[0] + 1):
                # First training day: RAU_chronic = NA / RAU_acute = TL
                if j == 1:
                    # Count number of training sessions
                    n_sessions_total += 1
                    # RAU_chronic[n_sessions_total] = NA (already set to NaN)
                    RAU_acute[n_sessions_total-1] = TL[n_sessions_total-1]
                # Rest of the week
                elif j >= 2:
                    # Count number of training sessions
                    n_sessions_total += 1
                    # RAU_chronic[n_sessions_total] = NA (already set to NaN)
                    RAU_acute[n_sessions_total-1] = np.sum(TL[:n_sessions_total]) / n_sessions_total
                
                # During first week of RAU ACWR = NA (already set to NaN)
        
        # From the beginning of the second week to end of third week
        elif 2 <= i < 5:
            # Loop over number of sessions in this week
            for j in range(1, sessions_week.loc[sessions_week['week'] == i, 'Freq'].iloc[0] + 1):
                # Count number of training sessions
                n_sessions_total += 1
                
                # RAU acute each 7 CALENDAR days
                # Calculate 7 days training blocks
                acute_TB = training_blocks(
                    training_dates=training_dates,
                    actual_TL=n_sessions_total,
                    diff_dates=6
                )
                
                RAU_acute[n_sessions_total-1] = np.sum(TL[acute_TB['previous_TL']-1:n_sessions_total]) / acute_TB['n_sessions']
                
                # RAU chronic
                # (acute_TB['previous_TL']-1) indicates the position of the first session
                # We are going to reuse this value to indicate the first value of the RAU chronic block
                RAU_chronic[n_sessions_total-1] = np.sum(TL[(acute_TB['previous_TL']-2)::-1]) / n_sessions_chronic
                n_sessions_chronic += 1
        
        # From fourth week to end of data
        elif i >= 5:
            # Loop over number of sessions in this week
            for j in range(1, sessions_week.loc[sessions_week['week'] == i, 'Freq'].iloc[0] + 1):
                # Count number of training sessions
                n_sessions_total += 1
                
                # RAU acute each 7 CALENDAR days
                # Calculate 7 days training blocks
                acute_TB = training_blocks(
                    training_dates=training_dates,
                    actual_TL=n_sessions_total,
                    diff_dates=6
                )
                
                RAU_acute[n_sessions_total-1] = np.sum(TL[acute_TB['previous_TL']-1:n_sessions_total]) / acute_TB['n_sessions']
                
                # RAU chronic
                chronic_TB = training_blocks(
                    training_dates=training_dates,
                    actual_TL=n_sessions_total,
                    diff_dates=20
                )
                
                # Number of sessions include in the chronic training block =
                # Number of sessions in chronic - number of sessions in acute
                RAU_chronic[n_sessions_total-1] = np.sum(TL[acute_TB['previous_TL']-1:chronic_TB['previous_TL']-1]) / chronic_TB['n_sessions']
    
    # Calculate ACWR
    RAU_ACWR = RAU_acute / RAU_chronic
    
    return {
        "RAU_acute": np.round(RAU_acute, 2),
        "RAU_chronic": np.round(RAU_chronic, 2),
        "RAU_ACWR": np.round(RAU_ACWR, 2)
    }

def EWMA(TL):
    """
    Calculate Exponentially Weighted Moving Average (EWMA)
    
    Parameters:
    -----------
    TL : list or array
        Training Load values
    
    Returns:
    --------
    dict
        EWMA_acute: Acute exponentially weighted moving average
        EWMA_chronic: Chronic exponentially weighted moving average
        EWMA_ACWR: Acute:Chronic Workload Ratio using EWMA
    """
    # Convert input to numpy array if it's not already
    TL = np.array(TL)
    
    # Initialize variables
    EWMA_chronic = np.zeros(len(TL))
    EWMA_acute = np.zeros(len(TL))
    
    # Lambda values
    lambda_acute = 2 / (7 + 1)
    lambda_chronic = 2 / (28 + 1)
    
    # Loop over the TL
    for i in range(len(TL)):
        # First training day: EWMA_chronic = TL / EWMA_acute = TL
        if i == 0:  # Python uses 0-based indexing
            EWMA_chronic[i] = TL[i]
            EWMA_acute[i] = TL[i]
        
        if i > 0:
            EWMA_chronic[i] = TL[i] * lambda_chronic + ((1 - lambda_chronic) * EWMA_chronic[i-1])
            EWMA_acute[i] = TL[i] * lambda_acute + ((1 - lambda_acute) * EWMA_acute[i-1])
    
    # Calculate ACWR
    EWMA_ACWR = EWMA_acute / EWMA_chronic
    
    return {
        "EWMA_acute": np.round(EWMA_acute, 2),
        "EWMA_chronic": np.round(EWMA_chronic, 2),
        "EWMA_ACWR": np.round(EWMA_ACWR, 2)
    }

def plot_ACWR(db, day_col, TL_col, acwr_data, acwr_methods=None, colour=None, xLabel=None, 
             y0Label=None, y1Label=None, plotTitle=None):
    """
    Plot the Acute:Chronic Workload Ratio (ACWR) and Training Load using Plotly
    
    Parameters:
    -----------
    db : pandas.DataFrame
        DataFrame containing the training data
    day_col : str
        Name of the day/date column in the database
    TL_col : str
        Name of the training load column in the database
    acwr_data : dict
        Dictionary containing ACWR data for different methods (RAC, RAU, EWMA)
    acwr_methods : list, optional
        List of ACWR methods to plot (default: all methods in acwr_data)
    colour : str, optional
        Color for the training load bars (default: "#87CEEB")
    xLabel : str, optional
        Label for x-axis (default: "Days")
    y0Label : str, optional
        Label for left y-axis (training load) (default: "Load [AU]")
    y1Label : str, optional
        Label for right y-axis (ACWR) (default: "Acute:chronic workload ratio")
    plotTitle : str, optional
        Title of the plot (default: "ACWR")
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    # Check variables
    if db is None:
        raise ValueError("you must provide a db")
    if day_col is None:
        raise ValueError("you must provide the name of the day training column in the database")
    if TL_col is None:
        raise ValueError("you must provide the name of the training load column in the database")
    if acwr_data is None:
        raise ValueError("you must provide ACWR data")
    
    # Set defaults
    if colour is None:
        colour = "#87CEEB"
    if xLabel is None:
        xLabel = "Days"
    if y0Label is None:
        y0Label = "Load [AU]"
    if y1Label is None:
        y1Label = "Acute:chronic workload ratio"
    if plotTitle is None:
        plotTitle = "ACWR"
    
    # If no methods specified, use all available
    if acwr_methods is None:
        acwr_methods = list(acwr_data.keys())
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add training load bars
    fig.add_trace(
        go.Bar(
            x=db[day_col],
            y=db[TL_col],
            name="Daily Loads",
            marker_color=colour,
            marker_line_color="#0000FF",
            marker_line_width=3
        ),
        secondary_y=False
    )
    
    # Colors for different ACWR methods
    colors = {
        "RAC_ACWR": "black", 
        "RAU_ACWR": "red", 
        "EWMA_ACWR": "green"
    }
    
    # Add ACWR lines for selected methods
    for method in acwr_methods:
        if method in acwr_data:
            fig.add_trace(
                go.Scatter(
                    x=db[day_col],
                    y=acwr_data[method],
                    name=method,
                    line=dict(color=colors.get(method, "black"), width=3),
                    mode="lines"
                ),
                secondary_y=True
            )
    
    # Configure axes
    fig.update_xaxes(title_text=xLabel)
    fig.update_yaxes(title_text=y0Label, secondary_y=False)
    fig.update_yaxes(title_text=y1Label, secondary_y=True)
    
    # Set y-axis ranges
    y0max = db[TL_col].max()
    y1max = max([acwr_data[m].max() for m in acwr_methods if m in acwr_data])
    fig.update_yaxes(range=[0, y0max * 1.1], secondary_y=False)
    fig.update_yaxes(range=[0, y1max * 1.1], secondary_y=True)
    
    # Update layout
    fig.update_layout(
        title=plotTitle,
        title_x=0.5,
        font=dict(
            family="Tahoma, Geneva, sans-serif",
            size=12
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        barmode='group',
        plot_bgcolor='white'
    )
    
    return fig

def process_training_data(df):
    """
    Process training data and calculate ACWR using different methods
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing training data with columns for ID, Week, Day, TL, and Training_Date
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with original data and ACWR calculations added
    """
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Ensure data types are correct
    data['Day'] = pd.to_numeric(data['Day'], errors='coerce')
    data['Week'] = pd.to_numeric(data['Week'], errors='coerce')
    data['TL'] = pd.to_numeric(data['TL'], errors='coerce')
    data['Training_Date'] = pd.to_datetime(data['Training_Date'], errors='coerce')
    
    # Sort by training date
    data = data.sort_values('Training_Date')
    
    # Calculate ACWR using different methods
    rac_results = RAC(data['TL'].values, data['Week'].values, data['Training_Date'].values)
    rau_results = RAU(data['TL'].values, data['Week'].values, data['Training_Date'].values)
    ewma_results = EWMA(data['TL'].values)
    
    # Add results to DataFrame
    data['RAC_Acute'] = rac_results['RAC_acute']
    data['RAC_Chronic'] = rac_results['RAC_chronic']
    data['RAC_ACWR'] = rac_results['RAC_ACWR']
    
    data['RAU_Acute'] = rau_results['RAU_acute']
    data['RAU_Chronic'] = rau_results['RAU_chronic']
    data['RAU_ACWR'] = rau_results['RAU_ACWR']
    
    data['EWMA_Acute'] = ewma_results['EWMA_acute']
    data['EWMA_Chronic'] = ewma_results['EWMA_chronic']
    data['EWMA_ACWR'] = ewma_results['EWMA_ACWR']
    
    return data