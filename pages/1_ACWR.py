import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import base64
from utils.acwr_functions import process_training_data, plot_ACWR

# App header
st.markdown("<div class='main-header'>Acute:Chronic Workload Ratio (ACWR) Calculator</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read the file
        df = pd.read_csv(uploaded_file, sep=';')
        
        # Show the original data
        st.markdown("<div class='sub-header'>Original Data</div>", unsafe_allow_html=True)
        st.dataframe(df)
        
        # Process the data
        with st.spinner('Calculating ACWR...'):
            processed_df = process_training_data(df)
        
        # Show the processed data
        st.markdown("<div class='sub-header'>Processed Data with ACWR Calculations</div>", unsafe_allow_html=True)
        st.dataframe(processed_df)
        
        # Download button for processed data
        csv = processed_df.to_csv(index=False, sep=';')
        
        st.download_button(
            label="Download Processed Data",
            data=csv,
            file_name="processed_training_data.csv",
            mime="text/csv",
        )
        
        # Plot the data
        st.markdown("<div class='sub-header'>ACWR Visualization</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-text'>
        Select which ACWR methods to display on the chart. The bars represent training load values.
        </div>
        """, unsafe_allow_html=True)
        
        # Let user select which ACWR methods to display
        acwr_methods = st.multiselect(
            "Select ACWR methods to display",
            ["RAC_ACWR", "RAU_ACWR", "EWMA_ACWR"],
            default=["RAC_ACWR"]
        )
        
        if acwr_methods:
            # Prepare data for plotting
            acwr_data = {
                "RAC_ACWR": processed_df["RAC_ACWR"].values,
                "RAU_ACWR": processed_df["RAU_ACWR"].values,
                "EWMA_ACWR": processed_df["EWMA_ACWR"].values
            }
            
            # Create the plot
            fig = plot_ACWR(
                db=processed_df,
                day_col="Day",
                TL_col="TL",
                acwr_data=acwr_data,
                acwr_methods=acwr_methods,
                plotTitle="Training Load and ACWR Over Time"
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Option to download the plot
            st.markdown("""
            <div class='info-text'>
            You can download the plot by clicking the camera icon that appears when you hover over the top-right of the chart.
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
        st.markdown("""
        <div class='info-text'>
        Please ensure your CSV file has the correct format with columns:
        <ul>
            <li>ID</li>
            <li>Week</li>
            <li>Day</li>
            <li>TL</li>
            <li>Training_Date</li>
        </ul>
        and that the values are in the proper format.
        </div>
        """, unsafe_allow_html=True)
else:
    # Show example format when no file is uploaded
    st.markdown("""
    <div class='info-text'>
    <b>Sample CSV format:</b><br>
    ID;Week;Day;TL;Training_Date<br>
    1;1;1;500;2023-01-01<br>
    1;1;3;450;2023-01-03<br>
    1;1;5;550;2023-01-05<br>
    ...
    </div>
    """, unsafe_allow_html=True)