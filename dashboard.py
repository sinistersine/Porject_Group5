import streamlit as st
import pandas as pd
import plotly.express as px
from numpy.random import default_rng as rng

# Set up the data to use it conveniently later on to use for the graphs


st.set_page_config(page_title="Bus Scheduling Dashboard", layout="wide")
st.title("Bus Scheduling & Energy Dashboard")
st.write(
    "1) Upload your bus schedule file below. "
    "Then switch between the Schedule and Energy tabs to view the cleaned table, "
)

# run the app with: streamlit run dashboard.py

# FILE UPLOAD
df_schedule = st.file_uploader(
    " 1) Upload your bus schedule file (CSV or Excel)",
    type=["csv", "xlsx"])


# THE TABS
tab_schedule, tab_energy= st.tabs(["Bus schedule", "Energy consumption"])

# THE ENERGY TAB
with tab_energy:
    tab_energy.subheader("Example of energy consumption data")
    df = rng(0).standard_normal((10, 1))
    tab_energy.line_chart(df, color="#f178a1")
    """
    Validates battery status throughout the bus schedule and adds state of charge (SOC) as a column.
    Returns rows where SOC drops below the minimum.
    """
    
    
    
    
    
# TBE SCHEDULE TAB
with tab_schedule:
    st.header("Bus Schedule")
    
    
    
    
