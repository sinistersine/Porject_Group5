import streamlit as st
import pandas as pd
import plotly.express as px
from numpy.random import default_rng as rng

st.set_page_config(page_title="Bus Scheduling Dashboard", layout="wide")
st.title("Bus Scheduling & Energy Dashboard")
st.write(
    "Upload your bus schedule file below. "
    "Then switch between the Schedule and Energy tabs to view the cleaned table, "
  
)
# run the app with: streamlit run dashboard.py
tab_schedule, tab_energy= st.tabs(["Bus schedule", "Energy consumption"])

schedule_file = st.file_uploader(
    "Upload your bus schedule file (CSV or Excel)",
    type=["csv", "xlsx"],
)

with tab_energy:
    tab_energy.subheader("Example of energy consumption data")
    df = rng(0).standard_normal((10, 1))
    tab_energy.line_chart(df, color="#f178a1")
    
with tab_schedule:
    st.header("Bus Schedule")
    
