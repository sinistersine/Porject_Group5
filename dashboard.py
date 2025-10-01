import streamlit as st
import pandas as pd
import plotly.express as px

# Pagina-instellingen
st.set_page_config(page_title="Bus Scheduling Dashboard", layout="wide")
st.title("🚌 Bus plan analysis")

# Sidebar: upload Excel-bestand
uploaded_file = st.sidebar.file_uploader("Upload het busplan (Excel)", type=["xlsx"])

# Tabs bovenaan
tab_gantt, tab_visuals, tab_analysis = st.tabs(["📊 Gantt Chart", "📈 Visualisaties", "🔍 Analyse"])

# Functie om Gantt Chart te maken
def plot_gantt_chart(uploaded_file):
    df = pd.read_excel(uploaded_file, sheet_name="Sheet1")

    # Kolommen naar datetime
    df['start time'] = pd.to_datetime(df['start time'], format="%H:%M:%S")
    df['end time'] = pd.to_datetime(df['end time'], format="%H:%M:%S")

    # Label voor service trips
    df['label'] = df.apply(lambda row: str(int(row['line'])) if row['activity'] == 'service trip' and pd.notna(row['line']) else '', axis=1)

    # Eén enkele rij
    df['row'] = "Planning"

    # Gantt Chart
    fig = px.timeline(
        df,
        x_start="start time",
        x_end="end time",
        y="row",
        color="activity",
        text="label",
        title="Gantt Chart – Bus Planning",
        color_discrete_map={
            "service trip": "#E75480",   # roze
            "material trip": "#EC7699", # lichtroze
            "idle": "#F198B3"           # pastel roze
        }
    )

    # Layout tweaks
    fig.update_yaxes(showticklabels=False)
    fig.update_traces(marker=dict(line=dict(color='black', width=1)), width=0.3)
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))

    return fig

# Tab 1: Gantt Chart
with tab_gantt:
    st.subheader("📊 Gantt Chart")
    if uploaded_file:
        fig = plot_gantt_chart(uploaded_file)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload een Excel-bestand in de sidebar om de Gantt Chart te zien.")

# Tab 2: Visualisaties
with tab_visuals:
    st.subheader("📈 Visualisaties")
    st.write("Hier komen extra grafieken, zoals energieverbruik per bus of per lijn.")

# Tab 3: Analyse
with tab_analysis:
    st.subheader("🔍 Analyse")
    st.write("Hier kun je inzichten tonen, zoals gemiddelde reistijd, idle-tijd, etc.")