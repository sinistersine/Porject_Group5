# 
from ftplib import all_errors
import streamlit as st
import pandas as pd
import plotly.express as px

import numpy as np

# === material-trip energy rule ===
BASE_MIN, BASE_KWH = 20, 10.32   # 20 min → 10.32 kWh
TOL = 0.05                       # match-tolerantie

@st.cache_data
def load_data(file):
    df = pd.read_excel(file, sheet_name="Sheet1")

    # times -> datetime
    start = pd.to_datetime(df['start time'], format="%H:%M:%S", errors='coerce')
    end   = pd.to_datetime(df['end time'],   format="%H:%M:%S", errors='coerce')

    # nacht-rollover: alleen waar beide niet NaT zijn én end < start
    roll_mask = (end < start) & start.notna() & end.notna()
    # cast bool -> int vóór to_timedelta (anders TypeError)
    end_fix = end + pd.to_timedelta(roll_mask.astype(int), unit='D')

    # duur in minuten
    df['duration_minutes'] = (end_fix - start).dt.total_seconds() / 60.0

    # material-trip energy fix
    is_mat = df['activity'].astype(str).str.lower().str.contains('material')

    # bestaande kWh normaliseren (komma → punt)
    if 'energy consumption' in df.columns:
        df['energy consumption'] = pd.to_numeric(
            df['energy consumption'].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        )
    else:
        df['energy consumption'] = np.nan

    exp = BASE_KWH * (df['duration_minutes'] / BASE_MIN)

    # audit (optioneel)
    df['energy_expected_material'] = np.where(is_mat, exp, np.nan)
    df['energy_diff'] = np.where(is_mat, df['energy consumption'] - exp, np.nan)
    df['energy_match'] = np.where(
        is_mat & (df['energy_diff'].abs() <= TOL), 'OK',
        np.where(is_mat, 'MISMATCH', '')
    )

    # ✅ daadwerkelijk fixen
    df.loc[is_mat, 'energy consumption'] = exp.round(3)

    # sla de gefixte tijden terug in df zodat Gantt de juiste gebruikt
    df['start time'] = start
    df['end time']   = end_fix

    return df

# command to see website: streamlit run dashboard.py

# Pagina-instellingen
st.set_page_config(page_title="Bus Scheduling Dashboard", layout="wide")
st.title("🚌 Bus plan analysis")

# Sidebar: upload Excel-bestand
uploaded_file = st.sidebar.file_uploader("1) Upload het busplan (Excel)", type=["xlsx"])

# Tabs bovenaan
tab_gantt, tab_visuals, tab_analysis, tab_errors = st.tabs(["📊 Gantt Chart", "📈 Visualisaties", "🔍 Analyse", "🚨 Fouten"])

# Functie om Gantt Chart te maken
def plot_gantt_chart(df):
    # Label voor service trips
    df['label'] = df.apply(
        lambda row: str(int(row['line'])) if row['activity'] == 'service trip' and pd.notna(row['line']) else '',
        axis=1
    )
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
    fig.update_xaxes(range=["00:00:00", "23:59:59"]) # Zorg dat de x-as altijd 24 uur toont en dat de gantt chart niet zo'n zoomed-in view heeft
    fig.update_traces(marker=dict(line=dict(color='black', width=1)), width=0.3)
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))

    return fig

# Tab 1: Gantt Chart
with tab_gantt:
    st.subheader("📊 Gantt Chart")
    if uploaded_file:
        df = load_data(uploaded_file)
        fig = plot_gantt_chart(df)
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
    if uploaded_file:
        df = load_data(uploaded_file)          # <-- en hier

        st.write("### Gemiddelde duur per activiteit (in minuten)")
        avg_duration = df.groupby('activity', as_index=False)['duration_minutes'].mean()
        st.dataframe(avg_duration)

        st.write("### Totale duur per bus (in minuten)")
        total_duration_per_bus = (df.groupby('bus', as_index=False)['duration_minutes']
                                    .sum()
                                    .rename(columns={'duration_minutes': 'total_duration_minutes'}))
        st.dataframe(total_duration_per_bus)

        # Energie-analyse (nu al gefixt voor material trips)
        if 'energy consumption' in df.columns:
            per_bus = df.groupby('bus', as_index=False).agg(
                verbruik_kWh=('energy consumption', lambda s: s.clip(lower=0).sum()),
                geladen_kWh =('energy consumption', lambda s: (-s.clip(upper=0)).sum()),
                netto_kWh   =('energy consumption', 'sum'),
            )
            BATTERY_KWH = 300.0
            per_bus['eind_SOC_%'] = (100 - (per_bus['netto_kWh'] / BATTERY_KWH) * 100).clip(0, 100)

            st.write("### Energie per bus (kWh) + eind-SOC schatting")
            st.dataframe(per_bus.sort_values('netto_kWh', ascending=False), use_container_width=True)

            st.write("### Audit: material trips check")
            st.dataframe(
                df.loc[df['energy_expected_material'].notna(),
                       ['activity','start time','end time','duration_minutes',
                        'energy_expected_material','energy consumption',
                        'energy_diff','energy_match']]
            )
        else:
            st.info("Kolom 'energy consumption' niet gevonden in het bestand.")
    else:
        st.info("Upload een Excel-bestand in de sidebar om de analyse te zien.")

# Tab 4: Fouten
# hier kunnen we alle constraints in zetten waar alle data aan moet voldoen
# en als er iets niet klopt, dat dat hier getoond wordt
with tab_errors:
    st.subheader("🚨 Fouten")
    st.write("Hier kun je een lijst tonen van alle fouten die zijn opgetreden tijdens de planning.")
