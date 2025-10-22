# streamlit run dashboard.py  (dit is wat je in terminal typt om de app te runnen)
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

# Functie om Gantt Chart met meerdere bussen in 1 figuur te maken
def plot_gantt_all_buses(df):
    df_plot = df.copy()

    # Label voor service trips
    df_plot['label'] = df_plot.apply(
        lambda row: str(int(row['line'])) if row['activity'] == 'service trip' and pd.notna(row['line']) else '',
        axis=1
    )

    # Y-as = bus ID
    df_plot['row'] = df_plot['bus'].apply(lambda x: f"Bus {int(x)}" if pd.notna(x) else "Onbekend")

    # Gantt Chart
   # Functie om Gantt Chart met meerdere bussen te maken (interactief)
def plot_gantt_interactive(df):
    df_plot = df.copy()

    # Label voor service trips
    df_plot['label'] = df_plot.apply(
        lambda row: str(int(row['line'])) if row['activity'] == 'service trip' and pd.notna(row['line']) else '',
        axis=1
    )

    # Y-as = bus ID
    df_plot['row'] = df_plot['bus'].apply(lambda x: f"Bus {int(x)}" if pd.notna(x) else "Onbekend")

    # Gantt Chart
    fig = px.timeline(
        df_plot,
        x_start="start time",
        x_end="end time",
        y="row",
        color="activity",
        text="label",
        title="Gantt Chart – Alle bussen",
        color_discrete_map={
            "service trip": "#F79AC9",   # pastelroze
            "material trip": "#CBA0E2",  # lichtlila
            "idle": "#FDB79F"            # perzik
        }
    )

    # X-as van 05:00 tot 01:00
    base_date = df_plot['start time'].min().date()
    start_range = pd.Timestamp(f"{base_date} 05:00:00")
    end_range = pd.Timestamp(f"{base_date} 01:00:00") + pd.Timedelta(days=1)
    fig.update_xaxes(range=[start_range, end_range])
    fig.update_yaxes(title="Bus", autorange="reversed")
    
    # Layout dynamisch op basis van aantal bussen
    fig.update_layout(
        height=400 + 30*len(df_plot['bus'].unique()),
        margin=dict(l=20, r=20, t=40, b=20),
        dragmode='zoom',  # activeer klik & sleep zoom
    )

    fig.update_traces(marker=dict(line=dict(color='black', width=1)))

    return fig

with tab_gantt:
    st.subheader("📊 Gantt Chart – Selecteer bus om uit te zoomen/zoomen")
    if uploaded_file:
        df = load_data(uploaded_file)

        # Maak lijst van unieke bussen voor selectie
        bus_options = sorted(df['bus'].dropna().unique())

        # Multi-select met 'Alle bussen' optie
        selected_buses = st.multiselect(
            "Selecteer één of meerdere bussen (of 'Alle bussen')",
            options=[0] + bus_options,
            default=[0],  # standaard 'Alle bussen'
            format_func=lambda x: f"Bus {int(x)}" if x != 0 else "Alle bussen"
)


        # Filter data op geselecteerde bus(s)
        if 0 in selected_buses or not selected_buses:  # 0 = Alle bussen
            df_plot = df.copy()
        else:
            df_plot = df[df['bus'].isin(selected_buses)].copy()
            
        # Label voor service trips
        df_plot['label'] = df_plot.apply(
            lambda row: str(int(row['line'])) if row['activity'] == 'service trip' and pd.notna(row['line']) else '',
            axis=1
        )
        df_plot['row'] = df_plot['bus'].apply(lambda x: f"Bus {int(x)}" if pd.notna(x) else "Onbekend")

        # Gantt chart
        import plotly.express as px

# Stel df_plot is je dataframe met start, end, bus en label
fig = px.timeline(
    df_plot,
    x_start="start time",
    x_end="end time",
    y="row",
    color="activity",
    text=None,  # we gebruiken geen standaard text
    color_discrete_map={
        "service trip": "#F79AC9",
        "material trip": "#CBA0E2",
        "idle": "#FDB79F"
    }
)

# Voeg verticaal georiënteerde annotations toe voor labels
for i, row in df_plot.iterrows():
    fig.add_annotation(
        x=row['start time'],           # begin van de balk
        y=row['row'],                  # bus-rij
        text=str(int(row['line'])),    # bv. 400 of 401
        showarrow=False,
        xanchor='left',                # linkerkant van de balk
        yanchor='bottom',
        textangle=-90,                 # verticaal
        font=dict(size=10, color='black')
    )

# Layout tweaks
fig.update_yaxes(title="Bus", autorange="reversed")
fig.update_layout(height=400 + 30*len(df_plot['bus'].unique()))
fig.update_traces(marker=dict(line=dict(color='black', width=1)))


        # Layout
        base_date = df_plot['start time'].min().date()
        start_range = pd.Timestamp(f"{base_date} 05:00:00")
        end_range = pd.Timestamp(f"{base_date} 01:00:00") + pd.Timedelta(days=1)
        fig.update_xaxes(range=[start_range, end_range])
        fig.update_yaxes(title="Bus", autorange="reversed")
        fig.update_layout(
            height=400 + 30*len(df_plot['bus'].unique()),
            margin=dict(l=20, r=20, t=40, b=20),
            dragmode='zoom'
        )
        fig.update_traces(marker=dict(line=dict(color='black', width=1)))

        st.plotly_chart(fig, use_container_width=True)



# Tab 2: Visualisaties
with tab_visuals:
    st.subheader("📈 Visualisaties")
    if uploaded_file:
        df = load_data(uploaded_file)

        # ===== Controls =====
        group_by = st.radio("Groepeer op", ["bus", "line"], horizontal=True)
        cap_kwh = st.number_input("Batterijcapaciteit (kWh)", min_value=50.0, max_value=1000.0, value=300.0, step=10.0)
        start_soc = st.slider("Start-SOC (%)", min_value=0, max_value=100, value=100, step=1)

        # optioneel filteren op specifieke bussen/lijnen
        opts = sorted(df[group_by].dropna().astype(str).unique().tolist())
        pick = st.multiselect(f"Selecteer {group_by}(s)", options=opts, default=opts[:min(5, len(opts))])
        if pick:
            df = df[df[group_by].astype(str).isin(pick)]

        # ===== SOC curve bouwen =====
        # tijdstempel voor volgorde (gebruik start time; end time kan over middernacht gaan)
        ts = 'start time'
        d = df.copy()
        d['energy consumption'] = pd.to_numeric(d['energy consumption'], errors='coerce').fillna(0.0)

        def build_soc(g):
            g = g.sort_values(ts).copy()
            # baseline punt op t0 met 0 verbruik zodat de lijn bij start_SOC begint
            if not g.empty:
                baseline = g.iloc[[0]].copy()
                baseline['energy consumption'] = 0.0
                baseline[ts] = g[ts].min()
                g = pd.concat([baseline, g], ignore_index=True)

            g['net_kwh_cum'] = g['energy consumption'].cumsum()
            g['soc_%'] = (start_soc - (g['net_kwh_cum'] / cap_kwh) * 100).clip(0, 100)
            return g[[group_by, ts, 'soc_%']]

        soc_df = (d.groupby(group_by, dropna=False)
                    .apply(build_soc)
                    .reset_index(drop=True))

        import plotly.express as px
        fig_soc = px.line(
            soc_df,
            x=ts, y='soc_%', color=group_by,
            title=f"SoH / SoC verloop per {group_by}",
            labels={'soc_%': 'SOC (%)', ts: 'Tijd'}
        )
        # Maak het wat leesbaarder
        fig_soc.update_yaxes(range=[0, 100])
        fig_soc.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))

        st.plotly_chart(fig_soc, use_container_width=True)

        # klein tabelletje erbij zodat je kan checken
        st.write("Voorbeeldpunten (eerste 30):")
        st.dataframe(soc_df.sort_values([group_by, ts]).head(30), use_container_width=True)

        st.caption("Positieve 'energy consumption' = verbruik (SOC omlaag), negatieve = laden (SOC omhoog).")
    else:
        st.info("Upload een Excel-bestand in de sidebar om de SOC-grafiek te zien.")

# Tab 3: Analyse
with tab_analysis:
    st.subheader("🔍 Analyse")
    if uploaded_file:
        df = load_data(uploaded_file)          

        st.write("### Gemiddelde duur per activiteit (in minuten)")
        avg_duration = df.groupby('activity', as_index=False)['duration_minutes'].mean()
        st.dataframe(avg_duration)

        # === NIEUW BLOK: Gemiddelde duur per activiteit per lijn ===
        df['line_str'] = df['line'].astype(str).str.replace('.0', '', regex=False)

        avg_duration_per_line_activity = (
            df.groupby(['line_str', 'activity'], dropna=False)['duration_minutes']
              .mean()
              .reset_index()
              .rename(columns={'line_str': 'line'})
        )

        st.write("### Gemiddelde duur per activiteit **per bus** (in minuten)")
        avg_duration_per_bus_activity = (
            df.groupby(['bus', 'activity'], dropna=False)['duration_minutes']
            .mean()
            .reset_index()
            .sort_values(['bus', 'activity'])
        )
        
        st.dataframe(
            avg_duration_per_bus_activity.pivot(
                index='bus',
                columns='activity',
                values='duration_minutes'
                ).round(2),
            use_container_width=True
        )

        # Alleen material trips per lijn
        mat = df['activity'].str.lower().eq('material trip')
        avg_material_per_line = (
            df.loc[mat]
              .groupby('line_str', dropna=False)['duration_minutes']
              .mean()
              .reset_index()
              .rename(columns={'line_str': 'line', 'duration_minutes': 'avg_material_duration_min'})
              .sort_values('line')
        )



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
