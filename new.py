# streamlit run dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

# === material-trip energy rule ===
BASE_MIN, BASE_KWH = 20, 10.32
TOL = 0.05

ACTIVITY_DISPLAY = {
    'service trip': 'service trip',
    'material trip': 'material trip',
    'idle': 'idle time',
    'charging': 'charging time'
}

ACTIVITY_COLORS = {
    'service trip': '#F79AC9',
    'material trip': '#CBA0E2',
    'idle': '#F39C12',
    'charging': '#A0E7E5'
}

@st.cache_data
def load_data(file):
    df = pd.read_excel(file, sheet_name="Sheet1")
    start = pd.to_datetime(df['start time'], format="%H:%M:%S", errors='coerce')
    end = pd.to_datetime(df['end time'], format="%H:%M:%S", errors='coerce')

    roll_mask = (end < start) & start.notna() & end.notna()
    end_fix = end + pd.to_timedelta(roll_mask.astype(int), unit='D')

    df['duration_minutes'] = (end_fix - start).dt.total_seconds() / 60.0
    is_mat = df['activity'].astype(str).str.lower().str.contains('material')

    if 'energy consumption' in df.columns:
        df['energy consumption'] = pd.to_numeric(
            df['energy consumption'].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        )
    else:
        df['energy consumption'] = np.nan

    exp = BASE_KWH * (df['duration_minutes'] / BASE_MIN)
    df['energy_expected_material'] = np.where(is_mat, exp, np.nan)
    df['energy_diff'] = np.where(is_mat, df['energy consumption'] - exp, np.nan)
    df['energy_match'] = np.where(
        is_mat & (df['energy_diff'].abs() <= TOL), 'OK',
        np.where(is_mat, 'MISMATCH', '')
    )
    df.loc[is_mat, 'energy consumption'] = exp.round(3)
    df['start time'] = start
    df['end time'] = end_fix
    return df


# === Helper functions ===
def check_bus_battery_survival(df):
    # placeholder check
    if df is None or df.empty:
        return {}
    return {"ok": True, "summary": f"{len(df)} trips checked"}


def get_battery_diagnostics(df):
    if df is None or df.empty:
        return pd.DataFrame()
    return df[df['activity'].str.contains("charge", case=False, na=False)]


def get_timetable_diagnostics(df):
    if df is None or df.empty:
        return pd.DataFrame()
    late = df[df['activity'].str.contains("service", case=False, na=False) & (df['duration_minutes'] > 60)]
    return late


def compute_plan_kpi(df, timetable):
    if df is None or df.empty:
        return {'kpi': 0, 'late_service_trips': 0, 'n_buses': 0, 'ok_buses': 0, 'mix': pd.DataFrame()}
    mix = df.groupby('activity', as_index=False)['duration_minutes'].sum()
    total_min = mix['duration_minutes'].sum()
    if total_min > 0:
        mix['percent'] = 100 * mix['duration_minutes'] / total_min
    stats = {
        'kpi': round(93.1, 1),
        'late_service_trips': int(len(df[df['activity'].str.contains("service", case=False, na=False)])),
        'n_buses': df['bus number'].nunique() if 'bus number' in df.columns else 0,
        'ok_buses': df['bus number'].nunique() if 'bus number' in df.columns else 0,
        'mix': mix
    }
    return stats


# === UI setup ===
st.set_page_config(page_title="Bus Planning dashboard", layout="wide")
st.title("🚌 Bus Planning dashboard")

uploaded_file = st.sidebar.file_uploader("1) Upload the busplan (Excel)", type=["xlsx"], key="busplan")

tab_gantt, tab_visuals, tab_analysis, tab_errors, tab_kpi = st.tabs(
    ["📊 Gantt-chart", "📈 Visualisations", "🔍 Analysis", "🚨 Errors", "📊 KPI Dashboard"]
)


# === Tab 1: Gantt chart ===
with tab_gantt:
    if uploaded_file:
        df = load_data(uploaded_file)

        st.subheader("Gantt-chart van busplanning")
        if not df.empty:
            fig = px.timeline(
                df,
                x_start="start time",
                x_end="end time",
                y="bus number",
                color="activity",
                color_discrete_map=ACTIVITY_COLORS,
                hover_data=["activity", "duration_minutes"]
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Geen gegevens om te tonen.")
    else:
        st.info("Upload eerst een busplanbestand om de Gantt-chart te bekijken.")


# === Tab 2: Visualisations ===
with tab_visuals:
    if uploaded_file:
        df = load_data(uploaded_file)
        st.subheader("Gemiddelde duur per activiteit")

        avg_duration = df.groupby("activity", as_index=False)["duration_minutes"].mean()
        fig = px.bar(
            avg_duration,
            x="activity",
            y="duration_minutes",
            color="activity",
            color_discrete_map=ACTIVITY_COLORS,
            title="Gemiddelde duur per activiteit (minuten)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload een bestand om visualisaties te zien.")


# === Tab 3: Analysis ===
with tab_analysis:
    if uploaded_file:
        df = load_data(uploaded_file)
        st.subheader("🔍 Analyse van busplanning")

        st.markdown("#### Batterijcontrole")
        diag = get_battery_diagnostics(df)
        st.dataframe(diag)

        st.markdown("#### Dienstregelingcontrole")
        timetable_diag = get_timetable_diagnostics(df)
        st.dataframe(timetable_diag)
    else:
        st.info("Upload een bestand om analyses uit te voeren.")


# === Tab 4: Errors ===
with tab_errors:
    if uploaded_file:
        df = load_data(uploaded_file)
        errors = df[df['energy_match'] == 'MISMATCH']
        st.subheader("🚨 Material trip energy mismatches")
        st.dataframe(errors)
    else:
        st.info("Upload een bestand om fouten te controleren.")


# === Tab 5: KPI Dashboard (aangepast) ===
def render_plan_card(plan_title, df, timetable):
    st.markdown(f"### {plan_title}")

    if df is None or df.empty:
        st.info("No data.")
        return

    stats = compute_plan_kpi(df, timetable)

    # Alleen taartdiagram tonen (geen KPI-metrics)
    pie = px.pie(
        stats['mix'],
        names='activity',
        values='minutes',
        title="Time distribution",
        hole=0.35,
        color='activity',
        color_discrete_map={
            'service trip': ACTIVITY_COLORS.get('service trip', '#664EDC'),
            'material trip': ACTIVITY_COLORS.get('material trip', '#D904B2'),
            'idle': ACTIVITY_COLORS.get('idle', '#E7DF12'),
        }
    )
    pie.update_traces(textposition='inside', texttemplate='%{label}<br>%{percent:.0%}')
    st.plotly_chart(pie, use_container_width=True)


with tab_kpi:
    if uploaded_file:
        df = load_data(uploaded_file)
        st.subheader("📊 KPI Dashboard")
        render_plan_card("Current Plan", df, None)
    else:
        st.info("Upload een bestand om het KPI-dashboard te bekijken.")