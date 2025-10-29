# streamlit run dashboard.py  (dit is wat je in terminal typt om de app te runnen)
from ftplib import all_errors
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

# === material-trip energy rule ===
BASE_MIN, BASE_KWH = 20, 10.32   # 20 min → 10.32 kWh
TOL = 0.05                       # match-tolerantie

# UI-only mapping for activity labels (safe display names; internals unchanged)
ACTIVITY_DISPLAY = {
    'service trip': 'service trip',
    'material trip': 'material trip',
    'idle': 'idle time',
    'charging': 'charging time'
}

# Colors
ACTIVITY_COLORS = {
    'service trip': '#F79AC9',
    'material trip': '#CBA0E2',
    'idle': '#F39C12',
    'charging': '#A0E7E5'
}

@st.cache_data
def load_data(file):
    df = pd.read_excel(file, sheet_name="Sheet1")

    # times -> datetime
    start = pd.to_datetime(df['start time'], format="%H:%M:%S", errors='coerce')
    end   = pd.to_datetime(df['end time'],   format="%H:%M:%S", errors='coerce')

    # nacht-rollover: alleen waar beide niet NaT zijn én end < start
    roll_mask = (end < start) & start.notna() & end.notna()
    end_fix = end + pd.to_timedelta(roll_mask.astype(int), unit='D')

    df['duration_minutes'] = (end_fix - start).dt.total_seconds() / 60.0

    # material-trip energy fix
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
    df['end time']   = end_fix

    return df


# === Streamlit setup ===
st.set_page_config(page_title="Bus Planning dashboard", layout="wide")
st.title("🚌 Bus Planning dashboard")

uploaded_file = st.sidebar.file_uploader("1) Upload the busplan (Excel)", type=["xlsx"], key="busplan")
tab_gantt, tab_visuals, tab_analysis, tab_errors = st.tabs(["📊 Gantt-chart", "📈 Visualizations", "🔍 Analysis", "🚨 Errors"])


# === Gantt helper ===
def plot_gantt_interactive(df, selected_buses=None):
    df_plot = df.copy()
    if selected_buses and 0 not in selected_buses:
        df_plot = df_plot[df_plot['bus'].isin(selected_buses)]

    df_plot['label'] = df_plot.apply(
        lambda row: '<br>'.join(list(str(int(row['line']))))
        if (row.get('activity') == 'service trip' and pd.notna(row.get('line')))
        else '',
        axis=1
    )
    df_plot['row'] = df_plot['bus'].apply(lambda x: f"Bus {int(x)}" if pd.notna(x) else "Onbekend")

    fig = px.timeline(
        df_plot,
        x_start="start time",
        x_end="end time",
        y="row",
        color="activity",
        text="label",
        title="Gantt Chart – Bus Planning",
        color_discrete_map={
            "service trip": "#664EDC",
            "material trip": "#D904B2",
            "idle": "#E7DF12",
            "charging": "#0DD11A"
        }
    )
    fig.update_traces(textposition="inside", textfont=dict(size=11), cliponaxis=False, marker=dict(line=dict(color='black', width=1)))
    base_date = df_plot['start time'].min().date()
    start_range = pd.Timestamp(f"{base_date} 05:00:00")
    end_range = pd.Timestamp(f"{base_date} 01:00:00") + pd.Timedelta(days=1)
    fig.update_xaxes(range=[start_range, end_range], title_text="Time", showgrid=True, gridcolor="LightGray", tickformat="%H:%M")
    fig.update_yaxes(title="Bus", autorange="reversed", showgrid=True, gridcolor="LightGray")
    fig.update_layout(height=400 + 30*len(df_plot['bus'].unique()), margin=dict(l=20, r=20, t=40, b=20), dragmode='zoom')
    return fig


# === Tab 1: Gantt ===
with tab_gantt:
    st.subheader("📊 Gantt Chart")
    if uploaded_file:
        df = load_data(uploaded_file)
        bus_options = sorted(df['bus'].dropna().unique())
        selected_buses = st.multiselect(
            "Select one or multiple busses (or 'All busses')",
            options=[0] + bus_options,
            default=[0],
            format_func=lambda x: f"Bus {int(x)}" if x != 0 else "All busses",
            key="selected_buses"
        )
        fig = plot_gantt_interactive(df, selected_buses)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload an Excel file to display the Gantt chart.")


# === Tab 2: Visuals ===
with tab_visuals:
    st.subheader("📈 Visualisation")
    if uploaded_file:
        df = load_data(uploaded_file)

        # Wipe any old session_state junk
        for k in ("group_by_radio", "cap_kwh_visual", "start_soc_visual"):
            if k in st.session_state:
                del st.session_state[k]

        # --- fixed config (no widgets) ---
        group_by_display = "Bus"
        group_by = "bus"
        cap_kwh = 300.0
        start_soc = 100

        # make globally available for other tabs
        globals()['cap_kwh'] = cap_kwh
        globals()['start_soc'] = start_soc

        # optional bus filter
        opts = sorted(df[group_by].dropna().astype(str).unique().tolist())
        pick = st.multiselect("Select Bus(es)", options=opts, default=opts[:min(5, len(opts))], key="pick_groups")
        if pick:
            df = df[df[group_by].astype(str).isin(pick)]

        ts = "start time"
        d = df.copy()
        d["energy consumption"] = pd.to_numeric(d["energy consumption"], errors="coerce").fillna(0.0)

        def build_soc(g):
            g = g.sort_values(ts).copy()
            if not g.empty:
                baseline = g.iloc[[0]].copy()
                baseline["energy consumption"] = 0.0
                baseline[ts] = g[ts].min()
                g = pd.concat([baseline, g], ignore_index=True)
            g["net_kwh_cum"] = g["energy consumption"].cumsum()
            g["soc_%"] = (start_soc - (g["net_kwh_cum"] / cap_kwh) * 100).clip(0, 100)
            return g[[group_by, ts, "soc_%"]]

        soc_df = d.groupby(group_by, dropna=False).apply(build_soc).reset_index(drop=True)

        # Detect <10% SoC
        under10 = soc_df[soc_df["soc_%"] < 10]
        if not under10.empty:
            hits = (under10.sort_values(ts)
                    .groupby(group_by, as_index=False)
                    .first()[[group_by, ts, "soc_%"]])
            st.error("⚠️ Battery <10% — bus(es) unavailable: " + ", ".join(hits[group_by].astype(str).tolist()))
            st.toast("⚠️ Battery below 10% detected!", icon="⚡")
            with st.expander("First moment <10% per bus"):
                st.dataframe(hits, use_container_width=True)

        import plotly.graph_objects as go
        fig_soc = px.line(
            soc_df, x=ts, y="soc_%", color=group_by,
            title=f"SoH / SoC progression per {group_by_display}",
            labels={"soc_%": "SOC (%)", ts: "Time"}
        )
        fig_soc.update_yaxes(range=[0, 100])
        fig_soc.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        fig_soc.add_hline(y=10, line_dash="dot", annotation_text="10% threshold", annotation_position="bottom right")

        if not under10.empty:
            fig_soc.add_trace(go.Scatter(
                x=under10[ts], y=under10["soc_%"],
                mode="markers", marker=dict(size=8, color="red"),
                name="<10% SoC"
            ))

        st.plotly_chart(fig_soc, use_container_width=True)
        st.write("Example points (first 30):")
        st.dataframe(soc_df.sort_values([group_by, ts]).head(30), use_container_width=True)
        st.caption("Positive = consumption (SOC decreases), negative = charging (SOC increases).")
    else:
        st.info("Upload an Excel file to see the SOC graph.")



