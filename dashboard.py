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

# Colors (kept as fallback if used in plot mapping)
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
st.set_page_config(page_title="Bus Planning dashboard", layout="wide")
st.title("🚌 Bus Planning dashboard")

# Sidebar: upload Excel-bestand
uploaded_file = st.sidebar.file_uploader("1) Upload the busplan (Excel)", type=["xlsx"], key="busplan")

# Tabs bovenaan
tab_gantt, tab_visuals, tab_analysis, tab_errors = st.tabs(["📊 Gantt-chart", "📈 Visualizations", "🔍 Analysis", "🚨 Errors"])

# Functie om Gantt Chart te plotten (één of meerdere bussen)
def plot_gantt_interactive(df, selected_buses=None):
    df_plot = df.copy()

    if selected_buses and 0 not in selected_buses:
        df_plot = df_plot[df_plot['bus'].isin(selected_buses)]

    # Label voor service trips → verticale layout via <br>
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

    # Zorg dat tekst IN de balk staat en niet wordt afgekapt
    fig.update_traces(
        textposition="inside",
        textfont=dict(size=11),
        cliponaxis=False,   # helpt voorkomen dat tekst geknipt wordt
        marker=dict(line=dict(color='black', width=1))
    )

    base_date = df_plot['start time'].min().date()
    start_range = pd.Timestamp(f"{base_date} 05:00:00")
    end_range = pd.Timestamp(f"{base_date} 01:00:00") + pd.Timedelta(days=1)
    fig.update_xaxes(range=[start_range, end_range], title_text="Time", showgrid=True, gridcolor="LightGray", tickformat="%H:%M")
    fig.update_yaxes(title="Bus", autorange="reversed", showgrid=True, gridcolor="LightGray")

    fig.update_layout(height=400 + 30*len(df_plot['bus'].unique()), margin=dict(l=20, r=20, t=40, b=20), dragmode='zoom')

    return fig


# Helper checks
def check_bus_battery_survival(bus_df, cap_kwh, start_soc_percent):
    battery = cap_kwh * (start_soc_percent / 100.0)
    bus_df = bus_df.sort_values("start time").reset_index(drop=True)
    for _, row in bus_df.iterrows():
        try:
            cons = float(row.get("energy consumption", 0) or 0)
        except Exception:
            cons = 0.0
        act = str(row.get("activity", "")).lower()
        is_charging = ("charge" in act) or ("charging" in act) or (cons < 0)
        if is_charging:
            battery = min(cap_kwh, battery + abs(cons))
        else:
            battery -= cons
        if battery < -1e-6:
            return False, "The bus won't make it with the battery and charging moments, pick another busplan."
    return True, "Battery OK"


def check_bus_timetable_feasible(bus_df, timetable_df):
    bus_df = bus_df.sort_values("start time").reset_index(drop=True)
    def lookup_travel_minutes(a, b):
        try:
            return float(timetable_df.loc[a, b])
        except Exception:
            return None
    for i in range(1, len(bus_df)):
        prev = bus_df.iloc[i - 1]
        curr = bus_df.iloc[i]
        prev_end = prev.get("end time")
        curr_start = curr.get("start time")
        if pd.isna(prev_end) or pd.isna(curr_start):
            return False, "The bus won't arrive on time, pick another bus plan"
        required_minutes = (curr_start - prev_end).total_seconds() / 60.0
        expected = lookup_travel_minutes(prev.get('end location'), curr.get('start location'))
        if expected is None:
            return False, "The bus won't arrive on time, pick another bus plan"
        if required_minutes < expected - 1e-6:
            return False, "The bus won't arrive on time, pick another bus plan"
    return True, "Timing OK"


def get_battery_diagnostics(bus_df, cap_kwh, start_soc_percent):
    """
    Return detailed battery simulation trace and a summary.
    Returns dict with keys: ok(bool), message(str), trace(pd.DataFrame)
    Trace columns: start time, end time, activity, energy consumption, batt_before_kwh, batt_after_kwh, note
    """
    battery = float(cap_kwh) * (float(start_soc_percent) / 100.0)
    rows = []
    g = bus_df.sort_values('start time').reset_index(drop=True).copy()
    for idx, r in g.iterrows():
        try:
            cons = float(r.get('energy consumption', 0) or 0)
        except Exception:
            cons = 0.0
        act = str(r.get('activity', '')).lower()
        is_charging = ('charge' in act) or ('charging' in act) or (cons < 0)
        before = battery
        note = ''
        if is_charging:
            battery = min(cap_kwh, battery + abs(cons))
            note = 'charging'
        else:
            battery = battery - cons
            if battery < 0:
                note = 'below_zero'
        after = battery
        rows.append({
            'start time': r.get('start time'),
            'end time': r.get('end time'),
            'activity': r.get('activity'),
            'energy consumption': cons,
            'batt_before_kwh': round(before, 3),
            'batt_after_kwh': round(after, 3),
            'note': note
        })

    trace = pd.DataFrame(rows)
    # find first violation
    viol = trace[trace['note'] == 'below_zero']
    if not viol.empty:
        first = viol.iloc[0]
        msg = f"Battery dies during activity at {first['start time']} (bus runs out of kWh)."
        return {'ok': False, 'message': msg, 'trace': trace}
    return {'ok': True, 'message': 'Battery OK', 'trace': trace}


def get_timetable_diagnostics(bus_df, timetable_df):
    """
    Return detailed timetable check diagnostics.
    Returns dict with keys: ok(bool), message(str), violations(list of dict)
    Each violation dict contains: idx, prev_end, curr_start, prev_end_loc, curr_start_loc, required_min, expected_min, reason
    """
    g = bus_df.sort_values('start time').reset_index(drop=True).copy()
    violations = []
    def lookup(a, b):
        try:
            return float(timetable_df.loc[a, b])
        except Exception:
            return None

    for i in range(1, len(g)):
        prev = g.iloc[i - 1]
        curr = g.iloc[i]
        prev_end = prev.get('end time')
        curr_start = curr.get('start time')
        if pd.isna(prev_end) or pd.isna(curr_start):
            violations.append({'idx': i, 'reason': 'missing_time', 'prev_end': prev_end, 'curr_start': curr_start})
            continue
        required = (curr_start - prev_end).total_seconds() / 60.0
        expected = lookup(prev.get('end location'), curr.get('start location'))
        if expected is None:
            violations.append({'idx': i, 'reason': 'missing_timetable', 'prev_end_loc': prev.get('end location'), 'curr_start_loc': curr.get('start location')})
            continue
        if required < expected - 1e-6:
            violations.append({'idx': i, 'reason': 'insufficient_travel_time', 'prev_end': prev_end, 'curr_start': curr_start, 'required_min': required, 'expected_min': expected, 'prev_line': prev.get('activity'), 'curr_line': curr.get('activity')})

    if violations:
        return {'ok': False, 'message': f"{len(violations)} timetable violation(s)", 'violations': violations}
    return {'ok': True, 'message': 'Timing OK', 'violations': []}

with tab_gantt:
    st.subheader("📊 Gantt Chart")
    if uploaded_file:
        df = load_data(uploaded_file)

        # Lijst van unieke bussen
        bus_options = sorted(df['bus'].dropna().unique())
        selected_buses = st.multiselect(
            "Select one or multiple busses (or 'All busses')",
            options=[0] + bus_options,
            default=[0],
            format_func=lambda x: f"Bus {int(x)}" if x != 0 else "All busses",
            key="selected_buses"
        )



        # ---- Run feasibility checks for displayed buses ----
        # determine bus ids to check
        if 0 in selected_buses or not selected_buses:
            buses_to_check = sorted(df['bus'].dropna().unique())
        else:
            buses_to_check = [b for b in selected_buses if b in df['bus'].values]

        # try to load timetable from uploaded or local file
        timetable = None
        try:
            g = globals()
            if 'uploaded_tt' in g and g['uploaded_tt'] is not None:
                timetable = pd.read_excel(g['uploaded_tt'], index_col=0)
            elif os.path.exists('Timetable.xlsx'):
                timetable = pd.read_excel('Timetable.xlsx', index_col=0)
            else:
                timetable = None
        except Exception:
            timetable = None

        any_fail = False
        # read cap_kwh and start_soc if available (from visualization controls), otherwise defaults
        g = globals()
        cap = g.get('cap_kwh', 300.0)
        soc0 = g.get('start_soc', 100)
        for bus_id in buses_to_check:
            bus_df = df[df['bus'] == bus_id].copy()
            ok_batt, batt_msg = check_bus_battery_survival(bus_df, cap, soc0)
            if not ok_batt:
                st.error(batt_msg)
                any_fail = True
                break
            if timetable is not None:
                ok_time, time_msg = check_bus_timetable_feasible(bus_df, timetable)
                if not ok_time:
                    st.error(time_msg)
                    any_fail = True
                    break

        if not any_fail:
            st.success("Selected busplan(s) appear feasible.")

    fig = plot_gantt_interactive(df, selected_buses)
    st.plotly_chart(fig, use_container_width=True)


# Tab 2: Visualisaties
with tab_visuals:
    st.subheader("📈 Visualisation")
    if uploaded_file:
        df = load_data(uploaded_file)

        # ===== Controls =====
        group_options = {"Bus": "bus", "Line": "line"}
        group_by_display = st.radio("Group by ", list(group_options.keys()), horizontal=True, key="group_by_radio")
        group_by = group_options[group_by_display]
        cap_kwh = st.number_input("Battery capacity (kWh)", min_value=50.0, max_value=1000.0, value=300.0, step=10.0, key="cap_kwh_visual")
        start_soc = st.slider("Start-SOC (%)", min_value=0, max_value=100, value=100, step=1, key="start_soc_visual")

        # optioneel filteren op specifieke bussen/lijnen
        opts = sorted(df[group_by].dropna().astype(str).unique().tolist())
        pick = st.multiselect(f"Select {group_by_display}(es)", options=opts, default=opts[:min(5, len(opts))], key="pick_groups")
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
            title=f"SoH / SoC progression per {group_by_display}",
            labels={'soc_%': 'SOC (%)', ts: 'Time'}
        )
        # Maak het wat leesbaarder
        fig_soc.update_yaxes(range=[0, 100])
        fig_soc.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))

        st.plotly_chart(fig_soc, use_container_width=True)

        # klein tabelletje erbij zodat je kan checken
        st.write("Example points (first 30):")
        st.dataframe(soc_df.sort_values([group_by, ts]).head(30), use_container_width=True)

        st.caption("Positive 'energy consumption' = consumption (SOC decreases), negative = charging (SOC increases).")
    else:
        st.info("Upload an Excel file in the sidebar to see the SOC graph.")

# Tab 3: Analysis
with tab_analysis:
    st.subheader("🔍 Analysis")
    if uploaded_file:
        df = load_data(uploaded_file)          


        # === NIEUW BLOK: Gemiddelde duur per activiteit per lijn ===
        df['line_str'] = df['line'].astype(str).str.replace('.0', '', regex=False)

        avg_duration_per_line_activity = (
            df.groupby(['line_str', 'activity'], dropna=False)['duration_minutes']
              .mean()
              .reset_index()
              .rename(columns={'line_str': 'line'})
        )

        st.write("### Average duration per activity **per bus** (in minutes)")
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
                ).round(2)
             .rename(columns=lambda c: ACTIVITY_DISPLAY.get(str(c).lower(), c)),
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



        st.write("### Total duration per bus (in minutes)")
        total_duration_per_bus = (df.groupby('bus', as_index=False)['duration_minutes']
                                    .sum()
                                    .rename(columns={'duration_minutes': 'total_duration_minutes'}))
        st.dataframe(total_duration_per_bus)

        # Energie-analyse (nu al gefixt voor material trips)
        if 'energy consumption' in df.columns:
            per_bus = df.groupby('bus', as_index=False).agg(
                consumption_kWh=('energy consumption', lambda s: s.clip(lower=0).sum()),
                charged_kWh =('energy consumption', lambda s: (-s.clip(upper=0)).sum()),
                netto_kWh   =('energy consumption', 'sum'),
            )
            BATTERY_KWH = 300.0
            per_bus['end_SOC_%'] = (100 - (per_bus['netto_kWh'] / BATTERY_KWH) * 100).clip(0, 100)

            st.write("### Energy per bus (kWh) + end-SOC estimate")
            st.dataframe(per_bus.sort_values('netto_kWh', ascending=False), use_container_width=True)

            st.write("### Audit: inspection of material trips")
            audit_display = df.loc[df['energy_expected_material'].notna(),
                                   ['activity','start time','end time','duration_minutes',
                                    'energy_expected_material','energy consumption',
                                    'energy_diff','energy_match']].copy()
            audit_display['activity'] = audit_display['activity'].astype(str).str.lower().map(ACTIVITY_DISPLAY).fillna(audit_display['activity'])
            st.dataframe(audit_display)
        else:
            st.info("Column 'energy consumption' (energy usage) wasn't found in the file.")
    else:
        st.info("Upload an Excel file in the sidebar to see the analysis.")

# Tab 4: Fouten
# hier kunnen we alle constraints in zetten waar alle data aan moet voldoen
# en als er iets niet klopt, dat dat hier getoond wordt
with tab_errors:
    st.subheader("🚨 Errors")

    uploaded_dist = st.file_uploader("Upload Distance Matrix Excel", type=["xlsx"], key="dist")
    uploaded_tt = st.file_uploader("Upload Timetable Excel", type=["xlsx"], key="tt")

    st.write("Here is a list of errors detected in the planning (feasibility checks).")

        # Option: only show buses selected in the Gantt
    only_selected = st.checkbox("Toon alleen bussen geselecteerd in Gantt", value=False, key="errors_only_selected")

    # ---- Load bus plan and timetable, then run per-bus checks ----
    try:
        if uploaded_file is not None:
            busplan = load_data(uploaded_file)
        elif os.path.exists('Bus Planning.xlsx'):
            busplan = load_data('Bus Planning.xlsx')
        else:
            busplan = None
    except Exception as e:
        st.error(f"Could not load bus plan: {e}")
        busplan = None

    # load timetable (if provided)
    try:
        if uploaded_tt is not None:
            timetable = pd.read_excel(uploaded_tt, index_col=0)
        elif os.path.exists('Timetable.xlsx'):
            timetable = pd.read_excel('Timetable.xlsx', index_col=0)
        else:
            timetable = None
    except Exception as e:
        st.error(f"Could not load timetable: {e}")
        timetable = None

    if busplan is None:
        st.info("No bus plan loaded. Upload the bus plan in the sidebar or place 'Bus Planning.xlsx' in the app folder.")
    else:
        # ensure time columns parsed (load_data already handles this)
        for col in ['start time', 'end time']:
            if col in busplan.columns:
                busplan[col] = pd.to_datetime(busplan[col], errors='coerce')

        # collect diagnostics per bus
        diagnostics = {}
        cap_kwh = 300.0
        start_soc = 100
        for bus, group in busplan.groupby('bus'):
            group = group.sort_values('start time').reset_index(drop=True)
            batt_diag = get_battery_diagnostics(group, cap_kwh, start_soc)
            tt_diag = None
            if timetable is not None:
                tt_diag = get_timetable_diagnostics(group, timetable)
            diagnostics[bus] = {'battery': batt_diag, 'timetable': tt_diag, 'group': group}

        # optionally filter by the Gantt selection
        buses_to_show = sorted(diagnostics.keys())
        if only_selected:
            sel = st.session_state.get('selected_buses', None)
            if sel and 0 not in sel:
                buses_to_show = [b for b in buses_to_show if b in sel]

        # display per-bus expanders with diagnostics
        st.write("### Detailed diagnostics per bus")
        if not buses_to_show:
            st.info("Geen bussen om te tonen met de huidige filterinstellingen.")
        for bus in buses_to_show:
            d = diagnostics[bus]
            batt = d['battery']
            tt = d['timetable']
            with st.expander(f"Bus {bus} — {batt['message'] if batt else 'No battery data'}", expanded=False):
                st.write("**Battery simulation**")
                st.write(batt['message'])
                st.dataframe(batt['trace'].sort_values('start time').reset_index(drop=True), use_container_width=True)

                if tt is not None:
                    st.write("---")
                    st.write("**Timetable checks**")
                    st.write(tt['message'])
                    if tt['violations']:
                        # show violations as a small dataframe
                        vt = pd.DataFrame(tt['violations'])
                        st.dataframe(vt, use_container_width=True)
                    else:
                        st.write("No timetable violations detected.")
