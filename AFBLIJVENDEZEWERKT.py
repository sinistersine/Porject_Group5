# streamlit run dashboard.py  (dit is wat je in terminal typt om de app te runnen)
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
tab_gantt, tab_visuals, tab_analysis, tab_errors, tab_kpi = st.tabs(["📊 Gantt-chart", "📈 Visualisations", "🔍 Analysis", "🚨 Errors", "📊 KPI"])

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

def _normalize_activity(s):
    return s.astype(str).str.lower().str.strip()

def compute_activity_mix(df):
    d = df.copy()
    d['activity'] = _normalize_activity(d['activity'])
    keep = d['activity'].isin(['service trip', 'material trip', 'idle'])
    d = d[keep]
    mix = (
        d.groupby('activity', as_index=False)['duration_minutes']
         .sum()
         .rename(columns={'duration_minutes': 'minutes'})
    )
    total = mix['minutes'].sum()
    if total <= 0:
        mix['pct'] = 0.0
    else:
        mix['pct'] = 100 * mix['minutes'] / total
    for a in ['service trip', 'material trip', 'idle']:
        if a not in mix['activity'].values:
            mix = pd.concat([mix, pd.DataFrame([{'activity': a, 'minutes': 0.0, 'pct': 0.0}])], ignore_index=True)
    return mix

def count_buses(df):
    return int(pd.Series(df['bus']).dropna().nunique())

def count_buses_make_it(df, cap_kwh=300, start_soc=100):
    ok = 0
    for bus_id, g in df.groupby('bus'):
        diag = get_battery_diagnostics(g, cap_kwh, start_soc)
        if diag['ok']:
            ok += 1
    return ok

def count_service_trips_not_on_time(df, timetable):
    if timetable is None or df is None or df.empty:
        return 0
    v = 0
    for _, g in df.groupby('bus'):
        tt = get_timetable_diagnostics(g, timetable)
        if not tt['ok']:
            for viol in tt['violations']:
                curr = str(viol.get('curr_line', '')).lower()
                if curr == 'service trip':
                    v += 1
    return v

def compute_plan_kpi(df, timetable, cap_kwh=300, start_soc=100):
    d = df.copy()
    d['activity'] = _normalize_activity(d['activity'])

    total_minutes = float(d['duration_minutes'].sum() or 0.0)
    idle_minutes = float(d.loc[d['activity'] == 'idle', 'duration_minutes'].sum() or 0.0)
    idle_ratio = (idle_minutes / total_minutes) if total_minutes > 0 else 0.0

    n_buses = count_buses(d)
    ok_buses = count_buses_make_it(d, cap_kwh, start_soc)
    fail_buses = max(n_buses - ok_buses, 0)
    battery_penalty = (fail_buses / n_buses) * 20 if n_buses > 0 else 0.0

    late_service = count_service_trips_not_on_time(d, timetable)
    total_service_trips = int(d.loc[d['activity'] == 'service trip'].shape[0])
    sched_penalty = (late_service / total_service_trips) * 20 if total_service_trips > 0 else 0.0

    kpi = 100 - (idle_ratio * 50 + battery_penalty + sched_penalty)
    kpi = float(np.clip(kpi, 0, 100))

    return {
        'kpi': round(kpi, 1),
        'total_minutes': total_minutes,
        'idle_minutes': idle_minutes,
        'idle_ratio': idle_ratio,
        'n_buses': n_buses,
        'ok_buses': ok_buses,
        'late_service_trips': late_service,
        'total_service_trips': total_service_trips,
        'mix': compute_activity_mix(d)
    }

def render_plan_card(plan_title, df, timetable):
    import plotly.express as px
    st.markdown(f"### {plan_title}")

    if df is None or df.empty:
        st.info("No data.")
        return

    stats = compute_plan_kpi(df, timetable)

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

    st.metric(label="KPI score (plan)", value=f"{stats['kpi']}/100")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Service trips not on time", f"{stats['late_service_trips']}")
    with c2:
        st.metric("Buses in plan", f"{stats['n_buses']}")
    with c3:
        st.metric("Buses that make charge", f"{stats['ok_buses']} / {stats['n_buses']}")

# Tab 1: Gantt Chart
with tab_gantt:
    st.subheader("📊 Gantt Chart")

    # Veilig initialiseren
    df = None
    selected_buses = []

    if uploaded_file:
        try:
            # Data laden
            df = load_data(uploaded_file)

            # Bus multiselect
            bus_options = sorted(df['bus'].dropna().unique())
            selected_buses = st.multiselect(
                "Selecteer bus(s) (of 'All busses')",
                options=[0] + bus_options,
                default=[0],
                format_func=lambda x: f"Bus {int(x)}" if x != 0 else "All busses"
            )

            # Feasibility checks per bus
            timetable = None
            if 'uploaded_tt' in st.session_state:
                timetable = st.session_state['uploaded_tt']
            elif os.path.exists('Timetable.xlsx'):
                timetable = pd.read_excel('Timetable.xlsx', index_col=0)

            any_fail = False
            cap = st.session_state.get('cap_kwh', 300.0)
            soc0 = st.session_state.get('start_soc', 100)
            if 0 in selected_buses:
                buses_to_check = sorted(df['bus'].dropna().unique())
            else:
                buses_to_check = [b for b in selected_buses if b in df['bus'].values]

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

            # Plot Gantt chart
            fig = plot_gantt_interactive(df, selected_buses)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Kon de Gantt-chart niet weergeven: {e}")

    else:
        st.info("Upload een Excel-bestand in de sidebar om de Gantt Chart te zien.")
# Tab 2: Visualisations
with tab_visuals:
    st.subheader("📈 Visualisation")

    if uploaded_file:
        df = load_data(uploaded_file)

        # ===== Fixed setup: group by bus =====
        group_by_display = "Bus"
        group_by = "bus"

        cap_kwh = 300
        start_soc = 100

        # Optional filter on specific buses
        opts = sorted(df[group_by].dropna().astype(str).unique().tolist())
        pick = st.multiselect(
            f"Select {group_by_display}(es)",
            options=opts,
            default=opts[:min(5, len(opts))],
            key="pick_groups"
        )
        if pick:
            df = df[df[group_by].astype(str).isin(pick)]

        # ===== Build SOC curve =====
        ts = "start time"
        d = df.copy()
        d["energy consumption"] = pd.to_numeric(
            d["energy consumption"], errors="coerce"
        ).fillna(0.0)

        def build_soc(g):
            g = g.sort_values(ts).copy()
            # Add baseline point so the curve starts at 100%
            if not g.empty:
                baseline = g.iloc[[0]].copy()
                baseline["energy consumption"] = 0.0
                baseline[ts] = g[ts].min()
                g = pd.concat([baseline, g], ignore_index=True)

            g["net_kwh_cum"] = g["energy consumption"].cumsum()
            g["soc_%"] = (
                start_soc - (g["net_kwh_cum"] / cap_kwh) * 100
            ).clip(0, 100)
            return g[[group_by, ts, "soc_%"]]

        soc_df = (
            d.groupby(group_by, dropna=False)
            .apply(build_soc)
            .reset_index(drop=True)
        )

        # ===== Check: battery below 10% =====
        under10 = soc_df[soc_df["soc_%"] < 10]
        if not under10.empty:
            # First time each bus drops below 10%
            hits = (
                under10.sort_values(ts)
                .groupby(group_by, as_index=False)
                .first()[[group_by, ts, "soc_%"]]
            )
            # Display alert
            st.error(
                "⚠️ Battery below 10% — these buses are unavailable: "
                + ", ".join(hits[group_by].astype(str).tolist())
            )
            # Show detailed info
            with st.expander("First moment below 10% per bus"):
                st.dataframe(hits, use_container_width=True)

        import plotly.express as px
        import plotly.graph_objects as go

        fig_soc = px.line(
            soc_df,
            x=ts,
            y="soc_%",
            color=group_by,
            title=f"SoH / SoC progression per {group_by_display}",
            labels={"soc_%": "SOC (%)", ts: "Time"},
        )

        # Make it easier to read
        fig_soc.update_yaxes(range=[0, 100])
        fig_soc.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        # Add 10% threshold line
        fig_soc.add_hline(
            y=10,
            line_dash="dot",
            annotation_text="10% threshold",
            annotation_position="bottom right"
        )

        # Add red dots for points under 10%
        if not under10.empty:
            fig_soc.add_trace(
                go.Scatter(
                    x=under10[ts],
                    y=under10["soc_%"],
                    mode="markers",
                    marker=dict(size=8, color="red"),
                    name="<10% SoC"
                )
            )

        st.plotly_chart(fig_soc, use_container_width=True)

        st.caption(
            "Positive 'energy consumption' = usage (SOC decreases), negative = charging (SOC increases)."
        )

    else:
        st.info("Upload an Excel file in the sidebar to see the SOC graph.")


# Tab 3: Analysis
with tab_analysis:
    st.subheader("🔍 Analysis")
    if uploaded_file:
        df = load_data(uploaded_file)          

        # lijn als string (voor latere analyses)
        df['line_str'] = df['line'].astype(str).str.replace('.0', '', regex=False)

        # ===== Total duration per bus =====
        total_duration_per_bus = (
            df.groupby('bus', as_index=False)['duration_minutes']
              .sum()
              .rename(columns={'duration_minutes': 'total_duration_minutes'})
        )

        # ===== Energie-analyse =====
        if 'energy consumption' in df.columns:
            per_bus = df.groupby('bus', as_index=False).agg(
                consumption_kWh=('energy consumption', lambda s: s.clip(lower=0).sum())
            )

            # ===== Merge total duration + energie =====
            bus_summary = pd.merge(total_duration_per_bus, per_bus, on='bus', how='outer')

            st.write("### Total duration + Energy per bus")
            st.dataframe(bus_summary.sort_values('bus'), use_container_width=True)

            # ===== Summary per bus per activity (zonder energy) =====
            st.write("### Summary per bus per activity")
            summary = (
                df.groupby(['bus', 'activity'], dropna=False)
                .agg(
                    num_trips=('activity', 'count'),
                    total_duration=('duration_minutes', 'sum')
                )
                .reset_index()
            )

            # eventueel afronden
            summary = summary.round({'total_duration': 2})

            # Pivot zodat per bus de activiteiten als kolommen komen
            pivot_summary = summary.pivot(index='bus', columns='activity', values=['num_trips','total_duration'])

            # Flatten multiindex kolommen voor leesbaarheid
            pivot_summary.columns = [f"{agg}_{act}" for agg, act in pivot_summary.columns]

            st.dataframe(pivot_summary.sort_values('bus'), use_container_width=True)


    else:
        st.info("Upload an Excel file in the sidebar to see the analysis.")


# Tab 4: Fouten
# hier kunnen we alle constraints in zetten waar alle data aan moet voldoen
# en als er iets niet klopt, dat dat hier getoond wordt
with tab_errors:
    st.subheader("🚨 Errors")

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
                        

# Tab 5: KPI Dashboard (plan vs plan)
with tab_kpi:
    st.subheader("📊 KPI — Old vs New busplan")

    # 1) Één timetable voor beide plannen (voor 'on-time' checks)
    uploaded_tt_file = st.file_uploader(
        "Upload Timetable (Excel) — gebruikt voor on-time checks",
        type=["xlsx"],
        key="tt_upload_global"
    )
    timetable = None
    if uploaded_tt_file:
        try:
            timetable = pd.read_excel(uploaded_tt_file, index_col=0)
            st.success("Timetable uploaded successfully!")
        except Exception as e:
            st.error(f"Timetable lezen faalde: {e}")

    # 2) Links: OUD plan (komt uit de sidebar-uploader). Rechts: NIEUW plan (extra uploader).
    col_old, col_new = st.columns(2, gap="large")

    # --- OLD PLAN (links) ---
    with col_old:
        st.markdown("#### Old busplan (from sidebar)")
        if uploaded_file:
            try:
                old_df = load_data(uploaded_file)
                # toont: Pie (service/material/idle), KPI score (plan), + 3 metrics
                render_plan_card("OLD — KPI", old_df, timetable)
            except Exception as e:
                st.error(f"Old busplan verwerken mislukte: {e}")
        else:
            st.info("Upload het **oude** busplan in de sidebar om KPIs te zien.")

    # --- NEW PLAN (rechts) ---
    with col_new:
        st.markdown("#### New busplan (upload hier)")
        new_upload = st.file_uploader(
            "Upload NEW busplan (Excel)",
            type=["xlsx"],
            key="new_busplan_upload"
        )
        if new_upload:
            try:
                new_df = load_data(new_upload)
                # idem: Pie + KPI + metrics
                render_plan_card("NEW — KPI", new_df, timetable)
            except Exception as e:
                st.error(f"New busplan verwerken mislukte: {e}")
        else:
            st.info("Upload het **nieuwe** busplan om te vergelijken.")
            
    # Tab 5: KPI Dashboard
with tab_kpi:
    st.subheader("📊 KPI")

    if uploaded_file:
        df = load_data(uploaded_file)

        # Upload timetable bestand
        uploaded_tt_file = st.file_uploader("Upload Timetable (Excel)", type=["xlsx"], key="tt_upload")
        timetable = None
        if uploaded_tt_file:
            timetable = pd.read_excel(uploaded_tt_file, index_col=0)
            st.success("Timetable uploaded successfully!")

        # Bereken KPI zoals eerder
        total_time = df.groupby('bus')['duration_minutes'].sum().reset_index().rename(columns={'duration_minutes':'total_minutes'})
        idle_time = df[df['activity'].str.lower()=='idle'].groupby('bus')['duration_minutes'].sum().reset_index().rename(columns={'duration_minutes':'idle_minutes'})
        kpi_df = pd.merge(total_time, idle_time, on='bus', how='left').fillna(0)
        kpi_df['idle_ratio'] = kpi_df['idle_minutes'] / kpi_df['total_minutes']

        # Battery violations
        battery_violations = []
        for bus_id, group in df.groupby('bus'):
            batt_diag = get_battery_diagnostics(group, cap_kwh=300, start_soc_percent=100)
            battery_violations.append(0 if batt_diag['ok'] else 1)
        kpi_df['battery_violation'] = battery_violations

        # Schedule violations (timetable checks)
        schedule_violations = []
        timetable_violations_detail = {}
        for bus_id, group in df.groupby('bus'):
            if timetable is not None:
                tt_diag = get_timetable_diagnostics(group, timetable)
                schedule_violations.append(0 if tt_diag['ok'] else 1)
                timetable_violations_detail[bus_id] = tt_diag['violations']
            else:
                schedule_violations.append(0)
        kpi_df['schedule_violation'] = schedule_violations

        # KPI score
        kpi_df['kpi_score'] = 100 - (kpi_df['idle_ratio'] * 50 + kpi_df['battery_violation'] * 20 + kpi_df['schedule_violation'] * 20)
        kpi_df['kpi_score'] = kpi_df['kpi_score'].clip(0,100)

        # --- Verwijder bar chart per bus ---
        # fig_kpi = px.bar(...)
        # st.plotly_chart(fig_kpi, use_container_width=True)  # <-- weggehaald


        # Totaal aantal bussen
        n_buses = len(kpi_df)

        # Idle penalty = idle_ratio * 20 per bus
        total_idle_penalty = (kpi_df['idle_ratio'] * 20).sum()

        # Battery penalty = battery_violation * 40 per bus
        total_battery_penalty = (kpi_df['battery_violation'] * 40).sum()

        # Schedule penalty = schedule_violation * 40 per bus
        total_schedule_penalty = (kpi_df['schedule_violation'] * 40).sum()

        # Remaining score = 100 per bus minus alle penalties
        total_remaining_score = 100 * n_buses - (total_idle_penalty + total_battery_penalty + total_schedule_penalty)

        labels = ['Idle Penalty', 'Battery Violation', 'Schedule Violation', 'KPI Score']
        values = [total_idle_penalty, total_battery_penalty, total_schedule_penalty, total_remaining_score]

        # KPI Score voor het hele busplan
        total_penalties = total_idle_penalty + total_battery_penalty + total_schedule_penalty
        overall_kpi_score = 100 - (total_penalties / n_buses)
        overall_kpi_score = max(0, min(overall_kpi_score, 100))  # clip tussen 0 en 100

        # --- Zet trofee en KPI-score boven pie chart ---
        st.markdown(f"### 🏆 Overall KPI Score for Bus Plan: **{overall_kpi_score:.2f} / 100**")

        fig_pie_total = px.pie(
            names=labels,
            values=values,
            title="Overall KPI Breakdown – Entire Bus Plan",
            color_discrete_map={
                'Idle Penalty': 'orange',
                'Battery Violation': 'red',
                'Schedule Violation': 'purple',
                'Remaining Score': 'green'
            }
        )

        fig_pie_total.update_traces(textinfo='percent+label')

        st.plotly_chart(fig_pie_total, use_container_width=True)

    else:
        st.info("Upload een Excel-bestand met het busplan in de sidebar om KPI's te bekijken.")

#