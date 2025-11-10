# streamlit run CodemetandereKPI.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

try:
    import plotly.express as px  # noqa: F401
except ModuleNotFoundError:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly==5.24.1"])
    import plotly.express as px  # noqa: F401
# ===================== PAGE =====================
st.set_page_config(page_title="Bus Planning dashboard", layout="wide")
st.title("🚌 Bus Planning dashboard")

# ===================== CONSTANTS =====================
BASE_MIN, BASE_KWH = 20, 10.32   # 20 min → 10.32 kWh  (material trips)
TOL = 0.05
BUFFER_MIN = 4                   # verplichte extra marge in timetable check
CAP_DEFAULT = 300.0
SOC0_DEFAULT = 100

ACTIVITY_DISPLAY = {
    'service trip': 'service trip',
    'material trip': 'material trip',
    'idle': 'idle time',
    'charging': 'charging time'
}

ACTIVITY_COLORS = {
    'service trip': '#664EDC',
    'material trip': '#D904B2',
    'idle': '#E7DF12',
    'charging': '#0DD11A'
}

# ===================== HELPERS =====================
def _norm_str(s):
    return str(s).strip().lower() if pd.notna(s) else s

def _normalize_activity(s):
    return s.astype(str).str.lower().str.strip()

def _safe_to_dt(series):
    t = pd.to_datetime(series, format="%H:%M:%S", errors="coerce")
    return t

def _ensure_rollover(start, end):
    # als end < start → middernacht-overgang
    roll_mask = (end < start) & start.notna() & end.notna()
    end_fix = end + pd.to_timedelta(roll_mask.astype(int), unit='D')
    return end_fix

# ===================== LOADERS =====================
@st.cache_data
def load_data(file):
    df = pd.read_excel(file, sheet_name="Sheet1")

    # times -> datetime
    start = _safe_to_dt(df['start time'])
    end   = _safe_to_dt(df['end time'])
    end   = _ensure_rollover(start, end)

    df['duration_minutes'] = (end - start).dt.total_seconds() / 60.0

    # energy normalize (comma → dot)
    if 'energy consumption' in df.columns:
        df['energy consumption'] = pd.to_numeric(
            df['energy consumption'].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        )
    else:
        df['energy consumption'] = np.nan

    # material rule (fix values)
    is_mat = df['activity'].astype(str).str.lower().str.contains('material')
    exp = BASE_KWH * (df['duration_minutes'] / BASE_MIN)
    df.loc[is_mat, 'energy consumption'] = exp.round(3)

    # write back parsed times
    df['start time'] = start
    df['end time']   = end
    return df

@st.cache_data
def load_timetable(file):
    tt = pd.read_excel(file, index_col=0)
    # normalize labels: lowercase/trim in both axes
    tt.index = tt.index.astype(str).str.strip().str.lower()
    tt.columns = tt.columns.astype(str).str.strip().str.lower()
    # numeric minutes
    tt = tt.apply(pd.to_numeric, errors='coerce')
    return tt

# ===================== TIMETABLE CHECKS =====================
def lookup_tt_minutes(tt, a, b):
    """Symmetrische lookup + label-normalisatie. Return minutes (float) of None."""
    if tt is None or pd.isna(a) or pd.isna(b):
        return None
    A = _norm_str(a)
    B = _norm_str(b)
    if A == B:
        return 0.0
    # try A→B
    try:
        val = float(tt.loc[A, B])
        if pd.notna(val):
            return val
    except Exception:
        pass
    # try B→A (soms alleen andere richting aanwezig)
    try:
        val = float(tt.loc[B, A])
        if pd.notna(val):
            return val
    except Exception:
        pass
    return None

def get_timetable_diagnostics(bus_df, timetable_df, buffer_min=BUFFER_MIN):
    """
    Returns: dict(ok, message, violations[list])
    violation item: {idx, prev_end, curr_start, prev_end_loc, curr_start_loc, required_min, expected_min, need_min, reason}
    """
    g = bus_df.sort_values('start time').reset_index(drop=True).copy()
    violations = []

    for i in range(1, len(g)):
        prev = g.iloc[i - 1]
        curr = g.iloc[i]
        prev_end = prev.get('end time')
        curr_start = curr.get('start time')

        if pd.isna(prev_end) or pd.isna(curr_start):
            violations.append({'idx': i, 'reason': 'missing_time', 'prev_end': prev_end, 'curr_start': curr_start})
            continue

        required = (curr_start - prev_end).total_seconds() / 60.0
        exp = lookup_tt_minutes(timetable_df, prev.get('end location'), curr.get('start location'))

        if exp is None:
            violations.append({
                'idx': i,
                'reason': 'missing_timetable',
                'prev_end_loc': prev.get('end location'),
                'curr_start_loc': curr.get('start location'),
                'required_min': round(required, 2)
            })
            continue

        need = exp + buffer_min
        if required + 1e-6 < need:
            violations.append({
                'idx': i,
                'reason': 'insufficient_travel_time',
                'prev_end': prev_end, 'curr_start': curr_start,
                'prev_end_loc': prev.get('end location'),
                'curr_start_loc': curr.get('start location'),
                'required_min': round(required, 2),
                'expected_min': round(exp, 2),
                'need_min': round(need, 2)
            })

    if violations:
        return {'ok': False, 'message': f"{len(violations)} timetable violation(s)", 'violations': violations}
    return {'ok': True, 'message': 'Timing OK', 'violations': []}

# ===================== BATTERY CHECKS =====================
def get_battery_diagnostics(bus_df, cap_kwh=CAP_DEFAULT, start_soc_percent=SOC0_DEFAULT):
    """
    Simpele kWh-simulatie over de dag.
    charging: negative kWh → SOC omhoog, anders omlaag.
    """
    battery = float(cap_kwh) * (float(start_soc_percent) / 100.0)
    rows = []
    g = bus_df.sort_values('start time').reset_index(drop=True).copy()
    for idx, r in g.iterrows():
        cons = pd.to_numeric(r.get('energy consumption', 0), errors='coerce')
        if pd.isna(cons):
            cons = 0.0
        act = str(r.get('activity', '')).lower()
        is_charging = ('charge' in act) or (cons < 0)
        before = battery
        note = ''
        if is_charging:
            battery = min(cap_kwh, battery + abs(cons))
            note = 'charging'
        else:
            battery = battery - cons
            if battery < -1e-6:
                note = 'below_zero'
        rows.append({
            'start time': r.get('start time'),
            'end time': r.get('end time'),
            'activity': r.get('activity'),
            'energy consumption': cons,
            'batt_before_kwh': round(before, 3),
            'batt_after_kwh': round(battery, 3),
            'note': note
        })
    trace = pd.DataFrame(rows)
    viol = trace[trace['note'] == 'below_zero']
    if not viol.empty:
        first = viol.iloc[0]
        msg = f"Battery dies during activity at {first['start time']}."
        return {'ok': False, 'message': msg, 'trace': trace}
    return {'ok': True, 'message': 'Battery OK', 'trace': trace}

# ===================== KPI CORE =====================
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
    mix['pct'] = 0.0 if total <= 0 else 100 * mix['minutes'] / total
    for a in ['service trip', 'material trip', 'idle']:
        if a not in mix['activity'].values:
            mix = pd.concat([mix, pd.DataFrame([{'activity': a, 'minutes': 0.0, 'pct': 0.0}])], ignore_index=True)
    return mix

def compute_plan_kpi(df, timetable, cap_kwh=CAP_DEFAULT, start_soc=SOC0_DEFAULT):
    d = df.copy()
    d['activity'] = _normalize_activity(d['activity'])

    total_minutes = float(d['duration_minutes'].sum() or 0.0)
    idle_minutes  = float(d.loc[d['activity'] == 'idle', 'duration_minutes'].sum() or 0.0)
    idle_ratio    = (idle_minutes / total_minutes) if total_minutes > 0 else 0.0

    # Battery violation rate (bus-level, 0/1 per bus)
    n_buses = int(pd.Series(d['bus']).dropna().nunique())
    batt_viol = 0
    sched_viol_transitions = 0
    total_transitions = 0

    for bus_id, g in d.groupby('bus'):
        batt = get_battery_diagnostics(g, cap_kwh, start_soc)
        if not batt['ok']:
            batt_viol += 1

        tt_diag = get_timetable_diagnostics(g, timetable, BUFFER_MIN) if timetable is not None else {'ok': True, 'violations': []}
        # transitions for this bus = len(g)-1 (non-negative)
        trans = max(len(g) - 1, 0)
        total_transitions += trans
        sched_viol_transitions += 0 if tt_diag['ok'] else len(tt_diag['violations'])

    battery_penalty = (batt_viol / n_buses) * 20.0 if n_buses > 0 else 0.0
    sched_ratio = (sched_viol_transitions / total_transitions) if total_transitions > 0 else 0.0
    schedule_penalty = sched_ratio * 20.0

    kpi = 100 - (idle_ratio * 50.0 + battery_penalty + schedule_penalty)
    kpi = float(np.clip(kpi, 0, 100))

    return {
        'kpi': round(kpi, 1),
        'idle_ratio': idle_ratio,
        'n_buses': n_buses,
        'battery_buses_failed': batt_viol,
        'battery_buses_total': n_buses,
        'schedule_violations': sched_viol_transitions,
        'schedule_transitions': total_transitions,
        'mix': compute_activity_mix(d)
    }

# ===================== UI STATE =====================
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'timetable_df' not in st.session_state:
    st.session_state['timetable_df'] = None

# ===================== SIDEBAR =====================
with st.sidebar:
    uploaded_file = st.file_uploader("1) Upload the busplan (Excel)", type=["xlsx"], key="busplan")
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.session_state['df'] = df
            st.success("Busplan loaded!")
            with st.expander("Preview data"):
                st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading busplan: {str(e)}")
            st.session_state['df'] = None
    else:
        st.info("Upload a busplan Excel to begin.")

    uploaded_tt = st.file_uploader("2) Upload Timetable (Excel)", type=["xlsx"], key="uploaded_tt")
    if uploaded_tt is not None:
        try:
            timetable_df = load_timetable(uploaded_tt)
            st.session_state['timetable_df'] = timetable_df
            st.success("Timetable loaded!")
        except Exception as e:
            st.error(f"Error loading timetable: {str(e)}")
            st.session_state['timetable_df'] = None

uploaded_tt = st.session_state.get('timetable_df', None)

# ===================== TABS =====================
tab_gantt, tab_visuals, tab_analysis, tab_errors, tab_kpi = st.tabs(
    ["📊 Gantt-chart", "📈 Visualisations", "🔍 Analysis", "🚨 Errors", "📊 KPI Dashboard"]
)

# ===================== GANTT =====================
def plot_gantt_interactive(df, selected_buses=None):
    df_plot = df.copy()
    if selected_buses and 0 not in selected_buses:
        df_plot = df_plot[df_plot['bus'].isin(selected_buses)]

    # line label (optional)
    def _lab(row):
        if (row.get('activity') == 'service trip') and pd.notna(row.get('line')):
            try:
                return str(int(row['line']))
            except Exception:
                return str(row['line'])
        return ''
    df_plot['label'] = df_plot.apply(_lab, axis=1)
    df_plot['row'] = df_plot['bus'].apply(lambda x: f"Bus {int(x)}" if pd.notna(x) else "Unknown")

    fig = px.timeline(
        df_plot,
        x_start="start time",
        x_end="end time",
        y="row",
        color="activity",
        text="label",
        title="Gantt Chart – Bus Planning",
        color_discrete_map=ACTIVITY_COLORS
    )
    fig.update_traces(textposition="inside", textfont=dict(size=11), cliponaxis=False, marker=dict(line=dict(color='black', width=1)))
    base_date = df_plot['start time'].min().date()
    start_range = pd.Timestamp(f"{base_date} 05:00:00")
    end_range = pd.Timestamp(f"{base_date} 01:00:00") + pd.Timedelta(days=1)
    fig.update_xaxes(range=[start_range, end_range], title_text="Time", showgrid=True, gridcolor="LightGray", tickformat="%H:%M")
    fig.update_yaxes(title="Bus", autorange="reversed", showgrid=True, gridcolor="LightGray")
    fig.update_layout(height=400 + 30*len(df_plot['bus'].dropna().unique()), margin=dict(l=20, r=20, t=40, b=20), dragmode='zoom')
    return fig

with tab_gantt:
    st.subheader("📊 Gantt Chart")
    df_session = st.session_state.get('df')
    selected_buses = []
    if df_session is not None:
        try:
            bus_options = sorted(df_session['bus'].dropna().unique())
            selected_buses = st.multiselect(
                "Select bus(es) (or 'All busses')",
                options=[0] + bus_options,
                default=[0],
                format_func=lambda x: f"Bus {int(x)}" if x != 0 else "All busses"
            )

            # quick feasibility
            timetable = st.session_state.get('timetable_df')
            any_fail = False
            cap = CAP_DEFAULT
            soc0 = SOC0_DEFAULT
            buses_to_check = sorted(df_session['bus'].dropna().unique()) if 0 in selected_buses else [b for b in selected_buses if b in df_session['bus'].values]

            for bus_id in buses_to_check:
                g = df_session[df_session['bus'] == bus_id].copy()
                batt = get_battery_diagnostics(g, cap, soc0)
                if not batt['ok']:
                    st.error(batt['message'])
                    any_fail = True
                    break
                if timetable is not None:
                    tt = get_timetable_diagnostics(g, timetable, BUFFER_MIN)
                    if not tt['ok']:
                        st.error(tt['message'])
                        any_fail = True
                        break
            if not any_fail:
                st.success("Selected busplan(s) appear feasible.")

            fig = plot_gantt_interactive(df_session, selected_buses)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not display the Gantt chart: {e}")
    else:
        st.info("Upload an Excel file in the sidebar to see the Gantt Chart.")

# ===================== VISUALS =====================
with tab_visuals:
    st.subheader("📈 Visualisation")
    if uploaded_file:
        dfv = load_data(uploaded_file)
        group_by = "bus"
        cap_kwh = CAP_DEFAULT
        start_soc = SOC0_DEFAULT

        opts = sorted(dfv[group_by].dropna().astype(str).unique().tolist())
        pick = st.multiselect("Select Bus(es)", options=opts, default=opts[:min(5, len(opts))], key="pick_groups")
        if pick:
            dfv = dfv[dfv[group_by].astype(str).isin(pick)]

        ts = "start time"
        d = dfv.copy()
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

        import plotly.graph_objects as go
        fig_soc = px.line(soc_df, x=ts, y="soc_%", color=group_by, title=f"SoH / SoC progression per Bus", labels={"soc_%": "SOC (%)", ts: "Time"})
        fig_soc.update_yaxes(range=[0, 100])
        fig_soc.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        fig_soc.add_hline(y=10, line_dash="dot", annotation_text="10% threshold", annotation_position="bottom right")
        st.plotly_chart(fig_soc, use_container_width=True)

        st.caption("Positive 'energy consumption' = usage (SOC down), negative = charging (SOC up).")
    else:
        st.info("Upload an Excel file to see visuals.")

# ===================== ANALYSIS =====================
with tab_analysis:
    st.subheader("🔍 Analysis")
    if uploaded_file:
        dfa = load_data(uploaded_file)
        dfa['line_str'] = dfa['line'].astype(str).str.replace('.0', '', regex=False)

        total_duration_per_bus = dfa.groupby('bus', as_index=False)['duration_minutes'].sum().rename(columns={'duration_minutes': 'total_duration_minutes'})
        if 'energy consumption' in dfa.columns:
            per_bus = dfa.groupby('bus', as_index=False).agg(consumption_kWh=('energy consumption', lambda s: s.clip(lower=0).sum()))
            bus_summary = pd.merge(total_duration_per_bus, per_bus, on='bus', how='outer')
            st.write("### Total duration + Energy per bus")
            st.dataframe(bus_summary.sort_values('bus'), use_container_width=True)

            st.write("### Summary per bus per activity")
            summary = (dfa.groupby(['bus', 'activity'], dropna=False)
                        .agg(num_trips=('activity', 'count'),
                             total_duration=('duration_minutes', 'sum'))
                        .reset_index())
            summary = summary.round({'total_duration': 2})
            pivot_summary = summary.pivot(index='bus', columns='activity', values=['num_trips','total_duration'])
            pivot_summary.columns = [f"{agg}_{act}" for agg, act in pivot_summary.columns]
            st.dataframe(pivot_summary.sort_values('bus'), use_container_width=True)
    else:
        st.info("Upload an Excel file for analysis.")

# ===================== ERRORS =====================
with tab_errors:
    st.subheader("🚨 Errors")
    try:
        if uploaded_file is not None:
            busplan = load_data(uploaded_file)
        elif os.path.exists('Bus Planning.xlsx'):
            busplan = load_data('Bus Planning.xlsx')
        else:
            busplan = None
    except Exception:
        busplan = None

    try:
        if uploaded_tt is not None:
            timetable = uploaded_tt
        elif os.path.exists('Timetable.xlsx'):
            timetable = load_timetable('Timetable.xlsx')
        else:
            timetable = None
    except Exception:
        timetable = None

    if busplan is None:
        st.info("No bus plan loaded.")
    else:
        # diagnostics per bus
        diagnostics = {}
        for bus, group in busplan.groupby('bus'):
            group = group.sort_values('start time').reset_index(drop=True)
            batt_diag = get_battery_diagnostics(group, CAP_DEFAULT, SOC0_DEFAULT)
            tt_diag = get_timetable_diagnostics(group, timetable, BUFFER_MIN) if timetable is not None else {'ok': True, 'message': 'No timetable provided', 'violations': []}
            diagnostics[bus] = {'battery': batt_diag, 'timetable': tt_diag, 'group': group}

        st.write("### Detailed diagnostics per bus")
        for bus in sorted(diagnostics.keys()):
            d = diagnostics[bus]
            batt = d['battery']
            tt = d['timetable']
            with st.expander(f"Bus {bus} — {batt['message'] if batt else 'No battery data'} | {tt['message']}", expanded=False):
                st.write("**Battery simulation**")
                st.dataframe(batt['trace'].sort_values('start time').reset_index(drop=True), use_container_width=True)
                st.write("---")
                st.write("**Timetable checks**")
                if tt['violations']:
                    vt = pd.DataFrame(tt['violations'])
                    st.dataframe(vt, use_container_width=True)
                else:
                    st.write("No timetable violations.")

# ===================== KPI (PLAN vs PLAN) =====================
with tab_kpi:
    st.subheader("📊 KPI Dashboard — Old vs New busplan")

    uploaded_tt_file = st.file_uploader("Upload Timetable (Excel) — used for on-time checks", type=["xlsx"], key="tt_upload_global")
    timetable = None
    if uploaded_tt_file:
        timetable = load_timetable(uploaded_tt_file)
        st.success("Timetable uploaded successfully!")

    col_old, col_new = st.columns(2, gap="large")

    def render_plan_card(plan_title, df, timetable):
        st.markdown(f"#### {plan_title}")
        if df is None or df.empty:
            st.info("No data.")
            return
        stats = compute_plan_kpi(df, timetable, CAP_DEFAULT, SOC0_DEFAULT)
        colA, colB = st.columns([1,1])
        with colA:
            pie = px.pie(
                stats['mix'],
                names='activity',
                values='minutes',
                title="Time distribution",
                hole=0.35,
                color='activity',
                color_discrete_map=ACTIVITY_COLORS
            )
            pie.update_traces(textposition='inside', texttemplate='%{label}<br>%{percent:.0%}')
            st.plotly_chart(pie, use_container_width=True)
        with colB:
            st.metric("KPI", f"{stats['kpi']}")
            st.write(f"**Idle ratio:** {stats['idle_ratio']:.2%}")
            st.write(f"**Battery fails (buses):** {stats['battery_buses_failed']} / {stats['battery_buses_total']}")
            st.write(f"**Schedule violations (transitions):** {stats['schedule_violations']} / {stats['schedule_transitions']}")

    with col_old:
        st.markdown("#### Old busplan (from sidebar)")
        if uploaded_file:
            try:
                old_df = load_data(uploaded_file)
                render_plan_card("OLD — KPI", old_df, timetable)
            except Exception as e:
                st.error(f"Old busplan processing failed: {e}")
        else:
            st.info("Upload the **old** busplan in the sidebar to see KPIs.")

    with col_new:
        st.markdown("#### New busplan (upload here)")
        new_upload = st.file_uploader("Upload NEW busplan (Excel)", type=["xlsx"], key="new_busplan_upload")
        if new_upload:
            try:
                new_df = load_data(new_upload)
                render_plan_card("NEW — KPI", new_df, timetable)
            except Exception as e:
                st.error(f"New busplan processing failed: {e}")
        else:
            st.info("Upload the **new** busplan to compare.")
