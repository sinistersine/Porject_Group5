# streamlit run dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np

# === material-trip energy rule ===
BASE_MIN, BASE_KWH = 20, 10.32  # 20 min → 10.32 kWh
TOL = 0.05  # match-tolerantie

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
    end = pd.to_datetime(df['end time'], format="%H:%M:%S", errors='coerce')

    # nacht-rollover: alleen waar beide niet NaT zijn én end < start
    roll_mask = (end < start) & start.notna() & end.notna()
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
    df['end time'] = end_fix

    return df


# ========= Page =========
st.set_page_config(page_title="Bus Planning dashboard", layout="wide")
st.title("🚌 Bus Planning dashboard")

# ========= Sidebar Uploaders =========
uploaded_file = st.sidebar.file_uploader("1) Upload OLD busplan (Excel)", type=["xlsx"], key="busplan")
uploaded_tt_sidebar = st.sidebar.file_uploader("2) (Optioneel) Upload Timetable (Excel)", type=["xlsx"], key="timetable_sidebar")
if uploaded_tt_sidebar is not None:
    try:
        st.session_state['uploaded_tt'] = pd.read_excel(uploaded_tt_sidebar, index_col=0)
        st.sidebar.success("Timetable geladen.")
    except Exception as e:
        st.sidebar.error(f"Kon Timetable niet laden: {e}")

# ========= Tabs =========
tab_gantt, tab_visuals, tab_analysis, tab_errors, tab_kpi = st.tabs(
    ["📊 Gantt-chart", "📈 Visualisations", "🔍 Analysis", "🚨 Errors", "📊 KPI Dashboard"]
)


# ========= Helpers =========
def plot_gantt_interactive(df, selected_buses=None):
    df_plot = df.copy()
    if selected_buses and 0 not in selected_buses:
        df_plot = df_plot[df_plot['bus'].isin(selected_buses)]

    # Label voor service trips → verticale layout via <br> (optioneel: lijnen tonen)
    def build_label(row):
        if str(row.get('activity', '')).lower() == 'service trip' and pd.notna(row.get('line')):
            try:
                s = str(int(row['line']))
            except Exception:
                s = str(row['line'])
            return '<br>'.join(list(s))
        return ''

    df_plot['label'] = df_plot.apply(build_label, axis=1)
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

    fig.update_traces(
        textposition="inside",
        textfont=dict(size=11),
        cliponaxis=False,
        marker=dict(line=dict(color='black', width=1))
    )

    base_date = df_plot['start time'].dropna().min()
    if pd.isna(base_date):
        return fig

    base_date = base_date.date()
    start_range = pd.Timestamp(f"{base_date} 05:00:00")
    end_range = pd.Timestamp(f"{base_date} 01:00:00") + pd.Timedelta(days=1)

    fig.update_xaxes(range=[start_range, end_range], title_text="Time", showgrid=True, gridcolor="LightGray", tickformat="%H:%M")
    fig.update_yaxes(title="Bus", autorange="reversed", showgrid=True, gridcolor="LightGray")
    fig.update_layout(height=400 + 30*len(df_plot['bus'].dropna().unique()), margin=dict(l=20, r=20, t=40, b=20), dragmode='zoom')
    return fig


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
    """Detailed battery simulation per bus."""
    battery = float(cap_kwh) * (float(start_soc_percent) / 100.0)
    rows = []
    g = bus_df.sort_values('start time').reset_index(drop=True).copy()
    for _, r in g.iterrows():
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
        msg = f"Battery dies during activity at {first['start time']} (bus runs out of kWh)."
        return {'ok': False, 'message': msg, 'trace': trace}
    return {'ok': True, 'message': 'Battery OK', 'trace': trace}


def get_timetable_diagnostics(bus_df, timetable_df):
    """Detailed timetable check diagnostics."""
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
            violations.append({
                'idx': i,
                'reason': 'insufficient_travel_time',
                'prev_end': prev_end,
                'curr_start': curr_start,
                'required_min': required,
                'expected_min': expected,
                'prev_line': prev.get('activity'),
                'curr_line': curr.get('activity')
            })

    if violations:
        return {'ok': False, 'message': f"{len(violations)} timetable violation(s)", 'violations': violations}
    return {'ok': True, 'message': 'Timing OK', 'violations': []}


# ======== KPI Helpers (PLAN-LEVEL) ========
def summarize_plan(df_plan: pd.DataFrame, timetable: pd.DataFrame | None):
    """
    Plan-level metrics:
      total_minutes, idle_minutes, idle_ratio,
      pie_pct (service/material/idle),
      n_buses, n_battery_ok, n_battery_fail,
      n_service_late, kpi_score
    """
    if df_plan is None or df_plan.empty:
        return {
            'total_minutes': 0, 'idle_minutes': 0, 'idle_ratio': 0.0,
            'pie_pct': {'service trip': 0, 'material trip': 0, 'idle time': 0},
            'n_buses': 0, 'n_battery_ok': 0, 'n_battery_fail': 0,
            'n_service_late': 0, 'kpi_score': 0
        }

    d = df_plan.copy()
    d['activity_norm'] = d['activity'].astype(str).str.lower()

    # Totals
    total_minutes = d['duration_minutes'].sum()
    idle_minutes = d.loc[d['activity_norm'] == 'idle', 'duration_minutes'].sum()
    idle_ratio = (idle_minutes / total_minutes) if total_minutes > 0 else 0.0

    # Pie (exclude charging)
    pie_subset = d[~d['activity_norm'].str.contains('charging', na=False)].copy()
    pie_counts = {
        'service trip': pie_subset.loc[pie_subset['activity_norm'] == 'service trip', 'duration_minutes'].sum(),
        'material trip': pie_subset.loc[pie_subset['activity_norm'] == 'material trip', 'duration_minutes'].sum(),
        'idle time':    pie_subset.loc[pie_subset['activity_norm'] == 'idle', 'duration_minutes'].sum(),
    }
    pie_total = sum(pie_counts.values()) or 1.0
    pie_pct = {k: (v / pie_total) * 100 for k, v in pie_counts.items()}

    # Buses
    n_buses = d['bus'].dropna().nunique()

    # Battery survival per bus
    n_ok, n_fail = 0, 0
    for _, g in d.groupby('bus'):
        diag = get_battery_diagnostics(g.sort_values('start time'), cap_kwh=300, start_soc_percent=100)
        if diag['ok']:
            n_ok += 1
        else:
            n_fail += 1

    # Timetable lateness
    n_late = 0
    if timetable is not None and n_buses > 0:
        for _, g in d.groupby('bus'):
            tt = get_timetable_diagnostics(g.sort_values('start time'), timetable)
            if not tt['ok']:
                n_late += sum(1 for v in tt['violations'] if v.get('reason') == 'insufficient_travel_time')

    # KPI score (weights tweakable)
    battery_fail_rate = (n_fail / (n_ok + n_fail)) if (n_ok + n_fail) > 0 else 0.0
    svc_hops = d.loc[d['activity_norm'] == 'service trip'].shape[0]
    lateness_rate = (n_late / svc_hops) if svc_hops > 0 else 0.0

    kpi = 100 - (idle_ratio * 50 + battery_fail_rate * 30 + lateness_rate * 20)
    kpi = float(np.clip(kpi, 0, 100))

    return {
        'total_minutes': total_minutes,
        'idle_minutes': idle_minutes,
        'idle_ratio': idle_ratio,
        'pie_pct': pie_pct,
        'n_buses': int(n_buses),
        'n_battery_ok': int(n_ok),
        'n_battery_fail': int(n_fail),
        'n_service_late': int(n_late),
        'kpi_score': round(kpi, 1),
    }


def pie_fig_from_pct(pie_pct: dict, title: str = ""):
    labels = ['service trip', 'material trip', 'idle time']
    values = [
        pie_pct.get('service trip', 0),
        pie_pct.get('material trip', 0),
        pie_pct.get('idle time', 0)
    ]
    fig = px.pie(
        names=labels,
        values=values,
        title=title,
        hole=0.35
    )
    fig.update_traces(textposition='inside', texttemplate='%{percent:.0%}')
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=260,
                      legend=dict(orientation='h', yanchor='bottom', y=-0.2))
    return fig


# =================== TAB 1: Gantt Chart ===================
with tab_gantt:
    st.subheader("📊 Gantt Chart")

    df = None
    selected_buses = []

    if uploaded_file:
        try:
            df = load_data(uploaded_file)

            # Bus multiselect
            bus_options = sorted(df['bus'].dropna().unique())
            selected_buses = st.multiselect(
                "Selecteer bus(s) (of 'All busses')",
                options=[0] + bus_options,
                default=[0],
                format_func=lambda x: f"Bus {int(x)}" if x != 0 else "All busses",
                key="gantt_buses"
            )

            # Feasibility checks per bus
            timetable = st.session_state.get('uploaded_tt')
            if timetable is None and os.path.exists('Timetable.xlsx'):
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


# =================== TAB 2: Visualisations ===================
with tab_visuals:
    st.subheader("📈 Visualisation")

    if uploaded_file:
        df = load_data(uploaded_file)

        # Fixed setup: group by bus
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

        # Build SOC curve
        ts = "start time"
        d = df.copy()
        d["energy consumption"] = pd.to_numeric(
            d["energy consumption"], errors="coerce"
        ).fillna(0.0)

        def build_soc(g):
            g = g.sort_values(ts).copy()
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

        # Check: battery below 10%
        under10 = soc_df[soc_df["soc_%"] < 10]
        if not under10.empty:
            hits = (
                under10.sort_values(ts)
                .groupby(group_by, as_index=False)
                .first()[[group_by, ts, "soc_%"]]
            )
            st.error(
                "⚠️ Battery below 10% — these buses are unavailable: "
                + ", ".join(hits[group_by].astype(str).tolist())
            )
            with st.expander("First moment below 10% per bus"):
                st.dataframe(hits, use_container_width=True)

        fig_soc = px.line(
            soc_df, x=ts, y="soc_%", color=group_by,
            title=f"SoH / SoC progression per {group_by_display}",
            labels={"soc_%": "SOC (%)", ts: "Time"},
        )
        fig_soc.update_yaxes(range=[0, 100])
        fig_soc.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        fig_soc.add_hline(y=10, line_dash="dot", annotation_text="10% threshold", annotation_position="bottom right")

        if not under10.empty:
            fig_soc.add_trace(
                go.Scatter(
                    x=under10[ts], y=under10["soc_%"],
                    mode="markers", marker=dict(size=8, color="red"),
                    name="<10% SoC"
                )
            )

        st.plotly_chart(fig_soc, use_container_width=True)
        st.caption("Positive 'energy consumption' = usage (SOC decreases), negative = charging (SOC increases).")

    else:
        st.info("Upload an Excel file in the sidebar to see the SOC graph.")


# =================== TAB 3: Analysis ===================
with tab_analysis:
    st.subheader("🔍 Analysis")

    if uploaded_file:
        df = load_data(uploaded_file)

        # lijn als string (voor latere analyses)
        if 'line' in df.columns:
            df['line_str'] = df['line'].astype(str).str.replace('.0', '', regex=False)

        # Total duration per bus
        total_duration_per_bus = (
            df.groupby('bus', as_index=False)['duration_minutes']
            .sum()
            .rename(columns={'duration_minutes': 'total_duration_minutes'})
        )

        # Energy per bus
        per_bus = df.groupby('bus', as_index=False).agg(
            consumption_kWh=('energy consumption', lambda s: s.clip(lower=0).sum())
        )

        # Merge
        bus_summary = pd.merge(total_duration_per_bus, per_bus, on='bus', how='outer')
        st.write("### Total duration + Energy per bus")
        st.dataframe(bus_summary.sort_values('bus'), use_container_width=True)

        # Summary per bus per activity
        st.write("### Summary per bus per activity")
        summary = (
            df.groupby(['bus', 'activity'], dropna=False)
            .agg(num_trips=('activity', 'count'), total_duration=('duration_minutes', 'sum'))
            .reset_index()
        ).round({'total_duration': 2})

        pivot_summary = summary.pivot(index='bus', columns='activity', values=['num_trips','total_duration'])
        pivot_summary.columns = [f"{agg}_{act}" for agg, act in pivot_summary.columns]
        st.dataframe(pivot_summary.sort_values('bus'), use_container_width=True)

    else:
        st.info("Upload an Excel file in the sidebar to see the analysis.")


# =================== TAB 4: Errors ===================
with tab_errors:
    st.subheader("🚨 Errors")
    st.write("Here is a list of errors detected in the planning (feasibility checks).")

    only_selected = st.checkbox("Toon alleen bussen geselecteerd in Gantt", value=False, key="errors_only_selected")

    # Allow timetable upload here too (if not given in sidebar)
    uploaded_tt_errors = st.file_uploader("Upload Timetable (Excel) for error checks", type=["xlsx"], key="tt_upload_errors")
    timetable = st.session_state.get('uploaded_tt')
    if uploaded_tt_errors is not None:
        try:
            timetable = pd.read_excel(uploaded_tt_errors, index_col=0)
            st.success("Timetable uploaded successfully!")
        except Exception as e:
            st.error(f"Timetable kon niet worden geladen: {e}")

    # Load busplan
    try:
        if uploaded_file is not None:
            busplan = load_data(uploaded_file)
        elif os.path.exists('Bus Planning.xlsx'):
            busplan = load_data('Bus Planning.xlsx')
        else:
            busplan = None
    except Exception:
        busplan = None

    if busplan is None:
        st.info("No bus plan loaded. Upload the bus plan in the sidebar or place 'Bus Planning.xlsx' in the app folder.")
    else:
        # ensure time columns parsed (load_data already handles this)
        for col in ['start time', 'end time']:
            if col in busplan.columns:
                busplan[col] = pd.to_datetime(busplan[col], errors='coerce')

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

        buses_to_show = sorted(diagnostics.keys())
        if only_selected:
            sel = st.session_state.get('gantt_buses', None)
            if sel and 0 not in sel:
                buses_to_show = [b for b in buses_to_show if b in sel]

        st.write("### Detailed diagnostics per bus")
        if not buses_to_show:
            st.info("Geen bussen om te tonen met de huidige filterinstellingen.")
        for bus in buses_to_show:
            d = diagnostics[bus]
            batt = d['battery']
            tt = d['timetable']
            header = f"Bus {bus} — {batt['message'] if batt else 'No battery data'}"
            with st.expander(header, expanded=False):
                st.write("**Battery simulation**")
                st.write(batt['message'])
                st.dataframe(batt['trace'].sort_values('start time').reset_index(drop=True), use_container_width=True)
                if tt is not None:
                    st.write("---")
                    st.write("**Timetable checks**")
                    st.write(tt['message'])
                    if tt['violations']:
                        vt = pd.DataFrame(tt['violations'])
                        st.dataframe(vt, use_container_width=True)
                    else:
                        st.write("No timetable violations detected.")


# =================== TAB 5: KPI Dashboard (Old vs New, Plan-level) ===================
# Tab 5: KPI Dashboard (plan vs plan)
with tab_kpi:
    st.subheader("📊 KPI Dashboard — Old vs New Busplan (Plan-level)")

    # ---------- Uploaders ----------
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Old busplan**")
        uploaded_old_in_tab = st.file_uploader("(Optional) Upload old busplan (Excel)", type=["xlsx"], key="busplan_old_kpi")
    with col_right:
        st.markdown("**New busplan**")
        uploaded_new = st.file_uploader("Upload NEW busplan (Excel)", type=["xlsx"], key="busplan_new")

    # Timetable (shared)
    st.markdown("---")
    uploaded_tt_file = st.file_uploader("Upload Timetable (Excel) for lateness checks", type=["xlsx"], key="tt_upload_plan")
    timetable = None
    if uploaded_tt_file:
        try:
            timetable = pd.read_excel(uploaded_tt_file, index_col=0)
            st.success("Timetable uploaded successfully!")
        except Exception as e:
            st.error(f"Timetable kon niet worden geladen: {e}")

    # ---------- Load plans ----------
    df_old = None
    # Priority: uploader in this tab > sidebar uploader
    if uploaded_old_in_tab is not None:
        try:
            df_old = load_data(uploaded_old_in_tab)
        except Exception as e:
            st.error(f"Kon OLD busplan niet laden: {e}")
    elif uploaded_file is not None:
        # fallback to sidebar upload
        try:
            df_old = load_data(uploaded_file)
        except Exception as e:
            st.error(f"Kon OLD busplan (sidebar) niet laden: {e}")

    df_new = None
    if uploaded_new is not None:
        try:
            df_new = load_data(uploaded_new)
        except Exception as e:
            st.error(f"Kon NEW busplan niet laden: {e}")

    # ---------- Guard rails ----------
    if df_old is None and df_new is None:
        st.info("Upload minstens één busplan om KPI's te zien (links = oud, rechts = nieuw).")
        st.stop()

    # ---------- Compute summaries ----------
    colA, colB = st.columns(2)
    if df_old is not None:
        old_sum = summarize_plan(df_old, timetable)
        with colA:
            st.markdown("### 🧓 Old Plan")
            st.plotly_chart(pie_fig_from_pct(old_sum['pie_pct'], "Activity Mix"), use_container_width=True)

            # KPI big number
            st.markdown(f"<div style='font-size:44px; font-weight:700; text-align:center;'>KPI: {old_sum['kpi_score']}</div>", unsafe_allow_html=True)

            # Stats under KPI
            c1, c2, c3 = st.columns(3)
            c1.metric("Service trips not on time", f"{old_sum['n_service_late']}")
            c2.metric("Buses in plan", f"{old_sum['n_buses']}")
            c3.metric("Buses that make charge", f"{old_sum['n_battery_ok']}")

            with st.expander("More details (idle ratio, battery fails)"):
                st.write(f"Idle ratio: {old_sum['idle_ratio']:.1%}")
                st.write(f"Battery fails: {old_sum['n_battery_fail']}")

    if df_new is not None:
        new_sum = summarize_plan(df_new, timetable)
        with colB:
            st.markdown("### ✨ New Plan")
            st.plotly_chart(pie_fig_from_pct(new_sum['pie_pct'], "Activity Mix"), use_container_width=True)

            # KPI big number
            st.markdown(f"<div style='font-size:44px; font-weight:700; text-align:center;'>KPI: {new_sum['kpi_score']}</div>", unsafe_allow_html=True)

            # Stats under KPI
            c1, c2, c3 = st.columns(3)
            c1.metric("Service trips not on time", f"{new_sum['n_service_late']}")
            c2.metric("Buses in plan", f"{new_sum['n_buses']}")
            c3.metric("Buses that make charge", f"{new_sum['n_battery_ok']}")

            with st.expander("More details (idle ratio, battery fails)"):
                st.write(f"Idle ratio: {new_sum['idle_ratio']:.1%}")
                st.write(f"Battery fails: {new_sum['n_battery_fail']}")

    # ---------- Side-by-side KPI bars (just for the vibes) ----------
    if (df_old is not None) and (df_new is not None):
        comp_df = pd.DataFrame({
            'plan': ['Old', 'New'],
            'kpi_score': [old_sum['kpi_score'], new_sum['kpi_score']]
        })
        fig_comp = px.bar(
            comp_df, x='plan', y='kpi_score',
            title="Plan KPI comparison",
            color='plan', text='kpi_score',
            range_y=[0, 100]
        )
        fig_comp.update_traces(texttemplate='%{text}', textposition='outside')
        fig_comp.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=350)
        st.plotly_chart(fig_comp, use_container_width=True)

    # tiny note about weights so future-you isn't guessing
    st.caption("KPI = 100 − (idle_ratio×50 + battery_fail_rate×30 + lateness_rate×20). Pas gerust de gewichten aan.")
