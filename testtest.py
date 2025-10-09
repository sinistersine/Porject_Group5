from ftplib import all_errors
import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_excel("Bus planning.xlsx")

# Kolommen naar datetime
df['start time'] = pd.to_datetime(df['start time'], format="%H:%M:%S")
df['end time']   = pd.to_datetime(df['end time'],   format="%H:%M:%S")

# --- fix 'line' per bus: leeg/rommel -> meest voorkomende lijn van die bus ---
allowed = {400, 401}
df["line"] = pd.to_numeric(df["line"], errors="coerce")  # rommel -> NaN

# modus per bus (meest voorkomende geldige lijn)
bus_mode = (
    df[df["line"].isin(allowed)]
      .groupby("bus")["line"]
      .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA)
)

# fallback: globale modus als een bus geen enkele geldige lijn heeft
global_mode = (
    df.loc[df["line"].isin(allowed), "line"].mode().iloc[0]
    if df["line"].isin(allowed).any() else 400
)

df["bus_mode_line"] = df["bus"].map(bus_mode).fillna(global_mode)

# zet lege/ongeldige lijnen naar de 'goede' lijn van die bus
bad = df["line"].isna() | ~df["line"].isin(allowed)
df.loc[bad, "line"] = df.loc[bad, "bus_mode_line"]

# (optioneel) wil je ALLE afwijkingen forceren naar de bus-modus? uncomment:
# df["line"] = df["bus_mode_line"]

df.drop(columns="bus_mode_line", inplace=True)
df["line"] = df["line"].astype("Int64")


# ---- tijd-analyses ----
df['duration_minutes'] = (df['end time'] - df['start time']).dt.total_seconds() / 60
avg_duration = df.groupby('activity', as_index=False)['duration_minutes'].mean()

total_duration_per_bus = (df.groupby('bus', as_index=False)['duration_minutes']
                            .sum()
                            .rename(columns={'duration_minutes': 'Total time in minutes'}))
# ---- energie-analyses (alles BINNEN de if uploaded_file) ----
df['energy consumption'] = pd.to_numeric(df['energy consumption'], errors='coerce').fillna(0)

per_bus = df.groupby('bus', as_index=False).agg(
verbruik_kWh=('energy consumption', lambda s: s.clip(lower=0).sum()),
geladen_kWh =('energy consumption', lambda s: (-s.clip(upper=0)).sum()),
netto_kWh   =('energy consumption', 'sum'))

BATTERY_KWH = 300.0  # pas aan naar echte cap
per_bus['eind_SOC_%'] = (100 - (per_bus['netto_kWh'] / BATTERY_KWH) * 100).clip(0, 100)

df