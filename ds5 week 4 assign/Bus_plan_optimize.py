import pandas as pd
from datetime import datetime, timedelta

# ---- 1. Excel bestanden inlezen ----
busplan = pd.read_excel("Bus planning.xlsx")
timetable = pd.read_excel("Timetable.xlsx")
distance_matrix = pd.read_excel("DistanceMatrix.xlsx", index_col=0)  # rijen=from, kolommen=to

# ---- 2. Parameters ----
BATTERY_CAPACITY = 300  # kWh
MIN_SOC = 0.15          # 15%
MAX_SOC = 0.9           # 90%
CHARGE_RATE_FAST = 450  # kW
CHARGE_RATE_SLOW = 60   # kW
CHARGE_MIN_TIME = 15    # minuten

# ---- 3. Helper functies ----

# tijd verschil in minuten
def minutes_diff(start, end):
    return (end - start).total_seconds() / 60

# laadduur berekenen
def compute_charge_time(current_soc, target_soc):
    soc_needed = target_soc - current_soc
    if soc_needed <= 0:
        return 0
    charge_needed = soc_needed * BATTERY_CAPACITY
    if target_soc <= MAX_SOC:
        time_hours = charge_needed / CHARGE_RATE_FAST
    else:
        fast_charge = (MAX_SOC - current_soc) * BATTERY_CAPACITY / CHARGE_RATE_FAST
        slow_charge = (target_soc - MAX_SOC) * BATTERY_CAPACITY / CHARGE_RATE_SLOW
        time_hours = fast_charge + slow_charge
    return max(time_hours * 60, CHARGE_MIN_TIME)

# bereken energieverbruik op basis van distance matrix
def compute_energy(start_loc, end_loc):
    try:
        return distance_matrix.loc[start_loc, end_loc]  # kWh
    except KeyError:
        return 0  # fallback

# ---- 4. Tijd kolommen omzetten naar datetime ----
busplan['start time'] = pd.to_datetime(busplan['start time'])
busplan['end time'] = pd.to_datetime(busplan['end time'])
timetable['departure_time'] = pd.to_datetime(timetable['departure_time'])

# ---- 5. Optimalisatie functie ----
def optimize_busplan(busplan, timetable):
    buses = {}  # bus_id : lijst van trips
    bus_id = 1
    timetable = timetable.sort_values('departure_time')

    for idx, trip in timetable.iterrows():
        assigned = False
        for b_id, trips in buses.items():
            last_trip = trips[-1]
            # material trip naar start locatie indien nodig
            travel_time = timedelta(minutes=minutes_diff(last_trip['end time'], last_trip['end time'] + timedelta(minutes=1)))  # placeholder
            available_time = last_trip['end time'] + travel_time
            if available_time <= trip['departure_time']:
                # batterij check
                soc = last_trip.get('soc', 1.0)
                energy_use = compute_energy(last_trip['end location'], trip['start'])
                soc -= energy_use / BATTERY_CAPACITY
                if soc < MIN_SOC:
                    charge_time = compute_charge_time(soc, MAX_SOC)
                    available_time += timedelta(minutes=charge_time)
                    soc = MAX_SOC
                trips.append({
                    'start location': trip['start'],
                    'end location': trip['end'],
                    'start time': trip['departure_time'],
                    'end time': trip['departure_time'] + timedelta(minutes=30),  # placeholder, kan van distance matrix
                    'activity': 'service trip',
                    'line': trip['line'],
                    'soc': soc
                })
                assigned = True
                break
        if not assigned:
            # nieuwe bus
            buses[bus_id] = [{
                'start location': trip['start'],
                'end location': trip['end'],
                'start time': trip['departure_time'],
                'end time': trip['departure_time'] + timedelta(minutes=30),
                'activity': 'service trip',
                'line': trip['line'],
                'soc': 1.0
            }]
            bus_id += 1

    # naar dataframe
    optimized_plan = []
    for b_id, trips in buses.items():
        for t in trips:
            t['bus'] = b_id
            optimized_plan.append(t)

    return pd.DataFrame(optimized_plan)

optimized_busplan = optimize_busplan(busplan, timetable)
optimized_busplan.to_excel("optimized_busplan.xlsx", index=False)
print("Optimalisatie klaar! Bestand opgeslagen als optimized_busplan.xlsx")
