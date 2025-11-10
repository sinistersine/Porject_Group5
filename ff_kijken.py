import pandas as pd
from datetime import datetime, timedelta

# === SETTINGS ===
BATTERY_CAPACITY = 300  # kWh
MAX_SOC = 0.9 * BATTERY_CAPACITY  # 90%
MIN_SOC = 0.1 * BATTERY_CAPACITY  # 10%
CHARGE_POWER = 450  # kW per hour
CHARGE_DURATION_MIN = 15  # minimum charging duration (minutes)
CONSUMPTION_PER_KM = 1.5  # kWh/km (average consumption)
BUFFER_MINUTES = 5  # minimal time between trips (minutes)

# === READ INPUT FILES ===
busplan = pd.read_excel("Bus planning.xlsx")
timetable = pd.read_excel("Timetable.xlsx")
distances = pd.read_excel("DistanceMatrix.xlsx")

# convert departure times to datetime.time objects
def to_time(t):
    if isinstance(t, str):
        return datetime.strptime(t.strip(), "%H:%M").time()
    return t

timetable["departure_time"] = timetable["departure_time"].apply(to_time)

# === PREPARE DISTANCE LOOKUP ===
distance_lookup = distances.set_index(["start", "end"])["distance_m"].to_dict()

def get_travel_time(start, end, line=None):
    """Return minimum travel time (minutes) between two stops."""
    try:
        if line:
            val = distances.query("start==@start and end==@end and line==@line")["min_travel_time"].values
            if len(val) == 0:
                val = distances.query("start==@start and end==@end")["min_travel_time"].values
        else:
            val = distances.query("start==@start and end==@end")["min_travel_time"].values
        return int(val[0])
    except Exception:
        print(f"⚠️ Warning: Missing travel time between {start} and {end}, assuming 10 min.")
        return 10

def energy_use_kwh(start, end):
    """Energy use in kWh for this connection."""
    d_m = distance_lookup.get((start, end), 0)
    return round((d_m / 1000) * CONSUMPTION_PER_KM, 2)

# === MAIN SIMULATION ===
buses = []
plan_records = []

for _, row in timetable.iterrows():
    start, dep_time, end, line = row["start"], row["departure_time"], row["end"], row["line"]

    dep_dt = datetime.combine(datetime.today(), dep_time)
    travel_time = timedelta(minutes=int(get_travel_time(start, end, line)))
    arr_dt = dep_dt + travel_time
    energy_needed = energy_use_kwh(start, end)

    assigned_bus = None

    for bus in buses:
        last_end_time = bus["available_from"]
        soc = bus["soc"]
        location = bus["location"]

        # bus can make it if it's free and has enough energy
        if last_end_time + timedelta(minutes=BUFFER_MINUTES) <= dep_dt:
            # if not at same location -> material trip
            if location != start:
                transfer_min = get_travel_time(location, start)
                transfer_time = timedelta(minutes=int(transfer_min))
                transfer_energy = energy_use_kwh(location, start)

                dep_transfer = last_end_time
                arr_transfer = dep_transfer + transfer_time
                bus["soc"] -= transfer_energy

                plan_records.append({
                    "start location": location,
                    "end location": start,
                    "start time": dep_transfer.time(),
                    "end time": arr_transfer.time(),
                    "activity": "material trip",
                    "line": "",
                    "energy consumption": round(transfer_energy, 2),
                    "bus": bus["id"]
                })
                bus["available_from"] = arr_transfer
                bus["location"] = start

            # check SOC -> must charge if too low
            if bus["soc"] - energy_needed < MIN_SOC:
                if bus["location"] != "ehvgar":
                    transfer_min = get_travel_time(bus["location"], "ehvgar")
                    transfer_time = timedelta(minutes=int(transfer_min))
                    transfer_energy = energy_use_kwh(bus["location"], "ehvgar")

                    dep_transfer = bus["available_from"]
                    arr_transfer = dep_transfer + transfer_time
                    bus["soc"] -= transfer_energy

                    plan_records.append({
                        "start location": bus["location"],
                        "end location": "ehvgar",
                        "start time": dep_transfer.time(),
                        "end time": arr_transfer.time(),
                        "activity": "material trip",
                        "line": "",
                        "energy consumption": round(transfer_energy, 2),
                        "bus": bus["id"]
                    })
                    bus["location"] = "ehvgar"
                    bus["available_from"] = arr_transfer

                # charging activity
                charge_start = bus["available_from"]
                charge_end = charge_start + timedelta(minutes=int(CHARGE_DURATION_MIN))
                energy_added = CHARGE_POWER * (CHARGE_DURATION_MIN / 60)
                bus["soc"] = min(MAX_SOC, bus["soc"] + energy_added)

                plan_records.append({
                    "start location": "ehvgar",
                    "end location": "ehvgar",
                    "start time": charge_start.time(),
                    "end time": charge_end.time(),
                    "activity": "charging",
                    "line": "",
                    "energy consumption": -round(energy_added, 2),
                    "bus": bus["id"]
                })
                bus["available_from"] = charge_end
                bus["location"] = "ehvgar"

            # if bus ready again
            if bus["available_from"] + timedelta(minutes=BUFFER_MINUTES) <= dep_dt and bus["soc"] - energy_needed >= MIN_SOC:
                assigned_bus = bus
                break

    if assigned_bus is None:
        # create new bus
        bus = {
            "id": len(buses) + 1,
            "soc": MAX_SOC,
            "available_from": arr_dt,
            "location": end
        }
        buses.append(bus)
        assigned_bus = bus
    else:
        assigned_bus["soc"] -= energy_needed
        assigned_bus["available_from"] = arr_dt
        assigned_bus["location"] = end

    # record the actual service trip
    plan_records.append({
        "start location": start,
        "end location": end,
        "start time": dep_time,
        "end time": arr_dt.time(),
        "activity": "service trip",
        "line": line,
        "energy consumption": round(energy_needed, 2),
        "bus": assigned_bus["id"]
    })

# === EXPORT NEW PLAN ===
new_plan = pd.DataFrame(plan_records)
new_plan.to_excel("optimized_busplan.xlsx", index=False)

print(f"✅ Optimized plan saved as 'optimized_busplan.xlsx' using {len(buses)} buses.")
