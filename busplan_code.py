import pandas as pd
from datetime import datetime, timedelta

# -------------------------------
# Parameters
# -------------------------------
BATTERY_CAPACITY = 300  # kWh
CHARGING_RATE_FAST = 450  # kW/h (0-90% SOC)
CHARGING_RATE_SLOW = 60   # kW/h (90-100% SOC)
MIN_CHARGING_MINUTES = 15
IDLE_CONSUMPTION_KW = 5  # kW per hour
SAFETY_MARGIN = 0.1       # 10%
SOH = 0.9                 # Example: 90% battery health

# -------------------------------
# Helper Functions
# -------------------------------
def parse_time(time_str):
    return datetime.strptime(str(time_str), "%H:%M")

def energy_consumption(distance_km):
    # Energy consumption between 0.7-2.5 kWh/km, average ~1.5 kWh/km
    return distance_km * 1.5

def idle_energy(minutes):
    return IDLE_CONSUMPTION_KW * (minutes / 60)

def charge_energy(soc_kwh, duration_minutes, soh_capacity):
    energy_added = 0
    soc_percentage = soc_kwh / soh_capacity
    remaining_time = duration_minutes

    # Fast charging 0-90%
    if soc_percentage < 0.9:
        max_fast_energy = 0.9 * soh_capacity - soc_kwh
        max_fast_time = max_fast_energy / CHARGING_RATE_FAST * 60
        time_fast = min(remaining_time, max_fast_time)
        energy_added += time_fast / 60 * CHARGING_RATE_FAST
        remaining_time -= time_fast
        soc_kwh += energy_added

    # Slow charging 90-100%
    if remaining_time > 0:
        max_slow_energy = soh_capacity - soc_kwh
        energy_added += min(max_slow_energy, remaining_time / 60 * CHARGING_RATE_SLOW)

    return energy_added

# -------------------------------
# Bus Class
# -------------------------------
class Bus:
    def __init__(self, bus_id, soh=SOH):
        self.bus_id = bus_id
        self.soh_capacity = BATTERY_CAPACITY * soh
        self.soc = self.soh_capacity
        self.location = "ehvapt"
        self.available_from = parse_time("05:00")
        self.plan = []

    def assign_trip_strict(self, trip, distance_matrix):
        trip_start = parse_time(trip["departure_time"])

        # 1. Material trip if bus not at start location
        if self.location != trip["start"]:
            dist_row = distance_matrix[(distance_matrix["start"] == self.location) & (distance_matrix["end"] == trip["start"])]
            travel_time_min = int(dist_row["max_travel_time"].values[0])
            distance_m = dist_row["distance_m"].values[0]

            energy_needed = energy_consumption(distance_m / 1000)
            min_soc = SAFETY_MARGIN * self.soh_capacity

            # Charging if SOC too low
            if self.soc - energy_needed < min_soc:
                needed_energy = energy_needed + min_soc - self.soc
                charging_minutes = max(MIN_CHARGING_MINUTES, needed_energy / CHARGING_RATE_FAST * 60)
                energy_added = charge_energy(self.soc, charging_minutes, self.soh_capacity)
                self.plan.append({
                    "start location": self.location,
                    "end location": self.location,
                    "start time": self.available_from,
                    "end time": self.available_from + timedelta(minutes=charging_minutes),
                    "activity": "charging",
                    "line": None,
                    "energy": -energy_added,
                    "bus": self.bus_id
                })
                self.soc += energy_added
                self.available_from += timedelta(minutes=charging_minutes)

            # Material trip
            self.plan.append({
                "start location": self.location,
                "end location": trip["start"],
                "start time": self.available_from,
                "end time": self.available_from + timedelta(minutes=travel_time_min),
                "activity": "material_trip",
                "line": None,
                "energy": energy_needed,
                "bus": self.bus_id
            })
            self.soc -= energy_needed
            self.location = trip["start"]
            self.available_from += timedelta(minutes=travel_time_min)

        # 2. Idle until exact departure
        if self.available_from < trip_start:
            idle_minutes = (trip_start - self.available_from).total_seconds() / 60
            idle_energy_consumed = idle_energy(idle_minutes)
            self.plan.append({
                "start location": self.location,
                "end location": self.location,
                "start time": self.available_from,
                "end time": trip_start,
                "activity": "idle",
                "line": None,
                "energy": idle_energy_consumed,
                "bus": self.bus_id
            })
            self.available_from = trip_start

        # 3. Service trip (strict timetable)
        dist_row = distance_matrix[(distance_matrix["start"] == trip["start"]) & (distance_matrix["end"] == trip["end"])]
        distance_m = dist_row["distance_m"].values[0]
        travel_time_min = int(dist_row["max_travel_time"].values[0])
        energy_needed = energy_consumption(distance_m / 1000)

        self.plan.append({
            "start location": trip["start"],
            "end location": trip["end"],
            "start time": trip_start,
            "end time": trip_start + timedelta(minutes=travel_time_min),
            "activity": "service_trip",
            "line": trip["line"],
            "energy": energy_needed,
            "bus": self.bus_id
        })
        self.soc -= energy_needed
        self.location = trip["end"]
        self.available_from = trip_start + timedelta(minutes=travel_time_min)

# -------------------------------
# Load Excel files
# -------------------------------
busplan = pd.read_excel("Bus_Planning-FIXED.xlsx")
timetable = pd.read_excel("Timetable.xlsx")
distance_matrix = pd.read_excel("DistanceMatrix.xlsx")

# -------------------------------
# Schedule Buses
# -------------------------------
buses = [Bus(bus_id=1), Bus(bus_id=2)]  # example two buses
for _, trip in timetable.iterrows():
    # Pick first available bus (can be improved to minimize number of buses)
    buses.sort(key=lambda b: b.available_from)
    bus = buses[0]
    bus.assign_trip_strict(trip, distance_matrix)

# -------------------------------
# Output Bus Plan
# -------------------------------
all_plans = []
for bus in buses:
    all_plans.extend(bus.plan)

bus_plan_df = pd.DataFrame(all_plans)
bus_plan_df.to_excel("Busplan_timetable_compliant.xlsx", index=False)
print("Nieuwe busplan opgeslagen als 'Busplan_timetable_compliant.xlsx'")
