import os
import sys
import traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

# Fixed cycle that mirrors your tlLogic definition exactly
# phase 0: NS green = 42s, phase 1: yellow = 3s,
# phase 2: EW green = 42s, phase 3: yellow = 3s
# Total cycle = 90s — this is what SUMO runs by default

J1 = "J1"
J2 = "J2"

FIXED_GREEN  = 42   # matches your tlLogic duration
YELLOW_TIME  = 3

traci.start(["sumo-gui", "-c", "simulation.sumocfg"])
traci.simulationStep()

# Force SUMO to use the static program (already default, but explicit)
traci.trafficlight.setProgram(J1, "0")
traci.trafficlight.setProgram(J2, "0")

cumulative_wait = 0
arrived         = 0
sim_steps       = 0

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    sim_steps += 1

    # count vehicles that finished this step
    arrived += len(traci.simulation.getArrivedIDList())

    # sum waiting time across all active vehicles
    step_wait = 0
    for veh in traci.vehicle.getIDList():
        step_wait += traci.vehicle.getWaitingTime(veh)
    cumulative_wait += step_wait

traci.close()

print("\n===== BASELINE RESULTS (Fixed 42s Cycle) =====")
print(f"Simulation Steps  : {sim_steps}")
print(f"Vehicles Arrived  : {arrived}")
print(f"Cumulative Wait   : {cumulative_wait:.0f} s")
print(f"Avg Wait/Step     : {cumulative_wait / max(sim_steps, 1):.3f} s")

import json

baseline_result = {
    "cumulative_wait":   cumulative_wait,
    "avg_wait_per_step": cumulative_wait / max(sim_steps, 1),
    "vehicles_arrived":  arrived,
    "steps":             sim_steps
}

with open("results/baseline_result.json", "w") as f:
    json.dump(baseline_result, f, indent=2)

print("Saved: baseline_result.json")