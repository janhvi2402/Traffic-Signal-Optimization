import os
import sys
import json
import traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

J1 = "J1"
J2 = "J2"

YELLOW_PHASE = {0: 1, 2: 3}


# CHANGE THESE to match each experiment you want to baseline

GREEN_TIME  = 10
YELLOW_TIME = 3


traci.start(["sumo", "-c", "simulation.sumocfg", "--no-warnings"])
traci.simulationStep()

# start both on NS green
j1_phase = 0
j2_phase = 0
traci.trafficlight.setPhase(J1, j1_phase)
traci.trafficlight.setPhase(J2, j2_phase)

cumulative_wait = 0
arrived         = 0
sim_steps       = 0

while traci.simulation.getMinExpectedNumber() > 0:

    # fixed alternating: NS green → yellow → EW green → yellow → repeat
    # no Q-table, no intelligence
    j1_next = 2 if j1_phase == 0 else 0
    j2_next = 2 if j2_phase == 0 else 0

    # yellow transition
    traci.trafficlight.setPhase(J1, YELLOW_PHASE[j1_phase])
    traci.trafficlight.setPhase(J2, YELLOW_PHASE[j2_phase])

    for _ in range(YELLOW_TIME):
        traci.simulationStep()
        sim_steps += 1
        arrived   += len(traci.simulation.getArrivedIDList())
        for veh in traci.vehicle.getIDList():
            cumulative_wait += traci.vehicle.getWaitingTime(veh)

    # green phase
    traci.trafficlight.setPhase(J1, j1_next)
    traci.trafficlight.setPhase(J2, j2_next)
    j1_phase = j1_next
    j2_phase = j2_next

    for _ in range(GREEN_TIME):
        traci.simulationStep()
        sim_steps += 1
        arrived   += len(traci.simulation.getArrivedIDList())
        for veh in traci.vehicle.getIDList():
            cumulative_wait += traci.vehicle.getWaitingTime(veh)

traci.close()

avg_wait = cumulative_wait / max(sim_steps, 1)

print("\n===== BASELINE RESULTS =====")
print(f"Green Time        : {GREEN_TIME}")
print(f"Yellow Time       : {YELLOW_TIME}")
print(f"Sim Steps         : {sim_steps}")
print(f"Vehicles Arrived  : {arrived}")
print(f"Cumulative Wait   : {cumulative_wait:.0f}s")
print(f"Avg Wait/Step     : {avg_wait:.2f}s")

# save with matching filename
result = {
    "green_time":        GREEN_TIME,
    "yellow_time":       YELLOW_TIME,
    "cumulative_wait":   cumulative_wait,
    "avg_wait_per_step": avg_wait,
    "vehicles_arrived":  arrived,
    "steps":             sim_steps
}

filename = f"baseline_gt{GREEN_TIME}_yt{YELLOW_TIME}.json"
with open(filename, "w") as f:
    json.dump(result, f, indent=2)

print(f"Saved: {filename}")