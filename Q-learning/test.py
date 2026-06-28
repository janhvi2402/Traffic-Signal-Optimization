import os
import sys
import json
import numpy as np
import pickle

def generate_route_file(scenario="medium"):
    
    configs = {
        "low":        {"NS1": 0.1, "EW1": 0.1, "NS2": 0.1, "EW2": 0.1},
        "medium":     {"NS1": 0.3, "EW1": 0.3, "NS2": 0.3, "EW2": 0.3},
        "high":       {"NS1": 0.6, "EW1": 0.6, "NS2": 0.6, "EW2": 0.6},
        "asymmetric": {"NS1": 0.5, "EW1": 0.1, "NS2": 0.1, "EW2": 0.5},
        "rush_hour":  None  # handled separately
    }
    
    if scenario == "rush_hour":
        _generate_rush_hour()
        return
    
    p = configs[scenario]
    
    content = f"""<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50"/>

    <route id="route_NS1" edges="N1_J1 J1_S1"/>
    <route id="route_EW1" edges="W_J1 J1_J2"/>
    <route id="route_NS2" edges="N2_J2 J2_S2"/>
    <route id="route_EW2" edges="J1_J2 J2_E"/>

    <flow id="f_NS1" type="car" route="route_NS1"
          begin="0" end="3600" probability="{p['NS1']}"/>
    <flow id="f_EW1" type="car" route="route_EW1"
          begin="0" end="3600" probability="{p['EW1']}"/>
    <flow id="f_NS2" type="car" route="route_NS2"
          begin="0" end="3600" probability="{p['NS2']}"/>
    <flow id="f_EW2" type="car" route="route_EW2"
          begin="0" end="3600" probability="{p['EW2']}"/>
</routes>"""
    
    with open("routes/test_scenario.rou.xml", "w") as f:
        f.write(content)


def _generate_rush_hour():
    content = """<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50"/>

    <route id="route_NS1" edges="N1_J1 J1_S1"/>
    <route id="route_EW1" edges="W_J1 J1_J2"/>
    <route id="route_NS2" edges="N2_J2 J2_S2"/>
    <route id="route_EW2" edges="J1_J2 J2_E"/>

    <!-- Quiet -->
    <flow id="f_NS1_q1" type="car" route="route_NS1" begin="0"    end="900"  probability="0.1"/>
    <flow id="f_EW1_q1" type="car" route="route_EW1" begin="0"    end="900"  probability="0.1"/>
    <flow id="f_NS2_q1" type="car" route="route_NS2" begin="0"    end="900"  probability="0.1"/>
    <flow id="f_EW2_q1" type="car" route="route_EW2" begin="0"    end="900"  probability="0.1"/>

    <!-- Rush hour -->
    <flow id="f_NS1_rh" type="car" route="route_NS1" begin="900"  end="1800" probability="0.6"/>
    <flow id="f_EW1_rh" type="car" route="route_EW1" begin="900"  end="1800" probability="0.6"/>
    <flow id="f_NS2_rh" type="car" route="route_NS2" begin="900"  end="1800" probability="0.6"/>
    <flow id="f_EW2_rh" type="car" route="route_EW2" begin="900"  end="1800" probability="0.6"/>

    <!-- Tapering -->
    <flow id="f_NS1_tp" type="car" route="route_NS1" begin="1800" end="2700" probability="0.3"/>
    <flow id="f_EW1_tp" type="car" route="route_EW1" begin="1800" end="2700" probability="0.3"/>
    <flow id="f_NS2_tp" type="car" route="route_NS2" begin="1800" end="2700" probability="0.3"/>
    <flow id="f_EW2_tp" type="car" route="route_EW2" begin="1800" end="2700" probability="0.3"/>

    <!-- Quiet again -->
    <flow id="f_NS1_q2" type="car" route="route_NS1" begin="2700" end="3600" probability="0.1"/>
    <flow id="f_EW1_q2" type="car" route="route_EW1" begin="2700" end="3600" probability="0.1"/>
    <flow id="f_NS2_q2" type="car" route="route_NS2" begin="2700" end="3600" probability="0.1"/>
    <flow id="f_EW2_q2" type="car" route="route_EW2" begin="2700" end="3600" probability="0.1"/>
</routes>"""
    
    with open("routes/test_scenario.rou.xml", "w") as f:
        f.write(content)


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

import traci

# CONFIG — GREEN_TIME and YELLOW_TIME must match train.py
GREEN_TIME  = 10
YELLOW_TIME = 3

J1 = "J1"
J2 = "J2"

ACTION_SPACE = [(0,0),(0,2),(2,0),(2,2)]
YELLOW_PHASE = {0: 1, 2: 3}

# LOAD Q-TABLE
with open("qtable.pkl", "rb") as f:
    q_table = pickle.load(f)

print(f"States loaded: {len(q_table)}")


# HELPERS
def bucket(x):
    if x == 0:    return 0
    elif x <= 2:  return 1
    elif x <= 6:  return 2
    elif x <= 12: return 3
    else:         return 4


def get_halted(lane_id):
    return traci.lane.getLastStepHaltingNumber(lane_id)


def get_state(j1_phase, j2_phase):
    if j1_phase == 0:
        j1_green = get_halted("N1_J1_0") + get_halted("S1_J1_0")
        j1_red   = get_halted("W_J1_0")  + get_halted("J2_J1_0")
    else:
        j1_green = get_halted("W_J1_0")  + get_halted("J2_J1_0")
        j1_red   = get_halted("N1_J1_0") + get_halted("S1_J1_0")

    if j2_phase == 0:
        j2_green = get_halted("N2_J2_0") + get_halted("S2_J2_0")
        j2_red   = get_halted("J1_J2_0") + get_halted("E_J2_0")
    else:
        j2_green = get_halted("J1_J2_0") + get_halted("E_J2_0")
        j2_red   = get_halted("N2_J2_0") + get_halted("S2_J2_0")

    return (
        bucket(j1_green), bucket(j1_red),
        bucket(j2_green), bucket(j2_red),
        j1_phase // 2,
        j2_phase // 2,
    )


# RUN SIMULATION


cumulative_wait = 0
arrived         = 0
sim_steps       = 0

SCENARIOS = ["low", "medium", "high", "asymmetric", "rush_hour"]

for scenario in SCENARIOS:
    print(f"\n===== SCENARIO: {scenario.upper()} =====")
    
    # Generate route file for this scenario
    generate_route_file(scenario)
    
    # Your existing simulation code stays exactly the same
    traci.start(["sumo", "-c", "test.sumocfg", "--no-warnings"])
    traci.simulationStep()

    j1_phase = 0
    j2_phase = 0
    traci.trafficlight.setPhase(J1, j1_phase)
    traci.trafficlight.setPhase(J2, j2_phase)

    cumulative_wait = 0
    arrived = 0
    sim_steps = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        state = get_state(j1_phase, j2_phase)
        action_idx = int(np.argmax(q_table[state])) if state in q_table else 0
        j1_new, j2_new = ACTION_SPACE[action_idx]

        if j1_new != j1_phase:
            traci.trafficlight.setPhase(J1, YELLOW_PHASE[j1_phase])
        if j2_new != j2_phase:
            traci.trafficlight.setPhase(J2, YELLOW_PHASE[j2_phase])

        if j1_new != j1_phase or j2_new != j2_phase:
            for _ in range(YELLOW_TIME):
                traci.simulationStep()
                sim_steps += 1
                arrived += len(traci.simulation.getArrivedIDList())
                for veh in traci.vehicle.getIDList():
                    cumulative_wait += traci.vehicle.getWaitingTime(veh)

        traci.trafficlight.setPhase(J1, j1_new)
        traci.trafficlight.setPhase(J2, j2_new)
        j1_phase, j2_phase = j1_new, j2_new

        for _ in range(GREEN_TIME):
            traci.simulationStep()
            sim_steps += 1
            arrived += len(traci.simulation.getArrivedIDList())
            for veh in traci.vehicle.getIDList():
                cumulative_wait += traci.vehicle.getWaitingTime(veh)

    traci.close()

    avg_wait = cumulative_wait / max(sim_steps, 1)

    # Save result with scenario field added
    result = {
        "scenario":          scenario,   # ← new field
        "alpha":             0.1,
        "gamma":             0.95,
        "episodes":          100,
        "epsilon_decay":     0.98,
        "green_time":        GREEN_TIME,
        "yellow_time":       YELLOW_TIME,
        "cumulative_wait":   cumulative_wait,
        "avg_wait_per_step": avg_wait,
        "vehicles_arrived":  arrived,
        "steps":             sim_steps
    }

    os.makedirs("results", exist_ok=True)
    filename = f"result_a{result['alpha']}_g{result['gamma']}_ep{result['episodes']}_{scenario}.json"
    with open(f"results/{filename}", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Avg Wait/Step : {avg_wait:.2f}s")
    print(f"Vehicles      : {arrived}")
    print(f"Saved         : {filename}")