import traci

J1 = "J1"
J2 = "J2"

def run_offset_fixed_time(cycle_ns=42, cycle_ew=42, yellow=3, max_steps=3600):
    """
    Same offset fixed-time baseline logic as PPO's test.py, ported to
    raw TraCI. J2 is offset by half a cycle so signals aren't
    synchronized — this is the SAME baseline used for both PPO and
    Q-learning comparisons, so improvement % means the same thing
    for both.

    Returns (cumulative_wait, avg_wait_per_step, vehicles_arrived, steps)
    """
    full_cycle = cycle_ns + yellow + cycle_ew + yellow
    half_cycle = full_cycle // 2

    traci.trafficlight.setPhase(J1, 0)
    traci.trafficlight.setPhase(J2, 0)

    cumulative_wait = 0.0
    arrived = 0
    steps = 0
    t = 0

    def phase_for(pos):
        if pos < cycle_ns:
            return 0  # NS green
        elif pos < cycle_ns + yellow:
            return 1  # NS yellow
        elif pos < cycle_ns + yellow + cycle_ew:
            return 2  # EW green
        else:
            return 3  # EW yellow

    while traci.simulation.getMinExpectedNumber() > 0 and steps < max_steps:
        pos_j1 = t % full_cycle
        pos_j2 = (t + half_cycle) % full_cycle

        traci.trafficlight.setPhase(J1, phase_for(pos_j1))
        traci.trafficlight.setPhase(J2, phase_for(pos_j2))

        traci.simulationStep()
        steps += 1
        t += 1

        arrived += len(traci.simulation.getArrivedIDList())
        for veh in traci.vehicle.getIDList():
            cumulative_wait += traci.vehicle.getWaitingTime(veh)

    avg_wait = cumulative_wait / max(steps, 1)
    return cumulative_wait, avg_wait, arrived, steps