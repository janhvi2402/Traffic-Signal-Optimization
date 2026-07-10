"""
Generates a randomized single-junction route file before every episode.

WHY THIS EXISTS
----------------
If NS/EW demand is roughly the same every episode, `time_in_phase` becomes
a near-perfect proxy for "queue has built up" — the policy learns to key
off elapsed time instead of the actual queue features, because it's a
lower-variance signal. That's why it "changes in the same manner no
matter the queue situation": it never learned to look at the queue at all.

This module regenerates the route file every reset() with randomized,
independently-sampled, asymmetric NS/EW flow rates (and optional bursts),
so elapsed time and queue state are decoupled and the policy is forced to
actually read the queue/wait features to do well.

EDGE IDS — adjust these if your net.xml differs. Inferred from your lane
names (N_J_0 -> edge N_J, etc.):
    incoming:  N_J, S_J, E_J, W_J
    outgoing:  J_N, J_S, J_E, J_W
"""
import os
import random

ROUTE_TEMPLATE = """<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.7"/>

    <route id="ns_through" edges="N_J J_S"/>
    <route id="sn_through" edges="S_J J_N"/>
    <route id="ew_through" edges="E_J J_W"/>
    <route id="we_through" edges="W_J J_E"/>

    <flow id="f_ns" type="car" route="ns_through" begin="0" end="{end}" vehsPerHour="{ns_rate:.1f}" departLane="best" departSpeed="max"/>
    <flow id="f_sn" type="car" route="sn_through" begin="0" end="{end}" vehsPerHour="{sn_rate:.1f}" departLane="best" departSpeed="max"/>
    <flow id="f_ew" type="car" route="ew_through" begin="0" end="{end}" vehsPerHour="{ew_rate:.1f}" departLane="best" departSpeed="max"/>
    <flow id="f_we" type="car" route="we_through" begin="0" end="{end}" vehsPerHour="{we_rate:.1f}" departLane="best" departSpeed="max"/>
</routes>
"""


def generate_single_junction_routes(out_path, episode_seed, sim_end=3600):
    """
    Writes a randomized .rou.xml. Called once per episode from reset(),
    before traci.start().

    Randomization scheme:
      - base NS rate and base EW rate sampled independently and widely
        (not just jittered around one shared value) so NS/EW imbalance
        varies episode to episode.
      - each direction (N->S vs S->N, E->W vs W->E) gets its own small
        perturbation so even NS isn't perfectly symmetric within an
        episode.
      - ~30% of episodes get a "bursty" demand: one direction's rate
        randomly ramps mid-episode, further breaking any fixed
        time-to-queue mapping.
    """
    rng = random.Random(episode_seed)

    ns_base = rng.uniform(30,900)
    ew_base = rng.uniform(30,900)

    ns_rate = ns_base * rng.uniform(0.8, 1.2)
    sn_rate = ns_base * rng.uniform(0.8, 1.2)
    ew_rate = ew_base * rng.uniform(0.8, 1.2)
    we_rate = ew_base * rng.uniform(0.8, 1.2)

    xml = ROUTE_TEMPLATE.format(
        end=sim_end,
        ns_rate=ns_rate, sn_rate=sn_rate,
        ew_rate=ew_rate, we_rate=we_rate,
    )

    # bursty variant: overwrite with two <flow> segments per direction
    # for one randomly chosen axis, so rate steps up/down mid-episode
    if rng.random() < 0.3:
        burst_axis = rng.choice(["NS", "EW"])
        mid = sim_end // 2
        if burst_axis == "NS":
            hi = ns_base * rng.uniform(1.5, 2.5)
            burst_block = (
                f'<flow id="f_ns_a" type="car" route="ns_through" begin="0" end="{mid}" '
                f'vehsPerHour="{ns_rate:.1f}" departLane="best" departSpeed="max"/>\n'
                f'    <flow id="f_ns_b" type="car" route="ns_through" begin="{mid}" end="{sim_end}" '
                f'vehsPerHour="{hi:.1f}" departLane="best" departSpeed="max"/>'
            )
            xml = xml.replace(
                f'<flow id="f_ns" type="car" route="ns_through" begin="0" end="{sim_end}" vehsPerHour="{ns_rate:.1f}" departLane="best" departSpeed="max"/>',
                burst_block,
            )
        else:
            hi = ew_base * rng.uniform(1.5, 2.5)
            burst_block = (
                f'<flow id="f_ew_a" type="car" route="ew_through" begin="0" end="{mid}" '
                f'vehsPerHour="{ew_rate:.1f}" departLane="best" departSpeed="max"/>\n'
                f'    <flow id="f_ew_b" type="car" route="ew_through" begin="{mid}" end="{sim_end}" '
                f'vehsPerHour="{hi:.1f}" departLane="best" departSpeed="max"/>'
            )
            xml = xml.replace(
                f'<flow id="f_ew" type="car" route="ew_through" begin="0" end="{sim_end}" vehsPerHour="{ew_rate:.1f}" departLane="best" departSpeed="max"/>',
                burst_block,
            )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(xml)

    return out_path