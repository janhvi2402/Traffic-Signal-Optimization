import os
import random

FLOWS = [
    ("W_to_N1", "W_J1", "J1_N1"),
    ("W_to_S1", "W_J1", "J1_S1"),
    ("W_to_J2", "W_J1", "J2_E"),
    ("W_to_N2", "W_J1", "J2_N2"),
    ("W_to_S2", "W_J1", "J2_S2"),

    ("N1_to_W", "N1_J1", "J1_W"),
    ("N1_to_S1", "N1_J1", "J1_S1"),
    ("N1_to_J2", "N1_J1", "J2_E"),

    ("S1_to_W", "S1_J1", "J1_W"),
    ("S1_to_N1", "S1_J1", "J1_N1"),
    ("S1_to_J2", "S1_J1", "J2_E"),

    ("E_to_N2", "E_J2", "J2_N2"),
    ("E_to_S2", "E_J2", "J2_S2"),
    ("E_to_J1", "E_J2", "J1_W"),
    ("E_to_N1", "E_J2", "J1_N1"),
    ("E_to_S1", "E_J2", "J1_S1"),

    ("N2_to_E", "N2_J2", "J2_E"),
    ("N2_to_S2", "N2_J2", "J2_S2"),
    ("N2_to_J1", "N2_J2", "J1_W"),

    ("S2_to_E", "S2_J2", "J2_E"),
    ("S2_to_N2", "S2_J2", "J2_N2"),
    ("S2_to_J1", "S2_J2", "J1_W"),
]


def generate_multi_junction_routes(out_path, episode_seed, sim_end=3600):

    rng = random.Random(episode_seed)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<routes>",
        '<vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="13.9"/>'
    ]

    for flow_id, src, dst in FLOWS:
        prob = rng.uniform(0.05, 0.30)

        lines.append(
            f'<flow id="{flow_id}" '
            f'type="car" '
            f'from="{src}" '
            f'to="{dst}" '
            f'begin="0" '
            f'end="{sim_end}" '
            f'probability="{prob:.3f}" '
            f'departLane="0" '
            f'departSpeed="max"/>'
        )

    lines.append("</routes>")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    return out_path