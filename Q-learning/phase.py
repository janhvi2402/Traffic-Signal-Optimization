import numpy as np
import traci
import os
import sys

logic = traci.trafficlight.getAllProgramLogics("J1")[0]

for i, phase in enumerate(logic.phases):
    print(i, phase.state)
