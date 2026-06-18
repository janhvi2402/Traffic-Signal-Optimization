import traci
import os
import sys

if 'SUMO_HOME' in os.environ:
    tools=os.path.join(os.environ["SUMO_HOME"],"tools")
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

sumoCmd=[
    "sumo-gui",
    "-c",
    "simulation.sumocfg"
]
traci.start(sumoCmd)

for i in range(1000):
    traci.simulationStep()

vehicles=traci.vehicle.getIDList()
print(len(vehicles))
traci.close()