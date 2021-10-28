# Importing carla
# TODO: Smarter way to get carla imported without having to manually adjust the path to the egg file?
import sys
import os
carla_egg_file = '/home/jo/carla/CARLA_0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-%s.egg' % ('win-amd64' if os.name == 'nt' else 'linux-x86_64')
if not os.path.isfile(carla_egg_file):
    print("WARNING: Carla egg file not found at %s" % carla_egg_file)
sys.path.append(carla_egg_file)

from .sensors import Sensors
from .world import World
from .info_text import InfoText
