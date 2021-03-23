import glob
import os
import sys
import time

from car_control import CarControl

try:
    sys.path.append(glob.glob('./dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


def connect():
    client = carla.Client('localhost', 2000)
    client.set_timeout(25.0)
    world = client.load_world("Town05")
    return world


def collectDataAuto(seconds):
    world = connect()
    car = CarControl(world)
    
    car.spawnCar(True)
    car.attachCamera(None)
    car.record()

    start = current_time()
    end = start
    while ( end - start < seconds):
        end = current_time()
        key = car.render()
        if (key == 27):
            break    
    car.destroy()

def drive(seconds):
    world = connect()
    car = CarControl(world)

    car.spawnCar(False)
    car.attachCamera(None)
    car.engage()
    start = current_time()
    end = start
    while ( end - start < seconds):
        end = current_time()
        key = car.render()
        if (key == 27):
            break    
    car.destroy()


def main():

    if sys.argv[1] == 'collect':
        collectDataAuto(int(sys.argv[2]))
    elif sys.argv[1] == 'drive':
        drive(int(sys.argv[2]))

def current_time():
    return round(time.time())
if __name__ == '__main__':
    main()
