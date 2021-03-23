import glob
import os
import sys

# from nn import NN
from neuralNetwork import NeuralNetwork as NN
import numpy as np
import pandas as pd
import cv2
try:
    sys.path.append(glob.glob('./dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import math
import carla
import cv2
from data import data_image
IMG_HEIGHT = 480
IMG_WIDTH = 640
MIN_SPEED = 10
MAX_SPEED = 35
class CarControl:

    def __init__(self, world):
        self.world = world
        self.car = None
        self.camera = None
        self.df = pd.DataFrame(
            {'imageName': [], 'steer': []})
        self.numCaptured = 0
        self.image_raw = None
        self.nn = NN([3072,10,1])
        # self.nn.load('model.npy')

    def spawnCar(self, auto):
        # spawns and returns car actor
        blueprint_library = self.world.get_blueprint_library()
        car_bp = blueprint_library.filter(
            'model3')[0]  # spawn tesla model 3 :)

        if car_bp.has_attribute('color'):
            car_bp.set_attribute('color', '204, 0, 0')  # tesla red color
        transform = self.world.get_map().get_spawn_points()[50]

        self.car = self.world.spawn_actor(car_bp, transform)
        
        print('created %s' % self.car.type_id)
        self.car.set_autopilot(auto)

    def attachCamera(self, manualCtrCar):
        # spawns and attaches camera sensor

        cam_bp = world.get_blueprint_library().find(
            'sensor.camera.rgb')  # it's actually RGBA (4 channel)

        cam_bp.set_attribute('image_size_x', f'{IMG_WIDTH}')
        cam_bp.set_attribute('image_size_y', f'{IMG_HEIGHT}')
        cam_bp.set_attribute('fov', '120')  # field of view

        # time in seconds between sensor captures
        cam_bp.set_attribute('sensor_tick', '0.2')  # change for capturing
        # attach the camera
        spawn_point = carla.Transform(carla.Location(x=2.6,z=1.2),carla.Rotation(-30,0,0))

        # added for manual driving
        if self.car is None:
            self.car = manualCtrCar
        # spawn the camera
        Attachment = carla.AttachmentType
        self.camera = self.world.spawn_actor(
            cam_bp, spawn_point, attach_to=self.car, attachment_type = Attachment.Rigid)

    def engage(self):
        self.camera.listen(lambda image: self.__getLiveFeed(image.raw_data))

    def __getLiveFeed(self, raw_img):
        img = np.array(raw_img).reshape((IMG_HEIGHT, IMG_WIDTH, 4))
        image = img[:, :, :3]
        self.image_raw = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        final_img = data_image(image)
        self.__predict(final_img)

    def __predict(self, img):

        predict = self.nn.predict(img)
        print(predict)
        self.control_v(predict[0][0])

    def record(self):
        # record car actions
        self.camera.listen(lambda image: self.__save(
            image, self.car.get_control()))

    def __save(self, image, control):
        # save the data and the png's

        if self.car.is_at_traffic_light():
            traffic_light = self.car.get_traffic_light()
            if traffic_light.get_state() != carla.TrafficLightState.Green:
                traffic_light.set_state(carla.TrafficLightState.Green)

        path = '../generated_data/image_data/%s.png' % image.frame
        image.save_to_disk(path)
        steer = control.steer
        throttle = control.throttle
        print('Speed:   % 15.0f km/h' % (self.get_speed()))
        self.__createRow(path, throttle, steer)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.image_raw = data_image(array) #array.copy()
        
        

    def __createRow(self, image_path, throttle, steer):
        # creates a row in the csv file

        # get the name without the folder path
        image_name = image_path.rsplit('/', 1)[-1]
        row = [image_name,steer]
        self.df.loc[len(self.df)] = row

        # print(
        #     f'saved row: {image_name} throttle: {throttle} steer: {steer} brake: {brake} , length: {self.numCaptured}')
        self.df.to_csv("../generated_data/data.csv", index=False)
        self.numCaptured += 1

    def destroy(self):

        cv2.destroyAllWindows()

        if self.camera is not None:
            self.camera.destroy()
        if self.car is not None:
            self.car.destroy()

        exit(0)
        print('destroying actors')

    def render(self):
        if self.image_raw is not None:
            cv2.imshow('FCARSIM', self.image_raw)
            key = cv2.waitKeyEx(30)
            if(key == 27):
                cv2.destroyAllWindows()
                return(key)
        return 0
    
    def get_speed(self):
        v = self.car.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
    
    def control_v(self,steer):
        if self.car.is_at_traffic_light():
            traffic_light = self.car.get_traffic_light()
            if traffic_light.get_state() != carla.TrafficLightState.Green:
                self.car.apply_control(carla.VehicleControl(throttle = 0, steer = steer * 0.01, brake = 1.0))
        else:
            # speed = self.get_speed()
            # if speed > speed_limit:
            #     speed_limit = MIN_SPEED 
            # else:
            #     speed_limit = MAX_SPEED
            #     throttle = 1.0 - steering**2 - (speed/speed_limit)**2
            #     print(throttle)
            self.car.apply_control(carla.VehicleControl(throttle = 0.3, steer = steer))
        
        