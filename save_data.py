import cv2
import pandas as pd
import numpy as np
from data import Data as dt
import pickle

def read_data_trainning(path):

    info = pd.read_csv(path + 'data.csv')
    steer = info['steer']
    path_image = info['imageName']
    pathImg = path + 'image_data/'
    # return [dt(cv2.imread(pathImg + path_image[x]),round(throttle[x],3),round(steer[x],3)).make_data_train() if round(steer[x],3) != 0  or  steer[x] >=0 else dt(cv2.imread(pathImg + path_image[x]),round(throttle[x],3),-0.001).make_data_train() for x in range(len(info) - 1 ) if  throttle[x + 1] != throttle[x] and steer[x - 1] != steer[x + 1]]
    return [dt(cv2.imread(pathImg + path_image[x]),round(steer[x],4)).make_data_train() for x in range(len(info))]

def run():
    data = read_data_trainning('../generated_data/')
    pickle.dump(data,open("data_train.pkl","wb"))
    # mfile = open('data_train.pkl', 'rb')
    # data = pickle.load(mfile)
    
    # mfile.close()
    
run()