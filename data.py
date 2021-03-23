import cv2
import pandas as pd
import numpy as np

class Data:

    def __init__(self,image,steer):
        
        self.steer = steer
        self.image = image
    
    def make_data_train(self):
        # result = np.array([[self.steer]])
        return (data_image(self.image), self.steer)




def scale(img,percent):

        width = int(img.shape[1] * percent / 100)
        height = int(img.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def data_image(img):
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0.0, -1.0, 0.0], 
                   [-1.0, 4.0, -1.0],
                   [0.0, -1.0, 0.0]])

    kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)

    #filter the source image
    img_rst = cv2.filter2D(img,-1,kernel)
    img_rst = img_rst[:360,:]
    # # img = data_image(img_rst)
    img_rst = scale(img_rst,100)
    # thresh = 128
    # cv2.imshow('FCARSIM',img_rst)
    # # threshold the image
    # img_rst = cv2.threshold(img_rst, thresh, 255, cv2.THRESH_BINARY)[1]
    return  img_rst#.reshape(1,-1)/225

def run():
    img = cv2.imread('../generated_data/image_data/2938.png')
    
    img = data_image(img)
    print(len(img[0]))
    
    
    while True:
        key = cv2.waitKeyEx(30)
        if(key == 27):
            cv2.destroyAllWindows()
            break
    
    # print(len(img.reshape(-1,1)))
    
if __name__ == '__main__':

    run()
