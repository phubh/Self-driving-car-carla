from neural import NeuralNetwork as NN   
import  numpy as np 



def main():
    nn = NN([2,2,1])
    x  = [[2],[2]]
    y = [[1]]
    nn.fit(x,y)

main()