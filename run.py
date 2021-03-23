from nn import NN
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from neuralNetwork import NeuralNetwork as NN
# load data
file = open('data_train.pkl', 'rb')

# dump information to that file
training_data = pickle.load(file)
X = np.array([data[0][0] for data in training_data])
y = np.array([data[1] for data in training_data])
# close the file
file.close()
# run NN
# nn.load('model.npy')
nn = NN([8192,15,1],0.0002)
nn.fit(X, y, 600, 30)
nn.save('model.npy')
# # nn.load('model.npy')
# # print(f'Weight: {}'nn.w)
# cost = nn.train(training_data, 6000, 200, 0.0001, 0.0, 0.0)
# # correct = nn.predict(test_data)
# # total = len(test_data) 
# # print('Evaluation: {0} / {1} = {2}%'.format(correct, total, 100 * correct/total))
# nn.save()

# plt.title('Dropout')
# plt.plot(np.arange(1, 6001), cost)
# plt.xlabel('epoch')
# plt.ylabel('cost')
# plt.grid()
# plt.show()
for (x, target) in zip(X, y):
   predict = nn.predict(x)
   print('[INFO] ground-truth: {}, predict: {}'.format(target, predict))