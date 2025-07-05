import neunet as nn
import numpy as np
import json

'''
#one more expample:

inp = np.array([[0,0,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[0,1,0]])
out = np.array([[0],[1],[0],[1],[1],[0]])

network = nn.Network([3, 1], [0])

network.back_propagation(10000, inp, out)

network.run([0,0,0])

network.save_weights()

...

with open('data.json', 'r', encoding='utf-8') as file:
    load = json.load(file)

ne = nn.Network([3, 1], [0])
ne.run([1,1,1], load)

'''


'''
#example of mnist dataset

def to_out(x):
    arr = [0 for i in range(10)]
    arr[x] = 1
    return np.array(arr)

data = np.load('mnist.npz', allow_pickle=True)
lst = data.files

inp_train = data['x_train'].reshape((60000, 784)) / 255
inp_test = data['x_test'].reshape((10000, 784)) / 255
out_train = np.zeros((60000,10))
out_test = np.zeros((10000,10))

for i in range(len(data['y_train'])): out_train[i] = to_out(data['y_train'][i])
for i in range(len(data['y_test'])): out_test[i] = to_out(data['y_test'][i])

print('Loaded')

network = nn.Network([784, 10, 10], [1], l=0.01)

network.back_propagation(12, inp_train, out_train)

network.run(inp_test[0])
print(out_test[0])

network.save_weights()

def to_number(x):
    x = x[0]
    x = list(map(lambda x: x.tolist(), x))
    numb = x.index(max(x))
    return numb

accuracy = 0
for i in range(len(inp_test)):
    result = network.run(inp_test[i])
    #print('NN:', to_number(result), 'ans:', to_number(np.array([out_test[1]])))
    if to_number(result) == to_number(np.array([out_test[i]])):
        accuracy += 1

print('accuracy:', str(accuracy/len(inp_test)*100)+'%')
'''

'''
from PIL import Image

img = np.array(Image.open('image.png').convert('L')).reshape(784) / 255

with open('data.json', 'r', encoding='utf-8') as file:
    load = json.load(file)

network = nn.Network([784, 10, 10], [1], l=0.5)

def to_number(x):
    x = x[0]
    x = list(map(lambda x: x.tolist(), x))
    numb = x.index(max(x))
    return numb

result = network.run(img, load)
print(to_number(result))'''