import neunet as nn
import numpy as np
import json


inp = np.array([[0,0,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[0,1,0]])
out = np.array([[0],[1],[0],[1],[1],[0]])

network = nn.Network([3, 1], [0])

network.back_propagation(10000, inp, out)

network.run([0,0,0])

network.save_weights()

'''with open('data.json', 'r', encoding='utf-8') as file:
    load = json.load(file)

ne = nn.Network([3, 1], [0])
ne.run([1,1,1], load)
'''



