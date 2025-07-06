import numpy as np
import json

class Network():
    def __init__(self, neurons: list, bias: list=[], type: str='p', activate_func: str='sigmoid', l: float=0.1):
        #activating functions
        self.funcs = {'sigmoid': self.sigmoid, 'sigmoid_d': self.sigmoid_d}

        #choosed function and it's derivetive
        self.function = self.funcs[activate_func]
        self.derivative = self.funcs[f'{activate_func}_d']

        #lambda variable (speed of training)
        self.l = l

        #declaration af all weights and all biases
        np.random.seed(1)
        self.weights_all = [2*np.random.random((neurons[i],neurons[i+1]))-1 for i in range(len(neurons)-1)]
        self.bias_indexes = bias
        '''self.deltas = [np.zeros((neurons[i],neurons[i+1])) for i in range(len(neurons)-1)]
        self.deltas_bias = [np.zeros((1,neurons[index+1])) for index in bias]'''
        self.neurons_amount = neurons
        self.neurons = [[np.zeros(index) for index in neurons[:-1]] for i in range(1000)]
        self.deltas = [[np.zeros(index) for index in list(reversed(neurons[1:]))] for i in range(1000)]
        if bias != []:
            try:
                self.bias_all = [2*np.random.random((1,neurons[index+1]))-1 for index in bias]
            except IndexError:
                print("IndexError: Bias neuron can't be on the last layer")
                exit(0)

    #activating function and it's derivative
    def sigmoid (self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_d (self, x):
        return x*(1-x)

    #training methods
    def back_propagation(self, epo: int, inputt: np.array, outputt: np.array):
        #concatenating input and output training sets
        train = np.concatenate((inputt, outputt), axis=1)
        lens = len(inputt[0])

        #training
        for e in range(epo):
            #randomising data to preventing nn form wrong regularities
            np.random.shuffle(train)
            for it, element in enumerate(train):
                if it % 1000 == 0 and it != 0:
                    iterations = 1000
                    deltas_neurons = [[self.deltas[i][j] * self.neurons[i][-(j+1)].T  for j in range(len(self.neurons[0]))] for i in range(iterations)]
                    deltas_neurons_sum = [np.squeeze(np.sum(np.array([[deltas_neurons[j][i]] for j in range(len(deltas_neurons))])), axis=0) for i in range(len(deltas_neurons[0]))]
                    deltas_sum = [np.sum(np.array([[np.squeeze(self.deltas[j][i])] for j in range(iterations)]), axis=0) for i in range(len(self.deltas[0]))]
                    #recent index of bias neuron's weights sequence number
                    bias_index = -1

                    #all the layer's weights adjustment
                    for i in range(len(self.weights_all)):
                        self.weights_all[-(i+1)] -= self.l * deltas_neurons_sum[i]
                        #biases weights adjustment
                        if (len(self.weights_all)-1)-i in self.bias_indexes:
                            self.bias_all[bias_index] -= self.l * deltas_sum[i]
                            bias_index -= 1

                    self.neurons = [[np.zeros(index) for index in self.neurons_amount[:-1]] for i in range(1000 if len(train) - it >= 1000 else len(train)%1000)]
                    self.deltas = [[np.zeros(index) for index in list(reversed(self.neurons_amount[1:]))] for i in range(1000 if len(train) - it >= 1000 else len(train)%1000)]

                #list of the values of neurons from all the layers
                neurons = [np.array([element[0:lens]])]
                #recent index of bias neuron's weights sequence number
                bias_index = 0

                #all layer's propagation
                for i in range(len(self.weights_all)):
                    input_ = neurons[i]
                    if i in self.bias_indexes:
                        input_hidden = input_ @ np.array(self.weights_all[i]) + np.array(self.bias_all[bias_index])
                        bias_index += 1
                    else:
                        input_hidden = input_ @ np.array(self.weights_all[i])
                    input_hidden_activated = self.function(input_hidden)
                    neurons.append(input_hidden_activated)

                #error calculation
                err = neurons[-1] - np.array([element[lens:]])
                #list of the values of deltas form all the layers
                '''deltas = [err * self.derivative(neurons[-1])]
                
                #recent index of bias neuron's delta's sequence number
                bias_index = -1
                #all the layer's deltas adjustment
                for i in range(len(self.weights_all)): 
                    self.deltas[-(i+1)] += deltas[i] * neurons[-(i+2)].T
                    dn = (deltas[i] @ self.weights_all[-(i+1)].T) * self.derivative(neurons[-(i+2)])
                    deltas.append(dn)
                    if (len(self.weights_all)-1)-i in self.bias_indexes:
                        self.deltas_bias[bias_index] += deltas[i]
                        bias_index -= 1
                
                if it % 1000 == 0 or it == len(train)-1:
                    #recent index of bias neuron's weights sequence number
                    bias_index = -1

                    #all the layer's weights adjustment
                    for i in range(len(self.weights_all)):
                        self.weights_all[-(i+1)] -= self.l * self.deltas[-(i+1)]
                        #biases weights adjustment
                        if (len(self.weights_all)-1)-i in self.bias_indexes:
                            self.bias_all[bias_index] -= self.l * self.deltas_bias[bias_index]
                            bias_index -= 1

                    for i in range(len(neurons)-1): self.deltas[i] *= 0 
                    for i in range(len(self.deltas_bias)): self.deltas_bias[i] *= 0 '''

                self.neurons[it%1000] = neurons[:-1]
                self.deltas[it%1000][0] = err * self.derivative(neurons[-1])
                for i in range(1, len(self.weights_all)): 
                    self.deltas[it%1000][i] = (self.deltas[it%1000][i-1] @ self.weights_all[-(i)].T) * self.derivative(neurons[-(i+1)])
                
                if it == len(train)-1:
                    iterations = it%1000
                    deltas_neurons = [[self.deltas[i][j] * self.neurons[i][-(j+1)].T  for j in range(len(self.neurons[0]))] for i in range(iterations)]
                    deltas_neurons_sum = [np.squeeze(np.sum(np.array([[deltas_neurons[j][i]] for j in range(len(deltas_neurons))])), axis=0) for i in range(len(deltas_neurons[0]))]
                    deltas_sum = [np.sum(np.array([[np.squeeze(self.deltas[j][i])] for j in range(iterations)]), axis=0) for i in range(len(self.deltas[0]))]
                    #recent index of bias neuron's weights sequence number
                    bias_index = -1

                    #all the layer's weights adjustment
                    for i in range(len(self.weights_all)):
                        self.weights_all[-(i+1)] -= self.l * deltas_neurons_sum[i]
                        #biases weights adjustment
                        if (len(self.weights_all)-1)-i in self.bias_indexes:
                            self.bias_all[bias_index] -= self.l * deltas_sum[i]
                            bias_index -= 1

                    self.neurons = [[np.zeros(index) for index in self.neurons_amount[:-1]] for i in range(1000 if len(train) >= 1000 else len(train)%1000)]
                    self.deltas = [[np.zeros(index) for index in list(reversed(self.neurons_amount[1:]))] for i in range(1000 if len(train) >= 1000 else len(train)%1000)]

            if e - int(epo/4) == 0 or e - int(epo/2) == 0 or e - int(epo/1.333333333) == 0:
                print(f'done: {round(e/epo, 2)*100}%')
        print(f'done: 100%')
        
    def run(self, inputt, load=None) -> np.array:
        if load != None:
            self.weights_all = list(map(lambda x: np.array(x), load['weights']))
            if self.bias_indexes != []:
                self.bias_all = list(map(lambda x: np.array(x), load['bias']))
        #list of the values of neurons from all the layers
        neurons = [np.array(inputt)]
        #recent index of bias neuron's weights sequence number
        bias_index = 0
        #all layer's propagation
        for i in range(len(self.weights_all)):
            input_ = neurons[i]
            if i in self.bias_indexes:
                input_hidden = input_ @ np.array(self.weights_all[i]) + np.array(self.bias_all[bias_index])
                bias_index += 1
            else:
                input_hidden = input_ @ np.array(self.weights_all[i])
            input_hidden_activated = self.function(input_hidden)
            neurons.append(input_hidden_activated)
        print(neurons[-1])
        return neurons[-1]
        
    def save_weights(self):
        if self.bias_indexes != []:
            data = {"weights":list(map(lambda x: x.tolist(), self.weights_all)),  "bias":list(map(lambda x: x.tolist(), self.bias_all))}
        else:
            data = {"weights":list(map(lambda x: x.tolist(), self.weights_all))}
        with open('data.json', "w", encoding='utf-8') as file:
            json.dump(data, file)

'''
------------how to run example---------------

inp = np.array([[0,0],[1,1],[0,1],[1,0]])
out = np.array([[0],[0],[1],[1]])

nn = Network([2, 3, 4, 1], [0, 1, 2])

nn.back_propagation(10000, inp, out)

nn.run([0,1])

nn.save_weights()

...

with open('data.json', 'r', encoding='utf-8') as file:
    load = json.load(file)

nn = Network([2, 3, 4, 1], [0, 1, 2])
nn.run([0,0], load)

---------------------------------------------
'''



