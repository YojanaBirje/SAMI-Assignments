import numpy as np
import sys
from skimage import transform as tf
from numpy import array, dot
from math import exp
from random import uniform ,random
from random import seed

 
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[round(uniform(-1,1),2) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[round(uniform(-1,1),2) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network
 
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
 
def sigmoid_func(activation):
	return 1.0 / (1.0 + exp(-activation))
 
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = sigmoid_func(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
 
def sigmoid_func_derivative(output):
	return output * (1.0 - output)
 
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * sigmoid_func_derivative(neuron['output'])
 
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']
 
def train_network(network, train, l_rate, n_outputs):
	prev_error=0
	while 1 :
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('lrate=%.3f, error=%.3f' % (l_rate, sum_error))
		if(abs(prev_error-sum_error)<=0.5) :
			break
		prev_error=sum_error

def Loadfile(filename) :
	lines=open(filename,"rb")
	data=[]
	
	for line in lines :
		count = 0
		sample=[]
		row=[]
		size=len(line)
		if size == 33 :
			l=[]
			for i in range(0,32) :
				l.append(int(line[i]))
			row.append(l)
			count +=1

			for line in lines :
				if count <= 31 :
					l=[]
					for i in range(0,32) :
						l.append(int(line[i]))
					row.append(l)
					count +=1
				else :
					row=tf.downscale_local_mean(np.array(row),(4,4)).reshape(1,64)
					r=row[0]
					r=r.tolist()
					sample=r[:]
					sample.append(int(line[1]))
					data.append(sample)
					break
	return data

file=sys.argv[1]

selected_class=[]
selected_class.append(int(sys.argv[2]))
selected_class.append(int(sys.argv[3]))
selected_class.append(int(sys.argv[4]))

dataset=Loadfile(file)

size=len(dataset)
i=0
for j in range(0,size) :
	if dataset[i][64] not in selected_class :
		del dataset[i]
	else :
		dataset[i][64]=selected_class.index(dataset[i][64])
		i=i+1

n_inputs = len(dataset[0]) - 1
n_outputs = 3

for hidden_neurones in [10,16,32] :
	network = initialize_network(n_inputs, hidden_neurones, n_outputs)
	train_network(network, dataset, 0.5, n_outputs)

	hidden=0
	print "No of Hidden Nodes : ",hidden_neurones
	for i in network :
		if hidden == 0:
			for neuron in i :
				print "Hidden Node Weight"
				print neuron['weights']
			hidden=1
		else :
			for neuron in i :
				print "Output Node Weight"
				print neuron['weights']