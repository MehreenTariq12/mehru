import cv2
import numpy as np
import scipy.misc
from matplotlib import pyplot as plt
class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes, learningrate):
        self.inodes = inputnodes
        self.onodes = outputnodes
        self.hnodes = hiddennodes
        self.lr = learningrate
        self.wih = (np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes)))
        self.who = (np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes)))
        self.activation_function = lambda x:scipy.special.expit(x)
        pass
    def train(self,input_list,target_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_ouputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_ouputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),np.transpose(hidden_ouputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_ouputs * (1.0 - hidden_ouputs)), np.transpose(inputs))

        pass
    def query(self,input_list):
        inputs = np.array(input_list,ndmin=2).T
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_ouputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who,hidden_ouputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

def main():
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.1
    epochs = 10
    n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
    training_data_file = open("E:\Final Year Research/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

    #print(len(data_list))
    #print(data_list[0])
    #all_values = data_list[2].split(',')
    #inputs = (np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
    #targets = np.zeros(output_nodes) + 0.01
    #targets[int(all_values[0])] = 0.99
    #n.train(inputs,targets)
    scorecard = []
    test_data_file = open("E:\Final Year Research/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        print(correct_label,"correct label")
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = np.argmax(outputs)
        print(label, "network's answer")
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
    scorecard_array = np.asarray(scorecard)
    print("performance = ", scorecard_array.sum() / scorecard_array.size)
    #print(all_values[0])
    #image_array = np.asfarray(all_values[1:]).reshape((28,28))
    #plt.imshow(image_array, cmap='Greys' ,interpolation='nearest')
    #plt.show()
    #print(n.query((np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01))
    cv2.waitKey(0)

main()
