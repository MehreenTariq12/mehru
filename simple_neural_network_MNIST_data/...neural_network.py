import numpy as np
import scipy.special
class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes, learningrate, train_data, test_data):
        self.inodes = inputnodes
        self.onodes = outputnodes
        self.hnodes = hiddennodes
        self.lr = learningrate
        self.scorecard = []
        self.wih = (np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes)))
        self.who = (np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes)))
        self.activation_function = lambda x:scipy.special.expit(x)
        self.training_data_file = open(train_data, 'r')
        self.training_data_list = self.training_data_file.readlines()
        self.training_data_file.close()
        self.test_data_file = open(test_data, 'r')
        self.test_data_list = self.test_data_file.readlines()
        self.test_data_file.close()
        pass
    def fulyConnectedForward(self,input_list):
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_ouputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_ouputs)
        final_outputs = self.activation_function(final_inputs)
        return (hidden_ouputs, final_outputs)
    def fullyConnectedBackwardPass(self,target_list,final_outputs, hidden_ouputs, inputs):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),np.transpose(hidden_ouputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_ouputs * (1.0 - hidden_ouputs)), np.transpose(inputs))
    def train(self, epochs):
        for e in range(epochs):
            for record in self.training_data_list:
                all_values = record.split(',')
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                targets = np.zeros(self.onodes) + 0.01
                targets[int(all_values[0])] = 0.99
                (hidden_ouputs, final_outputs) = self.fulyConnectedForward(inputs)
                self.fullyConnectedBackwardPass(targets, final_outputs,hidden_ouputs, inputs)
    def test(self):
        for record in self.test_data_list:
            all_values = record.split(',')
            correct_label = int(all_values[0])
            print(correct_label, "correct label")
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
             inputs = np.array(inputs, ndmin=2).T
            hidden_inputs = np.dot(self.wih, inputs)
            hidden_ouputs = self.activation_function(hidden_inputs)
            final_inputs = np.dot(self.who, hidden_ouputs)
            final_outputs = self.activation_function(final_inputs)
            label = np.argmax(final_outputs)
            print(label, "network's answer")
            if label == correct_label:
                self.scorecard.append(1)
            else:
                self.scorecard.append(0)
        scorecard_array = np.asarray(self.scorecard)
        return scorecard_array.sum() / scorecard_array.size


def main():
    training_data_file = "E:\Final Year Research/mnist_train.csv"
    testing_data_file = "E:\Final Year Research/mnist_test.csv"
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.1
    epochs = 1
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, training_data_file, testing_data_file)
    n.train(epochs)
    accuracy = n.test()
    print("accuracy = ", accuracy, "%")


main()
