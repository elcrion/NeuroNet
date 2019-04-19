"""

Gleb Iakovlev, CSC 578 Homework2 

NN578_network.py
==============

Modified from the NNDL book code "network.py" to be 
compatible with Python 3.
4/2019: Also an instance variable 'termpercent'
is added to do early termination at an arbitrary accuracy.

Also from "network2.py", the function ("load(filename)") is
added to this file, renamed as "load_network(filename)".
This function loads a saved network encoded in a json file.

"""

"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import json

# Third-party libraries
import numpy as np

class Network(object):


    def __init__(self, sizes, termpercent=1.0):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        # 4/2019 nt: addition
        self.termpercent = termpercent


    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a


    def output_results(self,data,n,j,type,show_output):
          """
            Return list for current epoch evaluation. 
            Print out results for current epoch 
            data: []  training/testing data
            n: {int} length of data array
            j: {int} current epoch index
            type: {int} output type {1} training {0} testing 
            show_output: {boolean} print output
          """
          history_list = []
          CE,MSE,correct,loglike = self.evaluate(data)
          history_list.extend([type,correct/n,MSE,CE,loglike])
          type_name = 'Training' if type==1 else 'Test'
          if show_output == True:
            print ("[Epoch {0}] {1}: MSE={2}, CrossEntropy={3}, LogLikelihood ={4}, Correct= {5} / {6} Acc: {7}".format(
                    j,type_name,MSE,CE,loglike,correct, n,correct/n))
          return history_list



    def SGD(self, training_data, epochs, mini_batch_size, eta,print_output,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.

        Extending function:  history_list to keep history of the 
        evaluation results for training (1) and testing (0) 
        Early stopping if accuracy >= tempercent
        print_output:boolean show output 
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        history_list = []     
        for j in range(epochs): #xrange(epochs):
            #random.shuffle(training_data) #4/2019 nt: supressed for now
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            test_list = []
            train_list = self.output_results(training_data,n,j,1,print_output)  

            ## Early stopping here , if  self.termpercent set (< 1)
            if(len(train_list) >0 and self.termpercent < 1 and train_list[1] >= self.termpercent):
                self.output_results(training_data,n,j,1,True)
                return history_list

            if(test_data):
                test_list = self.output_results(test_data,n_test,j,0,print_output)
            history_list.extend([train_list,test_list])
        return history_list

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        Extend method to reduce append list for activations and cost
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        ## initialize list of all layers according to dimensions in sizes 
        activations = [np.zeros((x,1), dtype=int) for x in self.sizes] 
        activations[0] = x #set first layer to input 
        zs = [np.zeros(w.shape) for w in self.weights] 
        ## Go over each layer in the activations list and set calculated sigmoid activations
        for index,(b, w) in enumerate(zip(self.biases, self.weights)):
            zs[index] = (np.dot(w, activations[index])+b)
            activations[index+1] = sigmoid(zs[index])

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        #print(activations) 
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    # Mean Squared Error calculation 
    def mse(self,a,y):
        return 0.5*np.linalg.norm(a-y)**2

    # Cross Entropy calculation     
    def cross_entropy(self,a,y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        # 9/17/2018 nt: ADDITION to accommodate (test_)data with non-scalar y
        correct = 0
        CE = 0
        MSE = 0
        loglike= 0
        if hasattr(test_data[0][1], "__len__"): # to check for scalar type    
            for x,y in test_data:
                index_output = np.argmax(y) # Get index of ouptut layer for higher value 
                a = self.feedforward(x) # current activations
                index_input = np.argmax(a)
                correct += 1 if index_input == index_output else 0 # check for correct result
                CE += self.cross_entropy(a, y)/len(test_data) 
                MSE += self.mse(a, y)/len(test_data) 
                loglike = np.nan_to_num(-np.log(a[index_output].item()))  # get negative logarythm from activations that matches maximum (1) of output layer   
        return CE,MSE,correct,loglike

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

#### Savinging a Network to a json file
def save_network(net, filename):
    """Save the neural network to the file ``filename``."""
    data = {"sizes": net.sizes,
            "weights": [w.tolist() for w in net.weights],
            "biases": [b.tolist() for b in net.biases]#,
            #"cost": str(net.cost.__name__)
           }
    f = open(filename, "w")
    json.dump(data, f)
    f.close()
        
#### Loading a Network from a json file
def load_network(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    #cost = getattr(sys.modules[__name__], data["cost"])
    #net = Network(data["sizes"], cost=cost)
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorize_target(n, target):
    """Return an array of shape (n,1) with a 1.0 in the target position
    and zeroes elsewhere.  The parameter target is assumed to be
    an array of size 1, and the 0th item is the target position (1).

    """
    e = np.zeros((n, 1))
    e[int(target[0])] = 1.0
    return e