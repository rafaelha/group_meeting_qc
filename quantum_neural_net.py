import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

dev = qml.device("strawberryfields.fock", wires=1, cutoff_dim=10)

def layer(v):
    # Matrix multiplication of input layer
    qml.Rotation(v[0], wires=0)
    qml.Squeezing(v[1], 0.0, wires=0)
    qml.Rotation(v[2], wires=0)

    # Bias
    qml.Displacement(v[3], 0.0, wires=0)

    # Element-wise nonlinear transformation
    qml.Kerr(v[4], wires=0)


@qml.qnode(dev)
def quantum_neural_net(var, x=None):
    # Encode input x into quantum state
    qml.Displacement(x, 0.0, wires=0)

    # "layer" subcircuits
    for v in var:
        layer(v)

    return qml.expval(qml.X(0))



def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def cost(var, features, labels):
    preds = [quantum_neural_net(var, x=x) for x in features]
    return square_loss(labels, preds)


data = np.loadtxt('covid.csv', skiprows=1, delimiter=',')
X = data[:,0]
xmax = np.max(X)
X = X/xmax
Y = data[:,1]
ymax = np.max(Y)
Y = Y/ymax

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(X, Y)
plt.xlabel("x", fontsize=18)
plt.ylabel("f(x)", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.tick_params(axis="both", which="minor", labelsize=16)
plt.show()


np.random.seed(0)
num_layers = 6
# var_init = 0.1 * np.random.randn(num_layers, 5)
# print(var_init)

var_init= np.array([[ 0.1672444 , -0.02746   ,  0.08835575,  0.18722878,  0.18929305],
                    [-0.0957819 ,  0.12777733, -0.01524759, -0.05143529,  0.05080969],
                    [ 0.01374947,  0.18976937,  0.08813493, -0.02532584,  0.07449287],
                    [ 0.04830769,  0.16256158,  0.02936889, -0.00605395, -0.05571956],
                    [-0.20538369,  0.10427885,  0.12307134, -0.11345193,  0.24573954],
                    [-0.10939146, -0.01677891,  0.02856047,  0.1096455 ,  0.16252224]])

opt = AdamOptimizer(0.01, beta1=0.9, beta2=0.999)

var = var_init
for it in range(10):
    var = opt.step(lambda v: cost(v, X, Y), var)
    print("Iter: {:5d} | Cost: {:0.7f} ".format(it + 1, cost(var, X, Y)))


var= np.array([[ 0.1672444 , -0.02746   ,  0.08835575,  0.18722878,  0.18929305],
                    [-0.0957819 ,  0.12777733, -0.01524759, -0.05143529,  0.05080969],
                    [ 0.01374947,  0.18976937,  0.08813493, -0.02532584,  0.07449287],
                    [ 0.04830769,  0.16256158,  0.02936889, -0.00605395, -0.05571956],
                    [-0.20538369,  0.10427885,  0.12307134, -0.11345193,  0.24573954],
                    [-0.10939146, -0.01677891,  0.02856047,  0.1096455 ,  0.16252224]])
x_pred = np.linspace(np.min(X), np.max(X)*5, 100)
predictions = np.array([quantum_neural_net(var, x=x_) for x_ in x_pred])

plt.figure()
plt.scatter(X*xmax, Y*ymax)
plt.plot(x_pred*xmax, predictions*ymax, color="green")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.tick_params(axis="both", which="major")
plt.tick_params(axis="both", which="minor")
plt.show()