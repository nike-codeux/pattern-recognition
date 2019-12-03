import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class Perceptron(object):
    def __init__(self, theta=0, lrate=0.01, epochs=100):
        self.theta = theta
        self.lrate = lrate
        self.epochs = epochs
    
    def signal(self, x):
        return 1 if x >= self.theta else 0

    # training function
    def fit(self, X, d, theta=None, lrate=None, epochs=None):
        
        if theta is None: theta = self.theta
        if lrate is None: lrate = self.lrate
        if epochs is None: epochs = self.epochs
        
        bias = np.ones((X.shape[0], 1), dtype=float)
        X = np.concatenate((bias, X), axis = 1)
        
        w = np.random.randn(X.shape[1])
        g = np.zeros(d.shape)
        y = np.zeros(d.shape)

        t = 0
        cost = []
        error_treshold = 0.001
        e = 1.0
        while t < epochs and e > error_treshold:
            e = 0
            for i in range(X.shape[0]):
                # computing output (signal function)
                y[i] = self.signal(w @ X[i])

                # updating weights and computing training error
                w += lrate * (d[i] - y[i]) * X[i]
                e += abs(d[i] - y[i])
                
            e /= X.shape[0]
            cost.append(e)
            t += 1

        self.epochs = t
        return w, cost


    def predict(self, w, X):
        y = np.zeros(X.shape[0])
        bias = np.ones((X.shape[0], 1), dtype=float)
        X = np.concatenate((bias, X), axis = 1)
        
        for i in range(X.shape[0]):
            y[i] = self.signal(w @ X[i])
            
        return y   
    
    def evaluate(self, w, X, y_test):
        y_pred = self.predict(w, X)
        return np.mean(np.abs(y_pred - y_test))
        

def split_dataset(x, rate=0.7):
    index = round(x.shape[0]*rate)
    return x[:index], x[index:]
    
def hyperplane(X, y, weights, bias):
    xvalues = [np.min(X[:,0]), np.max(X[:,0])]
    
    slope = - weights[1] / weights[2]
    intercept = - bias / weights[2]
    x_hyperplane = np.linspace(0,10)
    y_hyperplane = slope * x_hyperplane + intercept
    fig = plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.plot(x_hyperplane, y_hyperplane, '-')
    plt.title("Dataset e hiperplano de decisão")
    plt.xlabel("Primeiro feature")
    plt.ylabel("Segundo feature")
    plt.show()
        
def main():

    features, labels = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=2.0, random_state=18)
    print("Amostras: " , features.shape)
    print("Classes: ", np.unique(labels))

    X = np.array(features)
    y = np.array(labels)
    
    train_x, test_x = split_dataset(X)
    train_y, test_y = split_dataset(y)

    model = Perceptron(lrate=0.01, epochs=50)
    w, cost = model.fit(train_x, train_y)
    erroTest = model.evaluate(w, test_x, test_y)
    erroTrain = model.evaluate(w, train_x, train_y)
    
    print('Total de épocas necessárias para o aprendizado: ', model.epochs)
    print('Pesos aprendidos: ', w)
    print('Erro em test: ', erroTest)
    print('Erro em treinamento: ', erroTrain)

    # plot the data
    fig = plt.figure()
    plt.plot(range(1, len(cost)+1), cost, color='purple')
    plt.grid(True)
    plt.xlabel('# épocas')
    plt.ylabel('(Erro absoluto)')
    plt.title('Custo de treinamento em função do número de épocas')
    plt.show()  
    
    figure, subfig = plt.subplots(1, 1, figsize=(5, 5))
    subfig.scatter(features[:, 0], features[:, 1], c=labels)
    subfig.set_title('ground truth', fontsize=20)
    plt.show()
    
    hyperplane(X, y, w, w[0])


if __name__ == "__main__": main()


