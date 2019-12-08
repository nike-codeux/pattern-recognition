import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
# %matplotlib inline


class RegresionLogistica(object):
    def __init__(self, theta=0, lrate=0.01, epochs=100):
        self.theta = theta
        self.lrate = lrate
        self.epochs = epochs
        self.weight = None
    
    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))
    
    def signal(self, x):
        return self.sigmoid(x)

    # training function
    def fit(self, X, y, lrate=None, epochs=None, epsilon=0.001):
        
        if lrate is None : lrate = self.lrate
        if epochs is None : epochs = self.epochs 

        bias = np.ones((X.shape[0], 1), dtype=float)
        X = np.concatenate((bias, X), axis = 1)
        
        # inicializando o vetor de pesos
        self.weight = np.random.random_sample(size=(X.shape[1]))
        
        E = error = 1e24
        cost = []
        t = 0
        
        while E >= epsilon :
            
            for i in range(X.shape[0]):
                # sigmoide das amostra e seus pesos
                y_linear = self.weight @ X[i]
                y_hat = self.signal(y_linear)

                # atualização dos pesos, gradiente descendete
                self.weight += lrate * (y[i] - y_hat) * X[i]
            
            E = error
            error = 0
            for i in range(X.shape[0]):
               y_linear = self.weight @ X[i]
               y_hat = self.signal(y_linear)
               # computo função costo, o erro
               error += -1*y[i] * np.log(y_hat) - (1-y[i]) * np.log(1 - y_hat)
            
            # calculando o erro de treinamento
            error /= X.shape[0]
            cost.append(error)
            t += 1
            E = abs(error - E)
   
        return cost
        
    def predict(self, X):
        bias = np.ones((X.shape[0], 1), dtype=float)
        X = np.concatenate((bias, X), axis = 1)
        z = np.dot(X, self.weight)
        return np.where(self.signal(z) >= 0.5, 1, 0)
    
    def evaluate(self, x, y_test):
        y_pred = self.predict(x)
        return np.mean(abs(y_pred - y_test))
        
def stardardize(x):
    means= x.mean(axis=0)    
    stds = x.std(axis=0)
    return (x-means)/stds

def split_dataset(x, rate=0.7):
    index = round(x.shape[0]*rate)
    return x[:index], x[index:]
    
def hyperplane(X, y, weights):
    xvalues = [np.min(X[:,0]), np.max(X[:,0])]
    bias = weights[0]
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
        
features, labels = make_blobs(n_samples=2000, n_features=2, centers=2, cluster_std=2.0, random_state=18)
print("Amostras: " , features.shape)
print("Classes: ", np.unique(labels))

X = np.array(features)
y = np.array(labels)

train_x, test_x = split_dataset(X)
train_y, test_y = split_dataset(y)
model = RegresionLogistica(lrate=0.01)
cost = model.fit(train_x, train_y,epochs=1000)

erroTrain = model.evaluate(train_x, train_y)
erroTest = model.evaluate(test_x, test_y)

print('Total de épocas necessárias para o aprendizado: ', len(cost))
print('Pesos aprendidos: ', model.weight)
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

hyperplane(X, y, model.weight)
