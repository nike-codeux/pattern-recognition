import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
%matplotlib inline

class Adaline(object):
    def __init__(self,input_size, theta=0.0, lrate=0.01):
        self.theta = theta
        self.lrate = lrate
        self.weight = np.random.randn(input_size+1) # +1 pelo bias
        
    def signal(self, x):
        return x
    
        
    def quantize_vector(self, y_hat):
     
        return [1 if y >= self.theta else 0 for y in y_hat]

    # training function
    def fit(self, x_input, y, epochs = 100, treshold = 0.001):
        
        bias = np.ones((x_input.shape[0], 1), dtype=float)
        x_input = np.concatenate((bias, x_input), axis = 1)
        t = 0
        cost = []
        n = x_input.shape[0]
        s_error=1e24
        while s_error >= 2* treshold and epochs>0:
            s_error=0
            for idx in range(x_input.shape[0]):
                y_hat = self.signal(np.dot(x_input[idx],self.weight))
                error = (y[idx] -y_hat)
                self.weight += self.lrate * error * x_input[idx]

            s_error += np.sum(error**2)
            
            
            cost.append(s_error/2.0)
            epochs-=1
            
            
                
        return cost
        
    def predict(self, x_input):
        bias = np.ones((x_input.shape[0], 1), dtype=float)
        x_input = np.concatenate((bias, x_input), axis = 1)
        return self.quantize_vector(np.dot(x_input, self.weight))
    
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
model = Adaline(input_size = 2, lrate=0.01)
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
