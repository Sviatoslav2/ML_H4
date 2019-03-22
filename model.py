import numpy as np


class MyLinearModel(object):
    def __init__(self, lr=0.005, lmbda=1.0, regularization="l2", 
                 convergence_threshold=.1e-10, max_iters=500, verbose=True):
        self.lr = lr
        self.lmbda = lmbda
        self.regularization = regularization
        
        self.convergence_threshold = convergence_threshold
        self.max_iters = max_iters
        self.verbose = verbose
        
        self.weights = None
    
    def fit(self, X, y):
        X = self._preprocess(X)
        
        self.weights = np.full(X.shape[1], 0.1)
        losses = []
        
        for i in range(self.max_iters):
            prediction = self._predict(X)
            
            loss = self._loss(prediction, y)
            if self.regularization == "l1":
                loss += self.lmbda * np.sum(np.abs(self.weights))
            elif self.regularization == "l2":
                loss += self.lmbda * np.sum(self.weights ** 2)
            losses.append(loss)
            if len(losses) > 2 and np.abs(losses[-1] - losses[-2]) <= self.convergence_threshold:
                break
            
            grads = self._grad(X, y, prediction)
            if self.regularization == "l1":
                grads += self.lmbda * self.weights / np.abs(self.weights) / len(y)
            elif self.regularization == "l2":
                grads += self.lmbda * self.weights / len(y)
            self.weights -= self.lr * grads
            
            if self.verbose:
                print("Epoch: {0:3} Loss: {1}".format(i + 1, loss))
    
    def _preprocess(self, X):
        return np.c_[X, np.ones(len(X))]

#################################################### ! 
#################################################### ! 
#################################################### ! 


class LinearRegression(MyLinearModel):
    def predict(self, X):
        return self._predict(self._preprocess(X))
    
    def score(self, X, y):
        prediction = self.predict(X)
        return 1 - np.sum((y - prediction) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    def _predict(self, X):
        return np.dot(X, self.weights)
    
    def _loss(self, prediction, y):
    
        #print(prediction)
        #print('####################################')
        #print(y)
    
        return np.mean((prediction - y) ** 2)
    
    def _grad(self, X, y, prediction):
        return X.T.dot(prediction - y) / len(y)


class LogisticRegression(MyLinearModel):
    def predict_proba(self, X):
        return self._predict(self._preprocess(X))
    
    def predict(self, X):
        return self.predict_proba(X) > 0.5
    
    def score(self, X, y):
        prediction = self.predict(X)
        return (prediction == y).sum() / len(y)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _predict(self, X):
        return self._sigmoid(np.dot(X, self.weights))
    
    def _loss(self, prediction, y):
        return -np.mean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))
    
    def _grad(self, X, y, prediction):
        return X.T.dot(prediction - y) / len(y)
        
        
def normalize_data(data):
    feature_sigmas = np.std(data, axis=0)
    feature_means = np.average(data, axis=0)
    data = (data - feature_means) / feature_sigmas
    return data
        
