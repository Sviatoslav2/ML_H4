from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def modelfitAccurasy(alg, dtrain, dtest,target_train, target_test):
    '''
    Accurasy and model
    '''
    alg.fit(dtrain, target_train)
    dtrain_predictions = alg.predict(dtrain)
    
    y_pred = alg.predict(dtest)
    y_true = target_test
    
    y_pred_train = alg.predict(dtrain)
    y_true_train = target_train
    
    #print(y_true_train, y_pred_train)
    #print("Model report ###############################################!")
    #print('Accuracy on train == ',accuracy_score(y_true_train, y_pred_train))
    #print('f1_score == ', f1_score(y_true_train, y_pred_train, average='macro'))
    #print('Accuracy on test == ',accuracy_score(y_true, y_pred))
    #print('f1_score == ', f1_score(y_true, y_pred, average='macro'))
    #print("End of model report ########################################!")   
    return alg , [accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')]


def modelfitRegresion(alg, dtrain, dtest,target_train, target_test):
    
    alg.fit(dtrain, target_train)
    dtrain_predictions = alg.predict(dtrain)
    
    y_pred = alg.predict(dtest)
    y_true = target_test
    
    y_pred_train = alg.predict(dtrain)
    y_true_train = target_train
    
    #print("Model report ###############################################!")
    #print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error((target_train).values, dtrain_predictions)))        
    #print("End of model report ########################################!")
    
    return alg , [np.sqrt(metrics.mean_squared_error((target_train).values, dtrain_predictions))] 

def ploting_data(data):
    plt.figure(figsize=(50,10))
    plt.plot(range(len(data)), data[0], linestyle='solid',label='accuracy')# red
    plt.plot(range(len(data)), data[1], linestyle='solid',label='f1_score') # bl < 400     
    
def lst_metrix_plot(lst_models, dtrain, dtest,target_train, target_test, model_train_func=modelfitAccurasy):


    lst_res = []
    for model in lst_models:
        alg, lst = model_train_func(model, dtrain, dtest,target_train, target_test)
        lst_res.append(lst)
    data = pd.DataFrame(lst_res)    
    if model_train_func == modelfitAccurasy:
        ploting_data(data)
    else:
        plt.plot(range(len(data)), data[0], linestyle='solid')