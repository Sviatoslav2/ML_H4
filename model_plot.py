from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
        
############################################################################################################################################ !
############################################################################################################################################ !
############################################################################################################################################ !      
class Classification_metrics:
    def __init__(self, y_true, y_pred):
        self.__y_true = y_true
        self.__y_pred = y_pred
        self.f1 = f1_score(self.__y_true, self.__y_pred, average='macro')
    def is_binary(self):
        return len(list(set(list(self.__y_true)))) == len(list(set(list(self.__y_pred))))== 2
    
    def get_accuracy_score(self): 
        return accuracy_score(self.__y_true, self.__y_pred)
        
    def get_roc(self):
        if self.is_binary():
            return roc_auc_score(self.__y_true, self.__y_pred)
        
    def get_auc(self):
        if self.is_binary():
            fpr, tpr, thresholds = metrics.roc_curve(self.__y_true, self.__y_pred, pos_label=2) 
            return metrics.auc(fpr, tpr)
##################################################################################### !
##################################################################################### !
##################################################################################### !

class Regresion_metrics:
    def __init__(self, y_true, y_pred):
        self.__y_true = y_true
        self.__y_pred = y_pred
        
    def get_root_mean_sq(self):
        return np.sqrt(metrics.mean_squared_error(self.__y_true, self.__y_pred))
        
    def get_mean_squared_error(self):
        return metrics.mean_squared_error(self.__y_true, self.__y_pred)
        
    def get_mean_absolute_error(self):
        return mean_absolute_error(self.__y_true, self.__y_pred)
        
    def get_r2_score(self):
        return r2_score(self.__y_true, self.__y_pred)
        
##################################################################################### !
##################################################################################### !
##################################################################################### !


def modelfit_for_classification(alg, dtrain, dtest,target_train, target_test, verbose=False):
    '''
    Classification 
    '''
    alg.fit(dtrain, target_train)
    dtrain_predictions = alg.predict(dtrain)
    
    y_pred = alg.predict(dtest)
    y_true = target_test
    
    y_pred_train = alg.predict(dtrain)
    y_true_train = target_train
    
    train_met = Classification_metrics(y_true_train, y_pred_train)
    test_met = Classification_metrics(y_true, y_pred)
    
    if verbose:
        print("Model report ############################################### !")
        print('Accuracy on train == ', train_met.get_accuracy_score())
        print('f1_score_train == ', f1_score(y_true_train, y_pred_train, average='macro'))
        print('ROC on train == ', train_met.get_roc())
        print('AUC on train == ',train_met.get_auc())
        print("############################################################ !")
        print('Accuracy on test == ', train_met.get_accuracy_score())
        print('f1_score on test == ', f1_score(y_true, y_pred, average='macro'))
        print('ROC on test == ', train_met.get_roc())
        print('AUC on test == ',train_met.get_auc())
        print("End of model report ######################################## !")
    return alg, [train_met.get_accuracy_score(), train_met.f1, train_met.get_roc(), train_met.get_auc()]
    
    
def modelfit_Regresion(alg, dtrain, dtest,target_train, target_test, verbose=False):
    alg.fit(dtrain, target_train)
    dtrain_predictions = alg.predict(dtrain)
    
    y_pred = alg.predict(dtest)
    y_true = target_test
    
    y_pred_train = alg.predict(dtrain)
    y_true_train = target_train
    
    train_met = Regresion_metrics(y_true_train, y_pred_train)
    test_met = Regresion_metrics(y_true_train, y_pred_train)
    
    if verbose:
        print("Model report ############################################### !")
        print("Root mean sq error on train: ",train_met.get_root_mean_sq())
        print("Mean squared error on train: ", train_met.get_mean_squared_error())
        print("Mean absolute error on train: ", train_met.get_mean_absolute_error())
        print("r2 score on train: ", train_met.get_r2_score())
        print("############################################################ !")
        print("Root mean sq error on test: ",test_met.get_root_mean_sq())
        print("Mean squared error on test: ", test_met.get_mean_squared_error())
        print("Mean absolute error on test: ", test_met.get_mean_absolute_error())
        print("r2 score on test: ", test_met.get_r2_score())
        print("End of model report ######################################## !")   
        
    return alg, [test_met.get_root_mean_sq(), test_met.get_mean_squared_error(), test_met.get_mean_absolute_error(), test_met.get_r2_score()]   
################################################################### !
################################################################### !
################################################################### !    

class Modelfit:
    
    def __init__(self,alg, dtrain, dtest, target_train, target_test, key="R"):
        if key not in ["r","c","C","R"]:
            raise ValueError(key)
        elif key in ['r','R']:
            self.__alg, self.__lst = modelfit_Regresion(alg, dtrain, dtest,target_train, target_test, verbose=False)
        else:
            self.__alg, self.__lst = modelfit_for_classification(alg, dtrain, dtest,target_train, target_test, verbose=False)
            
    def get_metrix(self):
        return self.__lst
    
    def get_alg(self):
        return self.__alg
    
class Plot_metrics_of_models:
    def __init__(self, lst, dtrain, dtest, target_train, target_test, key="R"):
        self.lst_of_models = [Modelfit(i,dtrain, dtest, target_train, target_test, key).get_metrix() for i in lst]
        self.is_regresion = key in ['r','R']
        
    def __make_data_frame(self):
        dct = {}
        for i in range(len(self.lst_of_models)):
            dct[i] = self.lst_of_models[i]
        return pd.DataFrame(data=dct)
        
    def plot_metrics(self):
        data = self.__make_data_frame()
        
        if self.is_regresion:
            plt.subplot(4, 1, 1)
            plt.plot(range(data.shape[1]), data.loc[0], linestyle='solid')
            print("Root mean sq error on test")
            plt.subplot(4, 1, 2)
            plt.plot(range(data.shape[1]), data.loc[1], linestyle='solid')
            print("Mean squared error on test")
            plt.subplot(4, 1, 3)
            print("Mean absolute error on test")
            plt.plot(range(data.shape[1]), data.loc[2], linestyle='solid')
            plt.subplot(4, 1, 4)
            print("r2 score on test")
            plt.plot(range(data.shape[1]), data.loc[3], linestyle='solid')
        else:
            print("Accuracy on train")
            print("f1_score on test")
            print("ROC on test")
            print("AUC on test")
            
            try:
                plt.subplot(4, 1, 1)
                plt.plot(range(data.shape[1]), data.loc[0], linestyle='solid')
                plt.subplot(4, 1, 2)
                plt.plot(range(data.shape[1]), data.loc[1], linestyle='solid')
                plt.subplot(4, 1, 3)
                plt.plot(range(data.shape[1]), data.loc[2], linestyle='solid')   
                plt.subplot(4, 1, 4)
                plt.plot(range(data.shape[1]), data.loc[3], linestyle='solid')
            except Exception:
                plt.subplot(3, 1, 1)
                plt.plot(range(data.shape[1]), data.loc[0], linestyle='solid')
                plt.subplot(3, 1, 2)
                plt.plot(range(data.shape[1]), data.loc[1], linestyle='solid')
                plt.subplot(3, 1, 2)
                plt.plot(range(data.shape[1]), data.loc[2], linestyle='solid')