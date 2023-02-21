from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from inspect import isfunction
import pandas as pd

class WeightedModels:
    def __init__(self, models, trainSplit = 0.85, randomState = 42, error = None):
        '''
        Takes a list of models, and then gives each model a weight based on their performance.

        ``` models ``` Receives a list of models.
        ``` trainsplit ``` Determains the percentage of the data you want to train the model on.
        ``` randomState ``` Sets a random state you wish, such that it splits the data based on given number
        ``` error ``` Pass an error function, to set how you would want the model to predict
        
        '''
        if type(models) is list:
            self.__models = models
        else:
            raise TypeError("Initialization expected a list of models, instead got " + str(type(models)))

        if error is None:
            self.__error = mean_squared_error
        elif isfunction(error):
            self.__error = error
        else:
            raise TypeError("Initialization expected an error function, instead got " + str(type(models)))

        if trainSplit >= 1:
            raise Exception("trainSplit expected to be less than 1")

        self.trainSplit = trainSplit
        self.randomState = randomState
        self.__weights = []

    @property
    def models(self):
        return self.__models

    @property
    def feature_importances_(self):
        if not self.__weights:
            raise Exception("Models are not fitted yet. Use .fit() before calling .feature_importances_")

        importances = []
        
        for model in self.__models:
            try:
                imprt = list(map(lambda x: x/sum(model.feature_importances_) ,model.feature_importances_))
                importances.append(imprt)
            except AttributeError:
                imprt = list(map(lambda x: x/sum(model.coef_) ,model.coef_))
                importances.append(imprt)

        weightedImpotances = []

        for i in range(len(importances)):
            weightedImpotances.append(list(map(lambda x: x * self.__weights[i], importances[i])))
            
        return [sum(x) for x in zip(*weightedImpotances)]

    @property
    def modelWeights(self):
        if not self.__weights:
            raise Exception("Models are not fitted yet. Use .fit() before calling .modelWeights")            
        
        return self.__weights
            
    def fit(self, X, y, val = None, showScores = True):
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(X, y, train_size=self.trainSplit, random_state=self.randomState)

        if val == 'self':
            val = [self.__X_test, self.__y_test]

        for model in self.__models:
            if (('eval_set' in model.fit.__code__.co_varnames) and (val is not None)):
                model.fit(self.__X_train, self.__y_train, eval_set = [(self.__X_train, self.__y_train), (val[0], val[1])])
            else:
                try:
                    model.fit(self.__X_train, self.__y_train)
                except Exception: # I know that this is a bad code but XGBoost is really dumb
                    model.fit(self.__X_train, self.__y_train, eval_set = [(self.__X_train, self.__y_train), (val[0], val[1])])
        
        self.__calculateWeights()

        if showScores:
            self.__showScores()  
        
    def __calculateWeights(self):
        preds = []
        acc = []
        for model in self.__models:
            preds.append(model.predict(self.__X_test))

        for pred in preds:
            acc.append(self.__error(self.__y_test, pred))
            
        tempo = list(map(lambda x: (sum(acc)-x),acc))
        self.__weights = list(map(lambda x: x/sum(tempo) ,tempo))

    def predict(self, X):
        if not self.__weights:
            raise Exception("Models are not fitted yet. Use .fit() before calling .predict()")
        
        preds = []
        for model in self.__models:
            preds.append(model.predict(X))
        
        for i in range(len(preds)):
            preds[i] = (list(map(lambda x: x * self.__weights[i], preds[i])))
            
        return [sum(x) for x in zip(*preds)]

    def setModelWeights(self, weights, showScores = True ,Silent = False):
        if not Silent:
            print('Warning: This function should be run after .fit(), since .fit() overwrites your new weights.')

        if sum(weights) <= 0:
            raise Exception("The sum of your weights cannot be below or equal to 0.")

        if len(weights) != len(self.__models):
            raise Exception("Expected weights inputed ("+ str(len(weights)) +") to be the same length as the number of models (" + str(len(self.__models)) +").")

        self.__weights = list(map(lambda x: x/sum(weights) ,weights))
        self.__showScores()

    def __showScores(self):
        preds = []
        for model in self.__models:
            preds.append(model.predict(self.__X_test))

        for i in range(len(preds)):
            preds[i] = (list(map(lambda x: x * self.__weights[i], preds[i])))
        
        print("Weighted model error is:", self.__error(self.__y_test,[sum(x) for x in zip(*preds)]))
    

class UniqueWeightedModels:
    def __init__(self, models, trainSplits = 0.85, randomStates = 42, errors = None):
        '''
        Takes a list of models, trainSplits, randomStates, and errors and then gives each model a weight based on their performance.

        ``` models ``` Receives a list of models.
        ``` trainsplits ``` Determains the percentage of the data you want to train the model on.
        ``` randomStates ``` Sets a random state you wish, such that it splits the data based on given number
        ``` errors ``` Pass an error function, to set how you would want the model to predict
        
        '''
        if type(models) is list:
            self.__models = models
        else:
            raise TypeError("Initialization expected a list of models, instead got " + str(type(models)))

        if errors is None:
            self.__errors = [mean_squared_error for i in range(len(models))]
        elif type(errors) is list:
            for error in errors:
                if not(isfunction(error)):
                    raise TypeError("Initialization expected a list of error functions, instead got " + str(type(error)))
                
                self.__checkForLen(errors)
                self.__errors = errors
        elif isfunction(errors):
            self.__errors = [errors for i in range(len(models))]
        else:
            raise TypeError("Initialization expected an error function, instead got " + str(type(errors)))
        
        if isinstance(trainSplits, list):
            self.__checkForLen(trainSplits)
            self.__trainSplits = trainSplits
        else:
            self.__trainSplits = [trainSplits for i in range(len(models))]

        for i in self.__trainSplits:
            if i >= 1:
                raise Exception("trainSplit expected to be less than 1")

        if isinstance(randomStates, list):
            self.__checkForLen(randomStates)
            self.__randomStates = randomStates
        else:
            self.__randomStates = [randomStates for i in range(len(models))]
        
        self.__weights = []

    @property
    def models(self):
        return self.__models

    @property
    def modelWeights(self):
        if not self.__weights:
            raise Exception("Models are not fitted yet. Use .fit() before calling .modelWeights")            
        
        return self.__weights
            
    def __checkForLen(self, lis):
        if len(lis) != len(self.__models):
            raise Exception("Length of passed parameter (" + str(len(lis)) + ") expected to be the same as the length of models (" + str(len(self.__models)) + ").")


    def fit(self, Xs, y, vals = 'self', showScores = True):

        if vals == 'self':
            vals = ['self' for i in range(len(self.__models))]

        if isinstance(Xs, pd.DataFrame):
            Xs = [Xs for i in range(len(self.__models))]

        self.__checkForLen(Xs)
        self.__checkForLen(vals)

        self.__Xs_train = []
        self.__Xs_test = []
        self.__ys_train = []
        self.__ys_test = []
        for X, split, state in zip(Xs, self.__trainSplits, self.__randomStates):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=state)
            self.__Xs_train.append(X_train)
            self.__Xs_test.append(X_test)
            self.__ys_train.append(y_train)
            self.__ys_test.append(y_test)

        self.__vals = []
        for val, X, y in zip(vals, self.__Xs_test, self.__ys_test):
            if val == 'self':
                self.__vals.append([X, y])
            else:
                self.__vals.append([val[0], val[1]])

        for model, X_train, y_train, val in zip(self.__models, self.__Xs_train, self.__ys_train, self.__vals):
            if ('eval_set' in model.fit.__code__.co_varnames):
                model.fit(X_train, y_train, eval_set = [(X_train, y_train), (val[0], val[1])])
            else:
                try:
                    model.fit(X_train, y_train)
                except Exception: # I know that this is a bad code but XGBoost is really dumb
                    model.fit(X_train, y_train, eval_set = [(X_train, y_train), (val[0], val[1])])

        self.__calculateWeights()

        if showScores:
            self.__showScores()  

    def __calculateWeights(self):
        preds = []
        acc = []

        for model, X_test in zip(self.__models, self.__Xs_test):
            preds.append(model.predict(X_test))

        for pred, error, y_test in zip(preds, self.__errors, self.__ys_test):
            acc.append(error(y_test, pred))
            
        tempo = list(map(lambda x: (sum(acc)-x),acc))
        self.__weights = list(map(lambda x: x/sum(tempo) ,tempo))

    def predict(self, Xs):
        if not self.__weights:
            raise Exception("Models are not fitted yet. Use .fit() before calling .predict()")
        
        self.__checkForLen(Xs)

        preds = []
        for model, X in zip(self.__models, Xs):
            preds.append(model.predict(X))
        
        for i in range(len(preds)):
            preds[i] = (list(map(lambda x: x * self.__weights[i], preds[i])))
            
        return [sum(x) for x in zip(*preds)]

    def setModelWeights(self, weights, showScores = True ,Silent = False):
        if not Silent:
            print('Warning: This function should be run after .fit(), since .fit() overwrites your new weights.')

        if sum(weights) <= 0:
            raise Exception("The sum of your weights cannot be below or equal to 0.")

        if len(weights) != len(self.__models):
            raise Exception("Expected weights inputed ("+ str(len(weights)) +") to be the same length as the number of models (" + str(len(self.__models)) +").")

        self.__weights = list(map(lambda x: x/sum(weights) ,weights))
        self.__showScores()

    def __showScores(self):
        preds = []
        for model, X_test in zip(self.__models, self.__Xs_test):
            preds.append(model.predict(X_test))

        for i in range(len(preds)):
            preds[i] = (list(map(lambda x: x * self.__weights[i], preds[i])))
        
        print("Weighted model error is:", self.__errors[0](self.__ys_test[0],[sum(x) for x in zip(*preds)]))
    