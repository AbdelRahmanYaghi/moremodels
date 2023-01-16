# MoreModels

A python library allowing you to use multiple models using the weight of each model based on their performance

install using 
```
pip install moremodels
```

Example code:

```
from moremodels import WeightedModel

model1 = catboost.CatBoostRegressor()
model2 = RandomForestRegressor()
model3 = xgboost.XGBRegressor()

models = [model1, model2, model3]

X = pd.read('X.csv')
y = pd.read('y.csv')

Xtest = pd.read('Xtest.csv)

model = WeightedModel(models, trainSplit = 0.85, randomState = 42, error = None)
# error parameter is used to pass on a function to calculate the error by. Which effects the weights of the models

model.fit(X, y, showScores = True)
# showScore allows the model to print the weighted model's score after fitting using the training and getting the weights

model.getModelWeights()

model.predict(Xtest)

weights = [100, 40, 260]

setModelWeights(weights, showScores = True, Silent = False)
# Allows you to set you own weights, in case you want to do some experiments.
# showScore allows the model to repredict and calculate the new predection based on your new weights. then proceeds by printing the new weighted model's score.  
```
