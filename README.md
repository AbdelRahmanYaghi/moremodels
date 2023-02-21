# MoreModels

A python library allowing you to use multiple models using the weight of each model based on their performance

install using 
```
pip install moremodels
```

Example code for WeightedModels Object:

```
from moremodels import WeightedModels

model1 = catboost.CatBoostRegressor()
model2 = RandomForestRegressor()
model3 = xgboost.XGBRegressor()

my_data = pd.read('my_data.csv')
test = pd.read('test.csv)

my_models = [model1, model2, model3]
models = WeightedModels( models = my_models, trainSplit = 0.8, randomState = 696969 )

models.fit(my_data, 'self') # 'self' here means that the validation dataset will be used from the internal split in the class 

print(models.modelWeights)

myPredictedData = models.predict(test)

print(models.models[0])

```


Example code for UniqueWeightedModels Object:

```
from moremodels import UniqueWeightedModels
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error

model1 = catboost.CatBoostRegressor()
model2 = RandomForestRegressor()
model3 = xgboost.XGBRegressor()

# Assume that you have applied 3 different feature engineering methods on the same dataset and ended up with:

X1, y = load_iris(return_X_y=True)

X2, y = load_iris(return_X_y=True)

X3, y = load_iris(return_X_y=True)

# Note: It is assumed that y would always be the same, since it's the target, so, applying operation on the target is not reccomended.

my_models = [model1, model2, model3]

models = WeightedModels( models = my_models, trainSplit = [0.8, 0.6, 0.75], randomState = 696969, error = mean_squared_error) 
# In the line above, since only one random state was passed, then it's assumed for all models. Same goes for error.
# Meaning that, you could pass [mean_squared_error, mean_squared_error, mean_squared_error] and it would be the same output.

models.fit([X1, X2, X3], y, 'self') 
# 'self' here means that the validation dataset will be used from the internal split in the class, and since its the only input, then its ['self', 'self', 'self'] 
# You could also pass more than one validation dataset, such that [[val_x1, val_y1], 'self', [val_x3, val_y3]]
# It is assumed that y will always be the same. Maybe, I'll change that in later updates.

print(models.modelWeights)

myPredictedData = models.predict([X_test1, X_test2, X_test3])

```