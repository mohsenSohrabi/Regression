# Performance of different regressors on auto_mpg dataset
In this repository, many different methods are applied on "[auto_mpg](https://archive.ics.uci.edu/ml/datasets/auto+mpg)" dataset. 
The methods used in this code are following:
* [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
* [Stochastic Gradient Descent Regression(SGD)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
* [KNN Regression](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
* [Gradient Booster Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
* [XGBoost Regression](https://xgboost.readthedocs.io/en/latest/)

## Results
According to the results, XGBoost has the best performance with minimium loss although it is not far better than gradient booster. In addition, it is obvious that KNN does a poor job on this dataset, and linear and SGD are very similar in their performance(linear is slightly better). 
![Results](https://github.com/mohsenSohrabi/Regression/blob/main/Results.png)
