
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb


def linear_regression( X_train, X_test, y_train, y_test):

    # scale values using pipeline to feed for linear regression
    linear_reg = make_pipeline(StandardScaler(), LinearRegression())
    # do linear regression
    linear_reg.fit(X_train, y_train)

    pred = linear_reg.predict(X_test)
    loss = mean_squared_error(pred,y_test)

    return loss


def stochastic_gradient_regression(X_train, X_test, y_train, y_test, max_iter=3000, tol=1e-4):
    sgd_reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=max_iter, tol=tol))
    sgd_reg.fit(X_train,y_train)

    pred = sgd_reg.predict(X_test)
    loss = mean_squared_error(pred,y_test)

    return loss


def knn_regression(X_train, X_test, y_train, y_test, n_neighbors=10):

    knn_reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_reg.fit(X_train,y_train)
    pred = knn_reg.predict(X_test)
    loss = mean_squared_error(pred,y_test)

    return loss


def gradient_booster_regression(X_train, X_test, y_train, y_test):

    gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                    random_state=0, loss='ls').fit(X_train, y_train)
    pred = gb_reg.predict(X_test)
    loss = mean_squared_error(pred, y_test)

    return loss


def xgb_regression(X_train, X_test, y_train, y_test):
    xg_reg = xgb.XGBRegressor(colsample_bytree=0.3, learning_rate=0.1,
                              max_depth=5, alpha=10, n_estimators=100)
    xg_reg.fit(X_train, y_train)
    pred = xg_reg.predict(X_test)
    loss = mean_squared_error(pred, y_test)

    return loss
