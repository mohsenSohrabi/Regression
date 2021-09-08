from dataProcessing import *
from regression import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main(name):
    # read data from data folder
    path = 'data/auto-mpg.data'
    header = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'model year', 'origin', 'car name']

    auto_df = read_data(path, header)

    auto_df = clean_data(auto_df)
    # print(auto_df['horsepower'])
    X, Y = get_X_Y_from_dataframe(auto_df)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=50)

    loss_dict = {'linear': linear_regression(X_train, X_test, y_train, y_test),
                 'SGD': stochastic_gradient_regression(X_train, X_test, y_train, y_test),
                 'KNN': knn_regression(X_train, X_test, y_train, y_test),
                 'grad_booster': gradient_booster_regression(X_train, X_test, y_train, y_test),
                 'XGB': xgb_regression(X_train, X_test, y_train, y_test)
                 }

    # Create a bar chart for results

    fig, ax = plt.subplots()
    hbars = ax.barh(np.arange(len(loss_dict.keys())), tuple(loss_dict.values()), align='center', color='indigo')
    ax.set_yticks(np.arange(len(loss_dict.keys())))
    ax.set_yticklabels(tuple(loss_dict.keys()))
    ax.invert_yaxis()
    ax.set_xlabel('Loss')
    ax.set_title('Loss of different regression algorithms on auto_mpg dataset')

    ax.bar_label(hbars, fmt='%.2f')
    ax.set_xlim(right=20)

    plt.show()


    # plt.figure(figsize=(5, 4))
    # plt.barh(loss_dict.values(), loss_dict.keys())
    # plt.xticks(np.arange(len(loss_dict.keys())), tuple(loss_dict.keys()), rotation=60)
    # plt.xlabel('Regression Methods')
    # plt.ylabel('Losses')
    # plt.show()


if __name__ == '__main__':
    main('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
