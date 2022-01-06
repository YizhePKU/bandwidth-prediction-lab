import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

np.random.seed(0)

with open("data/processed.pickle", "rb") as file:
    datasets = pickle.load(file)


def linear_regression_avg_bandwidth(train_ds, test_ds, n=10):
    """Use the average value of the past n bandwidth datapoint as the only feature.

    Discard the first n datapoints."""

    def make_features(ds):
        xs = []
        ys = []
        for bw, rsrp in ds:
            xs.append(
                np.array(
                    [sum(bw[i - n : i]) / n for i in range(n, len(bw))],
                    dtype=np.float32,
                )
            )
            ys.append(np.array(bw[n:], dtype=np.float32))
        return np.concatenate(xs).reshape(-1, 1), np.concatenate(ys)

    train_x, train_y = make_features(train_ds)
    test_x, test_y = make_features(test_ds)

    model = LinearRegression()
    model.fit(train_x, train_y)

    test_y_pred = model.predict(test_x)

    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(test_y, test_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(test_y, test_y_pred))

    # Plot outputs
    plt.scatter(test_x, test_y, color="black", s=1)
    plt.plot(test_x, test_y_pred, color="blue", linewidth=2)
    plt.xticks(())
    plt.yticks(())
    plt.show()


ds = datasets["LTE"]
train_ds, test_ds = train_test_split(ds)

linear_regression_avg_bandwidth(train_ds, test_ds)
