import matplotlib.pyplot as plt 
import numpy as np
from sklearn.utils.validation import check_is_fitted
from matplotlib.colors import ListedColormap

# This is a function that plots the original dataset (X, y) and decision boundary:
def plot_linear_decision_boundary(X, y, model):
    check_is_fitted(model)
    theta = model.intercept_.tolist() + model.coef_[0].tolist()
    
    X0 = X[y==0] # subset of the non admitted students
    X1 = X[y==1] # subset of the admitted students
    
    # Plottin the dataset:
    plt.scatter(X0[:, 0], X0[:, 1], marker="o", color="red", label="Non admitted")
    plt.scatter(X1[:, 0], X1[:, 1], marker="*", color="blue", label="Admitted")
    plt.xlabel("Math score")
    plt.ylabel("English score")
    
    # Plotting the decision boundary:
    mini, maxi = X[:, 0].min(), X[:, 0].max()
    plot_x1 = np.linspace(mini, maxi)
    plot_x2 = - (theta[0] + theta[1] * plot_x1) / theta[2]
    plt.plot(plot_x1, plot_x2, color="green", label="Decision boundary")
    
    plt.title("Plot of the training data and decision boundary")
    plt.legend()
    plt.show()


# =====================================================
# This fonction plots the decision boundary and the training dataset
# You can read it if you want, but you don't need to fully understand it.
def plot_kn_decision_boundary(X, y, model, k=5):
    print("Please wait. This might take few seconds to plot ...")
    min_x1, max_x1 = min(X[:, 0]) - 0.1, max(X[:, 0]) + 0.1
    min_x2, max_x2 = min(X[:, 1]) - 0.1, max(X[:, 1]) + 0.1

    plot_x1, plot_x2 = np.meshgrid(np.linspace(min_x1, max_x1, 50), np.linspace(min_x2, max_x2, 50))
    points = np.c_[plot_x1.ravel(), plot_x2.ravel()]
    # preds = np.array([ func(x, X, y, k) for x in points ])
    preds = model.predict(points)
    preds = preds.reshape(plot_x1.shape)

    X0 = X[y==0]
    X1 = X[y==1]

    plt.pcolormesh(plot_x1, plot_x2, preds, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(X0[:, 0], X0[:, 1], color="red", label="Rejected")
    plt.scatter(X1[:, 0], X1[:, 1], color="blue", label="Accepted")
    plt.xlabel("Microship Test 1")
    plt.xlabel("Microship Test 2")
    plt.title("Decision boundary with k = {}".format(k))
    plt.legend()
    plt.show()