from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

def svm_train_and_plot(X, y, C):
    clf = SVC(C=C, kernel="linear").fit(X, y) # Training
    theta = np.concatenate([clf.intercept_, clf.coef_[0]]) # The parameters vector theta
    
    # Plotting the dataset and linear decision boundary
    X0 = X[y==0] # data-points where the class-label is 0
    X1 = X[y==1] # data-points where the class-label is 1
    plt.scatter(X0[:, 0], X0[:, 1], marker="$0$", color="red")
    plt.scatter(X1[:, 0], X1[:, 1], marker="$1$", color="blue")
    
    plot_x1 = np.linspace(0, 4)
    plot_x2 = - (theta[0] + theta[1] * plot_x1) / theta[2]
    plt.plot(plot_x1, plot_x2, color="green")
    
    plt.title("SVM Decision Boundary with C = {}".format(C))
    plt.show()

# ==============================================================================
def nonlinear_svm_train_and_plot(X, y, C, gamma):
    print("Please wait. This might take some time (few seconds) ...")
    
    clf = SVC(C=C, kernel="rbf", gamma=gamma).fit(X, y) # Training
    
    # Plotting the dataset and nonlinear decision boundary
    X0 = X[y==0] # data-points where the class-label is 0
    X1 = X[y==1] # data-points where the class-label is 1
    plt.scatter(X0[:, 0], X0[:, 1], marker="$0$", color="red")
    plt.scatter(X1[:, 0], X1[:, 1], marker="$1$", color="blue")
    
    x1_min, x1_max = np.min(X[:, 0]), np.max(X[:, 0])
    x2_min, x2_max = np.min(X[:, 1]), np.max(X[:, 1])
    plot_x1, plot_x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.004), np.arange(x2_min, x2_max, 0.004))
    Z = clf.predict(np.c_[plot_x1.ravel(), plot_x2.ravel()])
    Z = Z.reshape(plot_x1.shape)
    
    plt.contour(plot_x1, plot_x2, Z, colors="green")
    
    plt.title("SVM Decision Boundary with C = {}, gamma = {}".format(C, gamma))
    plt.show()

# ==============================================================================
def processEmail(text):
    import re, string
    from stemming.porter2 import stem
    
    text = text.lower()                                         # Lower case
    text = re.sub("<[^<>]+>", " ", text)                        # Strip all HTML
    text = re.sub("[0-9]+", "number", text)                     # Handle numbers
    text = re.sub("(http|https)://[^\s]*", "httpaddr", text)    # Handle URLS
    text = re.sub("[^\s]+@[^\s]+", "emailaddr", text)           # Handle email addresses
    text = re.sub("[$]+", "dollar", text)                       # Handle $ sign
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove any punctuation
    text = re.sub("\s+", " ", text).strip()                     # Replace multiple white spaces with one space
    text = re.sub("[^a-zA-Z0-9 ]", "", text)                    # Remove any other non-alphanumeric characters
    text = " ".join([ stem(word) for word in text.split(" ") ]) # Stemming all words
    text = " ".join([ word for word in text.split(" ") if len(word) > 1 ]) # Removing too short words
    return text

# ==============================================================================

