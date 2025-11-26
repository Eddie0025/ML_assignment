#Q1. Implement Gaussian Naïve Bayes Classifier on the Iris dataset
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#1. Step-by-Step Gaussian Naïve Bayes Implementation
class GaussianNB_Manual:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c]  = X_c.var(axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def gaussian_prob(self, class_idx, x):
        mean = self.mean[class_idx]
        var  = self.var[class_idx]
        numerator = np.exp(-(x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        preds = []

        for x in X:
            posteriors = []

            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(np.log(self.gaussian_prob(c, x)))
                posterior = prior + likelihood
                posteriors.append(posterior)

            preds.append(np.argmax(posteriors))

        return np.array(preds)
gnb_manual = GaussianNB_Manual()
gnb_manual.fit(X_train, y_train)

y_pred_manual = gnb_manual.predict(X_test)
acc_manual = accuracy_score(y_test, y_pred_manual)

print("Manual Gaussian Naive Bayes Accuracy:", acc_manual)

#2. In-built Gaussian Naïve Bayes (sklearn)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred_builtin = gnb.predict(X_test)
acc_builtin = accuracy_score(y_test, y_pred_builtin)

print("Built-in GaussianNB Accuracy:", acc_builtin)

#Q2. Explore about GridSearchCV toot in scikit-learn. This is a tool that is often used for tuning hyperparameters of machine learning models. Use this tool to find the best value of K for K-NN Classifier using any dataset
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (very important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define model
knn = KNeighborsClassifier()

# Define parameter grid for K
param_grid = {
    'n_neighbors': np.arange(1, 31)   # test K = 1 to 30
}

# Apply GridSearchCV
grid = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,               # 5-fold cross-validation
    scoring='accuracy'
)

grid.fit(X_train, y_train)

# Best K value
print("Best K found by GridSearchCV:", grid.best_params_['n_neighbors'])
print("Best cross-validation accuracy:", grid.best_score_)

# Test accuracy with best K
best_knn = grid.best_estimator_
y_pred = best_knn.predict(X_test)

print("Test Set Accuracy:", accuracy_score(y_test, y_pred))

# --------------------------------------------------
# Name: Adityavir Singh Randhawa
# Roll No: 102483009
# Sub Group: 3C53
# --------------------------------------------------