#Q!. Generate a dataset with atleast seven highly correlated columns and a target variable.
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. Generate dataset
np.random.seed(42)
n = 1000
base = np.random.randn(n)

X = np.column_stack([
    base + np.random.normal(0, 0.1, n),
    base*2 + np.random.normal(0, 0.1, n),
    base*3 + np.random.normal(0, 0.1, n),
    base + np.random.normal(0, 0.2, n),
    base*1.5 + np.random.normal(0, 0.1, n),
    base*0.5 + np.random.normal(0, 0.1, n),
    base*2.5 + np.random.normal(0, 0.1, n),
])

true_w = np.array([3, 2, 1.5, 4, 2.5, 1, 3.5])
y = X @ true_w + np.random.normal(0, 0.1, n)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Ridge Regression (Gradient Descent)
def ridge_gd(X, y, lr, lam, iters=500):
    m, n = X.shape
    w = np.zeros(n)
    for _ in range(iters):
        y_pred = X @ w
        error = y_pred - y
        grad = (1/m) * (X.T @ error + lam * w)
        w -= lr * grad
    final_cost = (1/(2*m)) * (np.sum(error**2) + lam*np.sum(w**2))
    return w, final_cost

# 3. Hyperparameter Search
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1, 10]
lambdas = [1e-15, 1e-10, 1e-5, 1e-3, 0, 1, 10, 20]

best_r2 = -999
best_result = None

for lr in learning_rates:
    for lam in lambdas:
        w, cost = ridge_gd(X_train, y_train, lr, lam)
        y_pred = X_test @ w
        r2 = r2_score(y_test, y_pred)

        if r2 > best_r2:
            best_r2 = r2
            best_result = {
                "learning_rate": lr,
                "lambda": lam,
                "cost": cost,
                "r2": r2,
                "weights": w
            }

# 4. Print Best Parameters
print("\nBEST PARAMETERS FOUND:")
print("Learning Rate:", best_result["learning_rate"])
print("Lambda:", best_result["lambda"])
print("Final Cost:", best_result["cost"])
print("R2 Score:", best_result["r2"])
print("Weights:", best_result["weights"])

#Q2. Load the Hitters dataset from the following link 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load the data from CSV (or local file)
df = pd.read_csv("Hitters.csv")

# 2. Pre-process the data
# (a) Drop rows where target Salary is missing
df = df.dropna(subset=['Salary'])

# (b) Drop rows with any other missing values (if present)
df = df.dropna()

# (c) Convert categorical variables to numeric (one-hot / dummy encoding)
# According to dataset description, columns like League, Division, NewLeague are categorical
cat_cols = ['League', 'Division', 'NewLeague']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 3. Split into input (X) and output (y), then train/test split
X = df.drop('Salary', axis=1).values
y = df['Salary'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Feature scaling (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Fit models — Linear, Ridge, Lasso
models = {
    'Linear': LinearRegression(),
    'Ridge (λ=0.5748)': Ridge(alpha=0.5748),
    'Lasso (λ=0.5748)': Lasso(alpha=0.5748, max_iter=10000)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    results[name] = {
        'r2': r2,
        'mse': mse,
        'coefficients': model.coef_,
        'intercept': model.intercept_
    }

# 6. Print evaluation results
for name, res in results.items():
    print(f"Model: {name}")
    print("  R² on test set: {:.4f}".format(res['r2']))
    print("  MSE on test set: {:.4f}".format(res['mse']))
    print("  Intercept: {:.4f}".format(res['intercept']))
    print("  Number of non-zero coefficients: {}".format(
        np.sum(res['coefficients'] != 0)
    ))
    print("-" * 40)

#Q3. Cross Validation for Ridge and Lasso Regression 
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load Dataset
boston = load_boston()
X = boston.data
y = boston.target

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. RidgeCV
ridge_alphas = [0.01, 0.1, 1, 10, 100, 200]

ridge_cv = RidgeCV(alphas=ridge_alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)

ridge_pred = ridge_cv.predict(X_test_scaled)

ridge_r2 = r2_score(y_test, ridge_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)

# 5. LassoCV
lasso_cv = LassoCV(alphas=None, cv=5, max_iter=5000)
lasso_cv.fit(X_train_scaled, y_train)

lasso_pred = lasso_cv.predict(X_test_scaled)

lasso_r2 = r2_score(y_test, lasso_pred)
lasso_mse = mean_squared_error(y_test, lasso_pred)

# 6. Print Results
print("======= Ridge Regression with Cross Validation =======")
print("Best alpha (λ):", ridge_cv.alpha_)
print("R² score:", ridge_r2)
print("MSE:", ridge_mse)
print()

print("======= Lasso Regression with Cross Validation =======")
print("Best alpha (λ):", lasso_cv.alpha_)
print("R² score:", lasso_r2)
print("MSE:", lasso_mse)
print("Number of selected features:", np.sum(lasso_cv.coef_ != 0))

#Q4. Multiclass Logistic Regression: Implement Multiclass Logistic Regression (step-by step) on Iris dataset using one vs. rest strategy? 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load Dataset
iris = load_iris()
X = iris.data
y = iris.target
classes = np.unique(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Helper Functions

# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary Logistic Regression training (for one-vs-rest)
def train_binary_logistic(X, y, lr=0.1, iters=2000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0

    for _ in range(iters):
        z = X @ w + b
        y_pred = sigmoid(z)
        
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)

        w -= lr * dw
        b -= lr * db

    return w, b

# Predict probability using trained weights
def predict_binary(X, w, b):
    return sigmoid(X @ w + b)

# 3. ONE-VS-REST MULTICLASS TRAINING
models = {}  # store w, b for each class

for c in classes:
    # create binary labels: class c = 1, others = 0
    y_binary = (y_train == c).astype(int)
    
    w, b = train_binary_logistic(X_train, y_binary, lr=0.1, iters=3000)
    models[c] = (w, b)

# 4. MULTICLASS PREDICTION
def predict_multiclass(X):
    probs = []
    for c in classes:
        w, b = models[c]
        prob = predict_binary(X, w, b)
        probs.append(prob)
    return np.argmax(np.array(probs).T, axis=1)

y_pred = predict_multiclass(X_test)

# 5. Accuracy
acc = accuracy_score(y_test, y_pred)

print("Predicted labels:", y_pred)
print("Actual labels:   ", y_test)
print("\nMulticlass Logistic Regression Accuracy:", acc)

# --------------------------------------------------
# Name: Adityavir Singh Randhawa
# Roll No: 102483009
# Sub Group: 3C53
# --------------------------------------------------