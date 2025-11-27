#Q1 — SVM on Iris Dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 80:20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#b) Train three SVM models → Linear, Polynomial, RBF
models = {
    "Linear SVM": SVC(kernel="linear"),
    "Polynomial SVM": SVC(kernel="poly", degree=3),
    "RBF SVM": SVC(kernel="rbf")
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

#c) Evaluation Metrics (Accuracy, Precision, Recall, F1)
for name, model in trained_models.items():
    y_pred = model.predict(X_test)

    print(f"\n===== {name} =====")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall   :", recall_score(y_test, y_pred, average='macro'))
    print("F1-Score :", f1_score(y_test, y_pred, average='macro'))

#d) Confusion Matrix for Each Kernel
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

#Q2 — Breast Cancer Dataset
#A) Load Breast Cancer Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#b. With Scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

svm_scaled = SVC(kernel="rbf")
svm_scaled.fit(X_train_scaled, y_train)

train_acc_s = svm_scaled.score(X_train_scaled, y_train)
test_acc_s  = svm_scaled.score(X_test_scaled, y_test)

print("\nSVM with scaling:")
print("Train Accuracy:", train_acc_s)
print("Test Accuracy :", test_acc_s)

