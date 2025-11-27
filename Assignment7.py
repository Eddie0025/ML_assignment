# Common imports for all tasks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import string
import re
import nltk
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

#Q1 — SMS Spam Collection (AdaBoost)
# Q1A: Load & preprocess SMS dataset
import warnings
warnings.filterwarnings('ignore')

# Update path if needed
spam_csv_path = "spam.csv"  # place file here, or provide path

try:
    df = pd.read_csv(spam_csv_path, encoding='latin-1')
except FileNotFoundError:
    raise FileNotFoundError("Place 'spam.csv' in notebook folder or update spam_csv_path")

# dataset specifics differ across sources; handle common cases
if {'v1','v2'}.issubset(df.columns):
    # some variants have v1 label column and v2 text column
    df = df.rename(columns={'v1':'label','v2':'text'})

# drop extra unnamed columns sometimes present
df = df.loc[:, ['label','text']]

# Convert labels: spam->1, ham->0
df['label_num'] = df['label'].map({'spam':1,'ham':0})
df = df.dropna(subset=['text','label_num']).reset_index(drop=True)

# Text preprocessing function
def preprocess_text(s):
    s = str(s).lower()
    s = re.sub(r'http\S+',' ', s)                      # remove URLs
    s = s.translate(str.maketrans('', '', string.punctuation))
    tokens = s.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

# Apply (this may take ~ few seconds)
df['text_clean'] = df['text'].apply(preprocess_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text_clean'])
y = df['label_num'].values

# Train-test split (80/20)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42, stratify=y)

# Class distribution
print("Full dataset class distribution:\n", df['label_num'].value_counts(normalize=True))
print("Train distribution:", np.bincount(y_train)/len(y_train))
print("Test distribution:", np.bincount(y_test)/len(y_test))

#Part B — Decision Stump Baseline
stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_train, y_train)
train_pred = stump.predict(X_train)
test_pred = stump.predict(X_test)

print("Stump Train accuracy:", accuracy_score(y_train, train_pred))
print("Stump Test accuracy: ", accuracy_score(y_test, test_pred))
print("Confusion matrix (test):\n", confusion_matrix(y_test, test_pred))

#Part C — Manual AdaBoost (T = 15)
# Manual AdaBoost implementation
from math import log

def manual_adaboost(X_train, y_train, X_test, y_test, T=15):
    n = X_train.shape[0]
    # initialize uniform weights
    W = np.ones(n) / n
    learners = []
    alphas = []
    errors = []

    X_train_arr = X_train  # sparse matrix accepted by sklearn
    # For printing misclassified indices we need original train indices; we have idx_train from earlier split
    # but we'll print sample indices relative to X_train (0..n-1)

    for t in range(1, T+1):
        stump = DecisionTreeClassifier(max_depth=1, random_state=42)
        # sklearn DecisionTreeClassifier supports sample_weight in fit
        stump.fit(X_train_arr, y_train, sample_weight=W)
        pred_train = stump.predict(X_train_arr)
        # misclassified mask
        mis_mask = (pred_train != y_train).astype(int)
        # weighted error
        err_t = np.sum(W * mis_mask) / np.sum(W)
        # avoid division by zero or perfect fit causing infinite alpha
        err_t = np.clip(err_t, 1e-12, 1-1e-12)
        alpha_t = 0.5 * np.log((1 - err_t) / err_t)

        # Print iteration info as requested
        mis_idx = np.where(mis_mask==1)[0]
        print(f"\nIteration {t}")
        print("Misclassified sample indices (relative to train):", mis_idx.tolist()[:200])
        if len(mis_idx)>0:
            print("Weights of misclassified samples (first 20):", W[mis_idx][:20].tolist())
        print("Weighted error (err_t):", err_t)
        print("Alpha:", alpha_t)

        # update weights: w_i <- w_i * exp(alpha * I[misclassified]*(+1) + ...),
        # standard formula: W_i *= exp(alpha_t * (1 if mis else -1))
        # Equivalent update:
        W = W * np.exp(alpha_t * (mis_mask*1 - (1-mis_mask)*1))
        # But more standard: multiply misclassified by exp(alpha), correct by exp(-alpha)
        # normalize
        W = W / np.sum(W)

        learners.append(stump)
        alphas.append(alpha_t)
        errors.append(err_t)

    # Final predictions on train and test by weighted vote
    def predict_combined(X, learners, alphas):
        # returns sign(sum alpha_t * h_t(x))
        # h_t predictions in {0,1} -> convert to {-1,1}
        agg = None
        for alpha, learner in zip(alphas, learners):
            pred = learner.predict(X)
            pred_signed = np.where(pred==1, 1, -1)
            if agg is None:
                agg = alpha * pred_signed
            else:
                agg += alpha * pred_signed
        final = np.where(agg >= 0, 1, 0)
        return final

    train_final = predict_combined(X_train_arr, learners, alphas)
    test_final = predict_combined(X_test, learners, alphas)

    print("\nManual AdaBoost Train accuracy:", accuracy_score(y_train, train_final))
    print("Manual AdaBoost Test accuracy: ", accuracy_score(y_test, test_final))
    print("Confusion matrix (test):\n", confusion_matrix(y_test, test_final))

    # Plot iteration vs weighted error & alpha
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(range(1, T+1), errors, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Weighted error')
    plt.title('Iteration vs weighted error')
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(range(1, T+1), alphas, marker='o', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Alpha')
    plt.title('Iteration vs alpha')
    plt.grid(True)
    plt.show()

    return learners, alphas, errors

learners_manual, alphas_manual, errors_manual = manual_adaboost(X_train, y_train, X_test, y_test, T=15)

#Part D — sklearn AdaBoost
# sklearn AdaBoost
clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=100,
    learning_rate=0.6,
    algorithm='SAMME.R', random_state=42
)
clf.fit(X_train, y_train)
train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)

print("sklearn AdaBoost Train accuracy:", accuracy_score(y_train, train_pred))
print("sklearn AdaBoost Test accuracy: ", accuracy_score(y_test, test_pred))
print("Confusion matrix (test):\n", confusion_matrix(y_test, test_pred))

#Q2 — UCI Heart Disease (AdaBoost experiments)
#Part A — Baseline decision stump
# Q2A: Load Heart Disease dataset (try multiple ways)
from sklearn import datasets

def load_heart():
    try:
        data = datasets.load_heart_disease()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        return X, y, feature_names
    except Exception:
        # fallback to openml retrieval (needs internet)
        try:
            from sklearn.datasets import fetch_openml
            heart = fetch_openml(name='heart-disease', as_frame=True)  # name may vary
            df_hd = heart.frame
            # Adjust: many OpenML heart datasets differ; try to find 'target' column
            if 'target' in df_hd.columns:
                y = (df_hd['target'].astype(int)).values
                X = df_hd.drop(columns=['target']).values
                feature_names = df_hd.drop(columns=['target']).columns.tolist()
                return X, y, feature_names
            # fallback: use UCI heart (Cleveland) if present with 'num' as target
            if 'num' in df_hd.columns:
                y = (df_hd['num'].astype(int).apply(lambda v: 1 if v>0 else 0)).values
                X = df_hd.drop(columns=['num']).values
                feature_names = df_hd.drop(columns=['num']).columns.tolist()
                return X, y, feature_names
        except Exception as e:
            raise RuntimeError("Unable to load heart dataset automatically. Please provide dataset or run in an environment with openml access.") from e

X_hd, y_hd, feat_names = load_heart()
print("Feature count:", X_hd.shape[1])

# Basic preprocessing:
# If categorical features exist encoded as numbers, we can keep them for tree-based models.
# Split
X_train, X_test, y_train, y_test = train_test_split(X_hd, y_hd, test_size=0.2, random_state=42, stratify=y_hd)

# Scale numeric features for stability (not strictly needed for trees)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# Decision stump baseline
stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_train_s, y_train)
y_tr_pred = stump.predict(X_train_s)
y_te_pred = stump.predict(X_test_s)
print("Stump Train acc:", accuracy_score(y_train, y_tr_pred))
print("Stump Test acc: ", accuracy_score(y_test, y_te_pred))
print("Confusion matrix (test):\n", confusion_matrix(y_test, y_te_pred))
print("Classification report (test):\n", classification_report(y_test, y_te_pred))

#Part B — Train AdaBoost grid (n_estimators × learning_rate)
# Q2B: Grid search over n_estimators and learning_rate
n_list = [5, 10, 25, 50, 100]
lr_list = [0.1, 0.5, 1.0]
results = []
for lr in lr_list:
    accs = []
    for n_est in n_list:
        clf = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=n_est,
            learning_rate=lr,
            algorithm='SAMME.R',
            random_state=42
        )
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        results.append({'n_estimators': n_est, 'learning_rate': lr, 'accuracy': acc, 'model': clf})
        accs.append(acc)
    # plot n_estimators vs accuracy for this lr
    plt.plot(n_list, accs, marker='o', label=f'lr={lr}')
plt.xlabel('n_estimators')
plt.ylabel('Test accuracy')
plt.title('n_estimators vs accuracy for different learning rates')
plt.grid(True)
plt.legend()
plt.show()

# Find best config
best = max(results, key=lambda r: r['accuracy'])
print("Best config:", best['n_estimators'], "lr=", best['learning_rate'], "accuracy=", best['accuracy'])
best_model = best['model']

#Part C — Misclassification pattern for best model
best_clf = best_model  # already trained
# sklearn AdaBoost stores estimators_ and estimator_weights_
estimators = best_clf.estimators_
est_weights = best_clf.estimator_weights_
# To compute training error of each weak learner:
weak_errors = []
for est in estimators:
    pred_train = est.predict(X_train_s)
    err = np.mean(pred_train != y_train)
    weak_errors.append(err)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(range(1,len(weak_errors)+1), weak_errors, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Weak learner training error')
plt.title('Weak learner error vs iteration')
plt.grid(True)

n = X_train_s.shape[0]
W = np.ones(n) / n
errors_seq = []
weights_seq = []
for est, alpha in zip(best_clf.estimators_, best_clf.estimator_weights_):
    pred = est.predict(X_train_s)
    mis = (pred != y_train).astype(int)
    err = np.sum(W * mis) / np.sum(W)
    errors_seq.append(err)
    # update
    # standard SAMME.R uses estimator_weight != 0. For reproducibility we use same update as manual:
    alpha_t = alpha
    W = W * np.exp(alpha_t * (mis*1 - (1-mis)*1))
    W = W / np.sum(W)
weights_seq = W

plt.subplot(1,2,2)
plt.hist(weights_seq, bins=30)
plt.xlabel('Sample weight')
plt.title('Sample weight distribution after final stage')
plt.tight_layout()
plt.show()

# Which samples got highest weights?
top_idx = np.argsort(weights_seq)[-10:][::-1]
print("Top 10 train sample indices with highest final weights (relative to train):", top_idx)
print("Their weights:", weights_seq[top_idx])
print("Their labels:", y_train[top_idx])

importances = best_clf.feature_importances_
if isinstance(feat_names, (list, np.ndarray)):
    fnames = feat_names
else:
    fnames = [f'X{i}' for i in range(X_hd.shape[1])]

feat_imp = pd.Series(importances, index=fnames).sort_values(ascending=False)
print("Top 5 features:\n", feat_imp.head(5))

# Plot top 10
feat_imp.head(10).plot(kind='barh', title='Top feature importances (AdaBoost)')
plt.gca().invert_yaxis()
plt.show()

#Q3 — WISDM Accelerometer (Activity classification with AdaBoost)
#Part A — Data preparation
# Q3A: Load WISDM (update path if needed)
w_path = "WISDM_ar_v1.1_raw.txt"  # place file here
try:
    raw = pd.read_csv(w_path, header=None, names=['user_id','activity','timestamp','x','y','z'], comment=';', engine='python')
except Exception as e:
    # Try alternative parsing if file uses semicolons/commas differently
    raw = pd.read_table(w_path, sep=',', header=None, names=['user_id','activity','timestamp','x','y','z'], engine='python')

# Clean rows: some lines end with commas or have noise
raw = raw.dropna(subset=['activity','x','y','z']).reset_index(drop=True)
# ensure numeric columns
raw['x'] = pd.to_numeric(raw['x'], errors='coerce')
raw['y'] = pd.to_numeric(raw['y'], errors='coerce')
raw['z'] = pd.to_numeric(raw['z'], errors='coerce')
raw = raw.dropna(subset=['x','y','z']).reset_index(drop=True)

# Create binary label: vigorous (Jogging, Upstairs) ->1 else 0
raw['activity_lower'] = raw['activity'].str.strip().str.lower()
vigorous = ['jogging','upstairs','running']  # include variants
raw['label'] = raw['activity_lower'].apply(lambda s: 1 if any(v in s for v in vigorous) else 0)

# For classification, aggregate windows into features (mean/std) per short sliding window per user
# Create windows of 2 seconds (assuming timestamp in ms, sampling ~20Hz -> window size 40 samples)
# Simpler: group by consecutive chunks - for demonstration create non-overlapping windows of 50 samples
window_size = 50
features = []
labels = []
for start in range(0, len(raw), window_size):
    chunk = raw.iloc[start:start+window_size]
    if len(chunk) < window_size:
        continue
    feat = {}
    feat['x_mean'] = chunk['x'].mean()
    feat['y_mean'] = chunk['y'].mean()
    feat['z_mean'] = chunk['z'].mean()
    feat['x_std'] = chunk['x'].std()
    feat['y_std'] = chunk['y'].std()
    feat['z_std'] = chunk['z'].std()
    # magnitude features
    mag = np.sqrt(chunk['x']**2 + chunk['y']**2 + chunk['z']**2)
    feat['mag_mean'] = mag.mean()
    feat['mag_std'] = mag.std()
    # label: majority label in chunk
    labels.append(int(chunk['label'].mode()[0]))
    features.append(feat)
X_w = pd.DataFrame(features).fillna(0)
y_w = np.array(labels)

print("Constructed feature matrix shape:", X_w.shape)
# Train-test split 70/30
X_train, X_test, y_train, y_test = train_test_split(X_w, y_w, test_size=0.3, random_state=42, stratify=y_w)

#Part B — Decision stump baseline
# Decision stump baseline
stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_train, y_train)
print("Stump train acc:", accuracy_score(y_train, stump.predict(X_train)))
print("Stump test  acc:", accuracy_score(y_test, stump.predict(X_test)))
print("Confusion matrix (test):\n", confusion_matrix(y_test, stump.predict(X_test)))

#Part C — Manual AdaBoost (T = 20)
# Manual AdaBoost for WISDM features (like before)
def manual_adaboost_df(X_train_df, y_train, X_test_df, y_test, T=20):
    X_train_arr = X_train_df.values
    X_test_arr = X_test_df.values
    n = X_train_arr.shape[0]
    W = np.ones(n)/n
    learners = []
    alphas = []
    errors = []
    for t in range(1, T+1):
        stump = DecisionTreeClassifier(max_depth=1, random_state=42)
        stump.fit(X_train_arr, y_train, sample_weight=W)
        pred = stump.predict(X_train_arr)
        mis = (pred != y_train).astype(int)
        err_t = np.sum(W * mis) / np.sum(W)
        err_t = np.clip(err_t, 1e-12, 1-1e-12)
        alpha_t = 0.5 * np.log((1-err_t)/err_t)
        mis_idx = np.where(mis==1)[0]
        print(f"\nIteration {t}, mis idx (first 100):", mis_idx.tolist()[:100])
        print("Weights of misclassified samples (first 20):", W[mis_idx][:20].tolist())
        print("err:", err_t, "alpha:", alpha_t)
        # update W
        W = W * np.exp(alpha_t * (mis*1 - (1-mis)*1))
        W /= np.sum(W)
        learners.append(stump)
        alphas.append(alpha_t)
        errors.append(err_t)
    # final predict
    def predict_comb(X_arr):
        agg = None
        for alpha, learner in zip(alphas, learners):
            p = learner.predict(X_arr)
            p_s = np.where(p==1, 1, -1)
            agg = (alpha*p_s) if agg is None else agg + alpha*p_s
        return np.where(agg>=0, 1, 0)
    train_pred = predict_comb(X_train_arr)
    test_pred = predict_comb(X_test_arr)
    print("Manual AdaBoost train acc", accuracy_score(y_train, train_pred))
    print("Manual AdaBoost test acc", accuracy_score(y_test, test_pred))
    print("Confusion matrix (test):\n", confusion_matrix(y_test, test_pred))
    # plots
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(range(1,T+1), errors, marker='o')
    plt.xlabel('Iteration'); plt.ylabel('Weighted error'); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(range(1,T+1), alphas, marker='o')
    plt.xlabel('Iteration'); plt.ylabel('Alpha'); plt.grid(True)
    plt.show()
    return learners, alphas, errors, W

learners_w, alphas_w, errors_w, final_weights_w = manual_adaboost_df(X_train, y_train, X_test, y_test, T=20)

#Part D — sklearn AdaBoost
# sklearn AdaBoost
clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)
clf.fit(X_train, y_train)
print("sklearn AdaBoost train acc:", accuracy_score(y_train, clf.predict(X_train)))
print("sklearn AdaBoost test acc: ", accuracy_score(y_test, clf.predict(X_test)))
print("Confusion matrix (test):\n", confusion_matrix(y_test, clf.predict(X_test)))
# Compare manual vs sklearn performance in your report.

