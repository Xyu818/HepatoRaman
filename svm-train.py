import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # added PCA import
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import LinearRegression

class BaselineRemoval:
    """Baseline removal using iterative Whittaker smoothing (ZhangFit)."""
    def __init__(self, input_array):
        self.input = np.array(input_array, dtype=float)
        self.lin   = LinearRegression()

    def _WhittakerSmooth(self, x, w, lam, diff=1):
        X = np.matrix(x)
        m = X.size
        E = eye(m, format='csc')
        D = E[1:] - E[:-1]
        W = diags(w, 0, shape=(m, m))
        A = csc_matrix(W + lam * D.T * D)
        B = csc_matrix(W * X.T)
        bg = spsolve(A, B)
        return np.array(bg).flatten()

    def ZhangFit(self, lam=100, porder=1, iters=15):
        """Return signal with baseline removed."""
        y = self.input.copy()
        m = y.size
        w = np.ones(m)
        for i in range(1, iters+1):
            bg = self._WhittakerSmooth(y, w, lam, porder)
            d  = self.input - bg
            dssn = np.abs(d[d<0].sum())
            if dssn < 0.001 * np.abs(self.input).sum() or i == iters:
                break
            w[d>=0] = 0
            w[d<0]  = np.exp(i * np.abs(d[d<0]) / dssn)
            w[0]    = np.exp(i * (d[d<0].max()) / dssn)
            w[-1]   = w[0]
        return self.input - bg

# -------------------- Load data --------------------
data1 = pd.read_excel('glycoprotein.xlsx')
data2 = pd.read_excel('AFP.xlsx')

# -------------------- Build dataset --------------------
X = []
y = []
# Columns 1..end are assumed to be samples; column 0 holds feature names
for i in range(1, data1.shape[1]):
    X.append(data1.iloc[:, i].values)
    y.append(0)  # class 0: Glycoprotein

for i in range(1, data2.shape[1]):
    X.append(data2.iloc[:, i].values)
    y.append(1)  # class 1: AFP

X = np.array(X)
y = np.array(y)

# -------------------- Train/Test split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= PCA block (new) =================
# Standardization (required before PCA/SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA (retain 95% cumulative variance)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Reduced dimensionality: {X_train_pca.shape[1]}, "
      f"Cumulative explained variance: {pca.explained_variance_ratio_.sum():.2f}")

# Use PCA-transformed data
X_train = X_train_pca
X_test = X_test_pca
# ================= end PCA block =================

# -------------------- Train & evaluate SVM --------------------
clf = SVC(kernel='linear', random_state=42)
clf.fit(X_train, y_train)
joblib.dump(clf, 'svm_model.pkl')

y_pred = clf.predict(X_test)
print("Predictions:", y_pred)
print("True labels:", y_test)

# Performance report
print(classification_report(y_test, y_pred))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
