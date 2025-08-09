import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.manifold import TSNE

# =================================================================
# Data Loading & Preprocessing
# =================================================================
data1 = pd.read_excel('glycoprotein.xlsx')
data2 = pd.read_excel('AFP.xlsx')

X, y = [], []
for i in range(1, data1.shape[1]):
    X.append(data1.iloc[:, i].values)
    y.append(0)

for i in range(1, data2.shape[1]):
    X.append(data2.iloc[:, i].values)
    y.append(1)

X = np.array(X)
y = np.array(y)

# =================================================================
# Train-Test Split
# =================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =================================================================
# Pipeline & Cross-Validation
# =================================================================
pipeline = Pipeline([
    ('rf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20],
    'rf__max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(best_model, 'optimized_rf_model.pkl')

# =================================================================
# Visualizations (English Only)
# =================================================================
# 1. Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Glycoprotein", "AFP"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

plot_confusion_matrix(y_test, y_pred)

# 2. t-SNE Visualization
def plot_tsne(X, y, title="t-SNE Visualization"):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, ticks=[0, 1], label="Class")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(title)
    plt.show()

plot_tsne(X_train, y_train, title="Training Set t-SNE")

# 3. Feature Importance
def plot_feature_importance(model, num_features=20):
    importances = model.named_steps['rf'].feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_importances = importances[sorted_idx][:num_features]
    top_features = sorted_idx[:num_features]

    plt.figure(figsize=(12, 6))
    plt.barh(range(num_features), top_importances, align='center')
    plt.yticks(range(num_features), top_features)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Index')
    plt.title(f"Top {num_features} Feature Importance (Random Forest)")
    plt.gca().invert_yaxis()  # 最重要的特征显示在顶部
    plt.show()

plot_feature_importance(best_model)

# 4. ROC Curve
def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

y_prob = best_model.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, y_prob)

# 5. Learning Curve
def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training Accuracy")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-Validation Accuracy")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend(loc="lower right")
    plt.show()

plot_learning_curve(best_model, X_train, y_train)
