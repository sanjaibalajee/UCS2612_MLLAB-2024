# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
import numpy as np

# %%
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, roc_curve, roc_auc_score,accuracy_score

# %%
red_wine_df = pd.read_csv('winequality-red.csv', sep=';')
red_wine_df['color'] = 'red'

# %%
red_wine_df.head()

# %%
red_wine_df.shape

# %%
white_wine_df = pd.read_csv('winequality-white.csv', sep=';')
white_wine_df['color'] = 'white'

# %%
white_wine_df.head()

# %%
white_wine_df.shape

# %%
wine_df = pd.concat([red_wine_df, white_wine_df])

# %%
wine_df.head()

# %%
wine_df.describe()

# %%
wine_df.shape

# %%
wine_df.info()

# %%
wine_df.columns

# %%
le = LabelEncoder()
wine_df['color'] = le.fit_transform(wine_df['color'])

# %%
wine_df.head()

# %%
corr_matrix = wine_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# %%
selected_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
sns.pairplot(wine_df[selected_features+ ['color']], hue='color', palette={0: 'red', 1: 'blue'})
plt.suptitle('Pairplot of Selected Features by Wine Color', y=1.02)
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(data=wine_df, x='quality', hue='color', multiple='stack', bins=10, kde=True,palette={0: 'red', 1: 'blue'})
plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Distribution of Wine Quality Scores by Color')
plt.legend(title='Color', labels=['Red', 'White'])
plt.show()

# %%
X = wine_df.drop(['color'], axis=1)
y = wine_df['color']

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# %%
lda = LDA(n_components=1)
X_lda = lda.fit_transform(X, y)

# %%
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette={0: 'red', 1: 'blue'}, legend='full')
plt.title('PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_lda[:, 0], y=[0] * len(X_lda), hue=y, palette={0: 'red', 1: 'blue'}, legend='full')
plt.title('LDA')
plt.xlabel('LD1')
plt.yticks([])
plt.tight_layout()

plt.suptitle('PCA vs LDA')
plt.show()

# %%
X_cls = wine_df.drop(['color'], axis=1)
y_cls = wine_df['color']

# %%
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# %%
scaler_cls = StandardScaler()
X_cls_train_scaled = scaler_cls.fit_transform(X_cls_train)
X_cls_test_scaled = scaler_cls.transform(X_cls_test)

# %%
# PCA
pca_cls = PCA(n_components=2)
X_cls_train_pca = pca_cls.fit_transform(X_cls_train_scaled)
X_cls_test_pca = pca_cls.transform(X_cls_test_scaled)

# LDA
lda_cls = LDA(n_components=1)
X_cls_train_lda = lda_cls.fit_transform(X_cls_train_scaled, y_cls_train)
X_cls_test_lda = lda_cls.transform(X_cls_test_scaled)

# KNN - original
cls_model_original = KNeighborsClassifier()
cls_model_original.fit(X_cls_train_scaled, y_cls_train)

# KNN using PCA Features
cls_model_pca = KNeighborsClassifier()
cls_model_pca.fit(X_cls_train_pca, y_cls_train)

# KNN using LDA Features
cls_model_lda = KNeighborsClassifier()
cls_model_lda.fit(X_cls_train_lda, y_cls_train)

# Classification Evaluation for Original Features
cls_pred_original = cls_model_original.predict(X_cls_test_scaled)
cls_accuracy_original = accuracy_score(y_cls_test, cls_pred_original)

# Classification Evaluation for PCA Features
cls_pred_pca = cls_model_pca.predict(X_cls_test_pca)
cls_accuracy_pca = accuracy_score(y_cls_test, cls_pred_pca)

# Classification Evaluation for LDA Features
cls_pred_lda = cls_model_lda.predict(X_cls_test_lda)
cls_accuracy_lda = accuracy_score(y_cls_test, cls_pred_lda)

print("Accuracy (Original Features):", cls_accuracy_original)
print("Accuracy (PCA Features):", cls_accuracy_pca)
print("Accuracy (LDA Features):", cls_accuracy_lda)

# %%
def plot_metrics(model, X, y_true, title):
    # Calculate predictions and probabilities
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Plot Confusion Matrix
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - ' + title)

    # Plot ROC Curve
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - ' + title)
    plt.legend()

    plt.tight_layout()
    plt.show()

# %%
#Plot metrics for Original Features
plot_metrics(cls_model_original, X_cls_test_scaled, y_cls_test, 'Original Features')

# %%
# Plot metrics for PCA Features
plot_metrics(cls_model_pca, X_cls_test_pca, y_cls_test, 'PCA Features')

# %%
# Plot metrics for LDA Features
plot_metrics(cls_model_lda, X_cls_test_lda, y_cls_test, 'LDA Features')