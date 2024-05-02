# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score

# %%
df = pd.read_csv('data.csv')

# %%
df.head()

# %%
df.shape

# %%
df.describe()

# %%
df.info()

# %%
df.columns

# %%
df.isnull().sum()

# %%
df.drop("Unnamed: 32", axis=1)

# %%
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

# %%
df.head()

# %%
plt.figure(figsize=(6, 6))
df['diagnosis'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Benign', 'Malignant'], colors=['skyblue', 'lightcoral'])
plt.title('Distribution of Diagnosis')
plt.ylabel('')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='radius_mean', y='texture_mean', hue='diagnosis', data=df, palette='coolwarm')
plt.title('Scatter Plot: Radius Mean vs. Texture Mean (Colored by Diagnosis)')
plt.xlabel('Radius Mean')
plt.ylabel('Texture Mean')
plt.legend(title='Diagnosis')
plt.show()

# %%
continuous_features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

plt.figure(figsize=(12, 15))
for i, feature in enumerate(continuous_features, 1):
    plt.subplot(5, 7, i)
    sns.histplot(df[feature], bins=30, kde=True)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of All Features')
plt.show()

# %%
selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
selected_df = df[selected_features + ['diagnosis']]
sns.pairplot(selected_df, hue='diagnosis', palette='husl', corner=True)
plt.title('Pairplot of Selected Features by Diagnosis')
plt.show()

# %%
scaler = StandardScaler()
X = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
y = df['diagnosis']
X_std = scaler.fit_transform(X)

# %%
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

# %%
bagging_clf = BaggingClassifier(random_state=42)
rf_clf = RandomForestClassifier(random_state=42)
adaboost_clf = AdaBoostClassifier(random_state=42)

# %%
bagging_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
adaboost_clf.fit(X_train, y_train)

# %%
bagging_pred = bagging_clf.predict(X_test)
rf_pred = rf_clf.predict(X_test)
adaboost_pred = adaboost_clf.predict(X_test)

# %%
bagging_accuracy = accuracy_score(y_test, bagging_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
adaboost_accuracy = accuracy_score(y_test, adaboost_pred)

# %%
print("Bagging Classifier Accuracy:", bagging_accuracy)
print("Random Forest Classifier Accuracy:", rf_accuracy)
print("AdaBoost Classifier Accuracy:", adaboost_accuracy)

# %%
models = ['Bagging', 'Random Forest', 'AdaBoost']
accuracies = [bagging_accuracy, rf_accuracy, adaboost_accuracy]

plt.bar(models, accuracies)
plt.xlabel('Ensemble Models')
plt.ylabel('Accuracy Score')
plt.title('Comparison of Ensemble Models')
plt.ylim(0.9, 1.0)
plt.show()

# %%
bagging_probs = bagging_clf.predict_proba(X_test)[:, 1]
bagging_auc = roc_auc_score(y_test, bagging_probs)
bagging_fpr, bagging_tpr, _ = roc_curve(y_test, bagging_probs)
plt.plot(bagging_fpr, bagging_tpr, label=f'Bagging (AUC = {bagging_auc:.2f})')

# %%
rf_probs = rf_clf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')

# %%
adaboost_probs = adaboost_clf.predict_proba(X_test)[:, 1]
adaboost_auc = roc_auc_score(y_test, adaboost_probs)
adaboost_fpr, adaboost_tpr, _ = roc_curve(y_test, adaboost_probs)
plt.plot(adaboost_fpr, adaboost_tpr, label=f'AdaBoost (AUC = {adaboost_auc:.2f})')

# %%
plt.figure(figsize=(8, 6))
plt.plot(bagging_fpr, bagging_tpr, label=f'Bagging Classifier (AUC = {bagging_auc:.2f})')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest Classifier (AUC = {rf_auc:.2f})')
plt.plot(adaboost_fpr, adaboost_tpr, label=f'AdaBoost Classifier (AUC = {adaboost_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Baseline (Random Classifier)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()