# %%
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
import scipy.stats
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
pd.set_option('display.float_format', lambda x: '%.2f' % x)
%matplotlib inline

# %%
data = pd.read_csv('diabetes_prediction_dataset.csv')


# %%


# %%
data.head()

# %%
# Data Statistics
data.describe()

# %%
num_rows, num_columns = data.shape
print("Number of rows:", num_rows)
print("Number of columns:", num_columns)

# %%
data['diabetes'].value_counts()

# %%
plt.pie(data['diabetes'].value_counts(), labels = ['non-diabetic', 'diabetic'],
       autopct = '%1.1f%%')
plt.title("Distribution of diabetics in dataset")
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(x='diabetes', y='HbA1c_level', data=data)
plt.title('HbA1c_level by Diabetes Status')
plt.xlabel('Diabetes Status')
plt.ylabel('HbA1c_level')
plt.xticks([0, 1], ['Non-Diabetic', 'Diabetic']) 
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(x='diabetes', y='blood_glucose_level', data=data)
plt.title('Blood Glucose Levels by Diabetes Status')
plt.xlabel('Diabetes Status')
plt.ylabel('Blood Glucose Level')
plt.xticks([0, 1], ['Non-Diabetic', 'Diabetic']) 
plt.show()

# %%
missing_values = data.isnull().sum().sum()
percentage_missing = (missing_values / data.shape[0]) * 100
print("Percentage of missing values:", percentage_missing)

# %%
label_encoder = preprocessing.LabelEncoder()
data['smoking_history'] = label_encoder.fit_transform(data['smoking_history'])
data['gender'] = label_encoder.fit_transform(data['gender'])
data.head()

# %%
numeric_data = data.select_dtypes(include='number')
print(numeric_data)

# %%
numerical_columns = ['age', 'bmi', 'HbA1c_level','smoking_history', 'blood_glucose_level']

# %%
z_scores = data[numerical_columns].apply(zscore)
print(z_scores)

# %%
threshold = 3
outliers = data[z_scores > threshold]
print(outliers)

# %%
outliers_count = (z_scores.abs() > threshold).sum().sum()
print("Number of outliers:", outliers_count)

# %%
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
print(data.head())

# %%
data.head()

# %%
corr_matrix = data.corr()
plt.figure(figsize=(10, 8))  # Set the figure size as desired
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()

# %%
X = data.drop(columns=['diabetes'], axis=1)
y = data['diabetes']

# %%
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# %%
X_train = pd.DataFrame((X_train_raw), columns=X_train_raw.columns)
X_test =  pd.DataFrame((X_test_raw), columns=X_test_raw.columns)
X_test.head()

# %%
pca = PCA(random_state=42)
pca.fit(X_train)

# %%
cumulative_variance = pca.explained_variance_ratio_.cumsum()
n_components_90 = (cumulative_variance <= 0.90).sum()
print("Number of components to explain 90% variance:", n_components_90)

# %%
pca = PCA(n_components=5, random_state=42)
pca.fit(X_train)

X_train_PCA = pd.DataFrame(pca.transform(X_train))
X_test_PCA = pd.DataFrame(pca.transform(X_test))

# %%
X_train_PCA.columns = [str(column_name) for column_name in X_train_PCA.columns]
X_test_PCA.columns = [str(column_name) for column_name in X_test_PCA.columns]

# %%
X_train_PCA.head()

# %%
model = tree.DecisionTreeClassifier()

# %%
model.fit(X_train_PCA, y_train)

# %%
fig = plt.figure(figsize=(20,15))
tree.plot_tree(model)
plt.show()

# %%
feature_importances = pd.Series(model.feature_importances_, model.feature_names_in_).sort_values()
feature_importances.plot.barh();
plt.title('Decision Tree Feature Importance')
plt.show()

# %%
y_pred_test = model.predict(X_test_PCA)
y_pred_train= model.predict(X_train_PCA)

# %%
train_decision_tree_report = classification_report(y_train, y_pred_train)
print(train_decision_tree_report)

# %%
test_decision_tree_report = classification_report(y_test, y_pred_test)
print(test_decision_tree_report)

# %%
RocCurveDisplay.from_estimator(model, X_test_PCA, y_test)
plt.show()

# %%
from sklearn.tree import DecisionTreeClassifier
drt = DecisionTreeClassifier()
drt.fit(X_train_PCA, y_train)

print(f'Training score : {drt.score(X_train_PCA, y_train)}')
print(f'Testing score : {drt.score(X_test_PCA, y_test)}')

# %%
from sklearn.model_selection import GridSearchCV
params= {"criterion": ["gini", "entropy"],
       "max_depth": [6,8,10,12,15],
       "min_samples_split":[10,50,100,150,200],
       "min_samples_leaf":[5,10,20,50]}

dt_cv= DecisionTreeClassifier()
Gsearch_dt= GridSearchCV(estimator= dt_cv,param_grid= params, cv=10, n_jobs=-1, verbose= 1,scoring= "accuracy")

Gsearch_dt.fit(X_train_PCA, y_train)

# %%
Gsearch_dt.best_score_

# %%
Gsearch_dt.best_params_

# %%
from sklearn.model_selection import GridSearchCV
params= {"criterion": ["gini", "entropy"],
       "max_depth": [6,8,10],
       "min_samples_split":[10,50,100],
       "min_samples_leaf":[70,80,90,100,110]}

dt_cv= DecisionTreeClassifier()
Gsearch_dt= GridSearchCV(estimator= dt_cv,param_grid= params, cv=10, n_jobs=-1, verbose= 1,scoring= "accuracy")

Gsearch_dt.fit(X_train_PCA, y_train)

# %%
Gsearch_dt.best_score_

# %%
Gsearch_dt.best_params_

# %%
drt1 = DecisionTreeClassifier(criterion= 'entropy', max_depth= 6, min_samples_leaf= 70, min_samples_split= 10)

drt1.fit(X_train_PCA, y_train)

print(f'Training score : {drt1.score(X_train_PCA, y_train)}')
print(f'Testing score : {drt1.score(X_test_PCA, y_test)}')

# %%
from sklearn.tree import plot_tree
fn=X_train_PCA.columns
cn=["yes","no"] 

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (3,3), dpi=500)

plot_tree(drt1,
           feature_names = fn, 
           class_names=cn,
           filled = True);

# %%
pred_train = drt1.predict(X_train_PCA)
pred_test = drt1.predict(X_test_PCA)

# %%
