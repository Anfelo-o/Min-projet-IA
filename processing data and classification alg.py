import pandas as pd
from sklearn.impute import SimpleImputer
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold

hepatitis = pd.read_csv("hepatitis.csv", na_values='?')

print(hepatitis.head())
print(hepatitis.tail())

print(hepatitis.info())
print(hepatitis.columns)


hepatitis.drop(columns=['ID'], inplace=True)

y = hepatitis['target']
X = hepatitis.drop(columns=['target'])


X = X.apply(pd.to_numeric, errors='coerce')

imputer = SimpleImputer(strategy='most_frequent')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


print((X == '?').sum().sum())
print(X.isnull().sum())
print(X.dtypes)

n_unique = X.nunique()
print("Number of unique values:\n", n_unique)

print(X.describe())

missing = pd.DataFrame({'missing': X.isnull().sum()})
print(missing)


############################################################

plt.figure(figsize=(20, 15))
X.boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot for Outlier Detection")
plt.show()


############################################################
# Correlation Heatmap

continuous_cols = [col for col in hepatitis.columns if hepatitis[col].nunique() > 2]
corr = hepatitis[continuous_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap for Continuous Variables")
plt.show()

binary_cols = [col1 for col1 in hepatitis.columns if hepatitis[col1].nunique() == 2]
corr1 = hepatitis[binary_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr1, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap for Binary Variables")
plt.show()
############################################################
# Missing values count
print(hepatitis.isna().sum())

############################################################
# Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=hepatitis, x='target')
plt.title("Target Variable Distribution")
plt.show()

num_features = hepatitis.select_dtypes(include=['int64', 'float64']).columns
cat_features = hepatitis.select_dtypes(include=['object']).columns
for i in num_features:
    print(i)
    hepatitis[i] = hepatitis[i].fillna(hepatitis[i].median())  # Corrigé : éviter inplace sur copie
for i in cat_features:
    print(i)
    hepatitis[i] = hepatitis[i].fillna('None')


print("target = 1:",hepatitis[hepatitis['target'] == 1].count())
print("target =2:",hepatitis[hepatitis['target'] == 2].count())


X = hepatitis[['age', 'gender', 'steroid', 'antivirals', 'fatigue',
       'malaise', 'anorexia', 'liverBig', 'liverFirm', 'spleen', 'spiders',
       'ascites', 'varices', 'bili', 'alk', 'sgot', 'albu', 'protime',
       'histology']]

y = hepatitis['target'].replace({1:0, 2:1})

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


select_feature = SelectKBest(chi2, k=5)
select_feature.fit(X_train_scaled, y_train)

print('Scores:', select_feature.scores_)
print('Selected feature indices:', select_feature.get_support(indices=True))
print('Selected feature names:', X.columns[select_feature.get_support()])

# Logistic Regression
start_time = time.time()
logreg = LogisticRegression(class_weight='balanced', random_state=42)
logreg.fit(X_train_scaled, y_train)
y_pred_lr = logreg.predict(X_test_scaled)
lr_execution_time = time.time() - start_time


start_time = time.time()
rf = RandomForestClassifier(n_estimators=100, min_samples_split=4, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_execution_time = time.time() - start_time


start_time = time.time()
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train))
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
xgb_execution_time = time.time() - start_time


algorithms = ['Logistic Regression', 'Random Forest', 'XGBoost']
execution_times = [lr_execution_time, rf_execution_time, xgb_execution_time]

plt.bar(algorithms, execution_times)
plt.xlabel('Algorithm')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time for Different Algorithms')
plt.show()


def Confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.show()
    return cm

cm_lr = Confusion_matrix(y_test, y_pred_lr, "Logistic Regression Confusion Matrix")
cm_rf = Confusion_matrix(y_test, y_pred_rf, "Random Forest Confusion Matrix")
cm_xgb = Confusion_matrix(y_test, y_pred_xgb, "XGBoost Confusion Matrix")


def calculate_metrics(conf_matrix):
    TP = conf_matrix[1,1]
    FN = conf_matrix[1,0]
    FP = conf_matrix[0,1]
    TN = conf_matrix[0,0]
    return TP, FN, FP, TN

TP1, FN1, FP1, TN1 = calculate_metrics(cm_lr)
TP2, FN2, FP2, TN2 = calculate_metrics(cm_rf)
TP3, FN3, FP3, TN3 = calculate_metrics(cm_xgb)

def accuracy(tp, fp, fn, tn): return (tp + tn) / (tp + tn + fp + fn)
def precision(tp, fp, fn, tn): return tp / (tp + fp) if (tp+fp)>0 else 0
def recall(tp, fp, fn, tn): return tp / (tp + fn) if (tp+fn)>0 else 0
def f1_score(tp, fp, fn, tn):
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return 2*p*r/(p+r) if (p+r)>0 else 0


print("      accuracy    precision    recall    f1_score ")
print("LogReg:", accuracy(TP1, FP1, FN1, TN1), precision(TP1, FP1, FN1, TN1), recall(TP1, FP1, FN1, TN1), f1_score(TP1, FP1, FN1, TN1))
print("RF:     ", accuracy(TP2, FP2, FN2, TN2), precision(TP2, FP2, FN2, TN2), recall(TP2, FP2, FN2, TN2), f1_score(TP2, FP2, FN2, TN2))
print("XGBoost:", accuracy(TP3, FP3, FN3, TN3), precision(TP3, FP3, FN3, TN3), recall(TP3, FP3, FN3, TN3), f1_score(TP3, FP3, FN3, TN3))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                             scale_pos_weight=(len(y) - sum(y)) / sum(y))
}


cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    cv_results[name] = scores
    print(f"{name} CV Accuracy: {scores} | Mean: {scores.mean():.4f} | Std: {scores.std():.4f}")


plt.figure(figsize=(10, 6))
for name, scores in cv_results.items():
    plt.plot(range(1, len(scores) + 1), scores, marker='o', label=f"{name} CV Accuracy")

plt.xticks(range(1, cv.get_n_splits() + 1))
plt.xlabel("CV Fold")
plt.ylabel("Accuracy")
plt.title("Cross-Validation Accuracy for Different Algorithms")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
new_patient = np.array([ 52,1,0,1,1,0,0,1,0,0,0,0,0,1.4,95,70,3.8,11,1]).reshape(1, -1)

# Logistic Regression
pred_lr = logreg.predict(scaler.transform(new_patient))

# Random Forest
pred_rf = rf.predict(new_patient)

# XGBoost
pred_xgb = xgb.predict(new_patient)

print("LR Prediction:", pred_lr)
print("RF Prediction:", pred_rf)
print("XGB Prediction:", pred_xgb)


