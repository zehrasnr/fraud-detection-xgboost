import pandas as pd
import numpy as np
import seaborn as sns

import joblib
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV



pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)


df_train = pd.read_csv("dataset/df_train_to_ModelFinal.csv")
df_test = pd.read_csv("dataset/df_test_to_ModelFinal.csv")

#Unnamed Silme
df_train = df_train.drop("Unnamed: 0", axis = 1)
df_test = df_test.drop("Unnamed: 0", axis = 1)

# Veri setini bağımsız ve bağımlı değişkenlere ayırın
X = df_train.drop(['isFraud', 'TransactionID'], axis=1)
y = df_train['isFraud']

#// ######## Base Model Belirlenmesi

def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [#('LR', LogisticRegression()),
                 #  ('KNN', KNeighborsClassifier()),
                #   ("SVC", SVC()),
                  # ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                #   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False)) #Commentli çünkü uzun sürüyor diye
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

base_models(X, y, scoring="roc_auc") #XGBOOST SEÇİLDİ


#// ######## Seçilen XGBoost Modeli Hiperparametre Optimizasyonu Öncesi

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)

#XGBoost İlk Model'in eğtimi

xgb_model = XGBClassifier(use_label_encoder=False)
xgb_model.fit(X_train, y_train)

#İlk Model sonrası hata takip Kısımları!
cv_results_ilk = cross_validate(xgb_model, X_train, y_train, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results_ilk['test_accuracy'].mean()
cv_results_ilk['test_f1'].mean()
cv_results_ilk['test_roc_auc'].mean()

y_valid_pred = xgb_model.predict(X_valid)
auc_score = roc_auc_score(y_valid, y_valid_pred)

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(xgb_model,X_train,15)

# -- Train Hatası
y_pred_Train = xgb_model.predict(X_train)
y_prob_Train = xgb_model.predict_proba(X_train)[:,1]
print(classification_report(y_train,y_pred_Train))
roc_auc_score(y_train, y_prob_Train)
# -- Test Hatası
y_pred_Test = xgb_model.predict(X_valid)
y_prob_Test = xgb_model.predict_proba(X_valid)[:,1]
print(classification_report(y_valid,y_pred_Test))
roc_auc_score(y_valid, y_prob_Test)


xgb_model = XGBClassifier(use_label_encoder=False)
eval_set = [(X_train, y_train), (X_valid, y_valid)]
xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
# Eğitim ve doğrulama hatalarını görselleştirme
results = xgb_model.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)
plt.figure(figsize=(10, 6))
plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
plt.plot(x_axis, results['validation_1']['logloss'], label='Validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.title('XGBoost Train vs Validation Log Loss')
plt.show()

#// ########Hiperparametre Optimizasyonu

from sklearn.model_selection import RandomizedSearchCV

# Hiperparametre aralığı
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],}

# RandomizedSearchCV kullanımı
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(use_label_encoder=False),
    param_distributions=param_grid,
    n_iter=20,  # İstediğimiz kadar iterasyon
    scoring='roc_auc',  # AUC kullanarak optimizasyon
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Optimizasyonun yapılması
random_search.fit(X_train, y_train)

# En iyi parametreler
print("Best parameters found: ", random_search.best_params_)

#Modelin kurulması:
best_params = random_search.best_params_
xgb_best_model = XGBClassifier(**best_params, use_label_encoder=False)
xgb_best_model.fit(X_train, y_train)

#Hata Değerlendirmeleri

#1
cv_results_final = cross_validate(xgb_best_model, X_train, y_train, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_final['test_accuracy'].mean()
cv_results_final['test_f1'].mean()
cv_results_final['test_roc_auc'].mean()
#2
y_valid_pred_best = xgb_best_model.predict(X_valid)
auc_score_best = roc_auc_score(y_valid, y_valid_pred)
#3
# -- Train Hatası
y_pred_best_Train = xgb_best_model.predict(X_train)
y_prob_best_Train = xgb_best_model.predict_proba(X_train)[:,1]
print(classification_report(y_train,y_pred_best_Train))
roc_auc_score(y_train, y_prob_best_Train)
# -- Test Hatası
y_pred_best_Test = xgb_best_model.predict(X_valid)
y_prob_best_Test = xgb_best_model.predict_proba(X_valid)[:,1]
print(classification_report(y_valid,y_pred_best_Test))
roc_auc_score(y_valid, y_prob_best_Test)

plot_importance(xgb_best_model,X_train,15)


#// ########## Optimizasyon Sonrası Bütün veri seti ile eğitim:
X = df_train.drop(['isFraud', 'TransactionID'], axis=1)
y = df_train['isFraud']

df_test2 = df_test[X.columns] #df_test2deki sıranın aynı olması için

# En iyi hiperparametrelerle final modeli tanımlama ve eğitme
final_xgb_model = XGBClassifier(**best_params, use_label_encoder=False)
final_xgb_model.fit(X, y)

#// ########## Submission için test verisi ile tahmin

# Tahminler - Sadece olasılıkların pozitif sınıfa (isFraud=1) ait kısmını alıyoruz
y_pred_test_final = final_xgb_model.predict_proba(df_test2)[:, 1]

# Submission DataFrame oluşturma
submission = pd.DataFrame({
    'TransactionID': df_test['TransactionID'],
    'isFraud': y_pred_test_final
})

# CSV olarak kaydetme
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully.")
