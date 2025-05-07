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

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)


df_train = pd.read_csv("dataset/df_train_to_Model.csv")
df_test = pd.read_csv("dataset/df_test_to_Model.csv")

#Unnamed Silme
df_train = df_train.drop("Unnamed: 0", axis = 1)
df_test = df_test.drop("Unnamed: 0", axis = 1)


#// ######// Encoding
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df_train, cat_th = 10, car_th=20)
cat_cols = [col for col in cat_cols if col not in ["isFraud","addr2"]]
cat_cols = [col for col in cat_cols if col not in num_but_cat]

cat_cols_test, num_cols_test, cat_but_car_test, num_but_cat_test = grab_col_names(df_test, cat_th = 10, car_th=20)
cat_cols_test = [col for col in cat_cols_test if col not in num_but_cat_test]

#One Hot Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df_train = pd.get_dummies(df_train, columns=cat_cols, drop_first=True)
df_test = pd.get_dummies(df_test, columns=cat_cols_test, drop_first=True)


set(df_train.columns.tolist())-set(df_test.columns.tolist())
set(df_test.columns.tolist())-set(df_train.columns.tolist())

df_test['card6_debit or credit'] = False


#Frequncy Encoding:

# Train için Frequency Encoding
for col in cat_but_car:
    train_freq = df_train[col].value_counts() / len(df_train)
    df_train[col + '_freq'] = df_train[col].map(train_freq)

df_train.drop(columns=cat_but_car, inplace=True)

# Test için Frequency Encoding
for col in cat_but_car_test:
    test_freq = df_test[col].value_counts() / len(df_test)
    df_test[col + '_freq'] = df_test[col].map(test_freq)

df_test.drop(columns=cat_but_car_test, inplace=True)


df_train.shape
df_test.shape


df_train_to_ModelFinal = df_train.to_csv("dataset/df_train_to_ModelFinal.csv")
df_test_to_ModelFinal = df_test.to_csv("dataset/df_test_to_ModelFinal.csv")






