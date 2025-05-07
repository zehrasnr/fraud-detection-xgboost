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


df_train = pd.read_csv("dataset/df_train_to_Feature.csv")
df_test = pd.read_csv("dataset/df_test_to_Feature.csv")

#// #####Yüzde 85 Eksik Değer içerenler

missing_values = df_train.isnull().sum() / len(df_train) * 100
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
missing_df = pd.DataFrame(missing_values).reset_index()

missing_85 = missing_df[missing_df[0]>85]

drop85 = missing_85["index"].tolist()

#Yüzde 85lerin atılması:
df_train = df_train.drop(drop85,axis=1)
df_test = df_test.drop(drop85,axis=1)

#// ####High Corrletaion Kolonların Bulunması


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


def high_correlated_cols_single_reduced(dataframe, corr_th=0.85):
    corr = dataframe.corr().abs()
    upper_triangle_matrix = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Yüksek korelasyonlu kolonları gruplandırmak için set kullanımı
    drop_list = []
    selected_cols = set()

    for col in upper_triangle_matrix.columns:
        # Eğer bu sütunun herhangi bir korelasyonu belirtilen eşiğin üstündeyse
        high_corr_cols = [index for index, value in upper_triangle_matrix[col].items() if value > corr_th]

        # Eğer bu kolon grupta zaten eklenmediyse
        if high_corr_cols and not selected_cols.intersection(high_corr_cols):
            # Yalnızca ilk kolonlardan birini seçip ekle
            selected_cols.add(col)
            drop_list.append(col)

    return drop_list

highcorr_list2 = high_correlated_cols_single_reduced(df_train[num_cols])
v_high_corr_cols = [col for col in highcorr_list2 if col.startswith(("V", "C","D"))]

#High Corr Yüzde 85lerin atılması:
df_train = df_train.drop(v_high_corr_cols,axis=1)
df_test = df_test.drop(v_high_corr_cols,axis=1)

#df_Train ve test hata düzeltme
df_train = df_train.drop("Unnamed: 0", axis = 1)
df_test = df_test.drop("Unnamed: 0", axis = 1)

#// #####Aykırı Değerlerin bastırılması:

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.01, q3=0.99):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df_train, cat_th = 10, car_th=20)
cat_cols_test, num_cols_test, cat_but_car_test, num_but_cat_test = grab_col_names(df_test, cat_th = 10, car_th=20)

num_cols = [col for col in num_cols if col not in ["TransactionAmt", "TransactionID", "TransactionDT"]]
num_cols_test = [col for col in num_cols_test if col not in ["TransactionAmt", "TransactionID", "TransactionDT"]]


for col in num_cols:
    print(col, check_outlier(df_train, col, 0.01, 0.99))

for col in num_cols_test:
    print(col, check_outlier(df_test, col, 0.01, 0.99))

for col in num_cols:
    replace_with_thresholds(df_train, col)

for col in num_cols_test:
    replace_with_thresholds(df_test, col)

#// #####Kalan Eksik Değerlerin -9999 ile doldurulması:
missing_values_check = df_train.isnull().sum() / len(df_train) * 100
missing_values_check = missing_values_check[missing_values_check > 0].sort_values(ascending=False)
missing_df_check = pd.DataFrame(missing_values_check).reset_index()
df_train["TransactionDT"].isnull().any()

df_train.fillna(-9999, inplace=True)
df_test.fillna(-9999, inplace=True)

#// #####PCA Uygulanması!

#V Kolonları Özelinde

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df_train, cat_th = 10, car_th=20)

only_V = [col for col in df_train if col.startswith(("V"))]
df_train_only_V = df_train[only_V]
df_test_only_V = df_test[only_V]

pca1 = PCA().fit(df_train[only_V])
plt.plot(np.cumsum(pca1.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()

df_train_only_V = StandardScaler().fit_transform(df_train_only_V)
pca = PCA()
pca_fit = pca.fit_transform(df_train_only_V)
np.cumsum(pca.explained_variance_ratio_) #32 adet olmasına karar verilmiştir.

    #Yeniden PCA Bakılırsa

pca_final = PCA(n_components=32)
pca_fit_final = pca_final.fit_transform(df_train_only_V)
np.cumsum(pca_final.explained_variance_ratio_)
pca_fit_test_final = pca_final.fit_transform(df_test_only_V)


only_V
others_train = [col for col in df_train.columns if col not in only_V]
others_test = [col for col in df_test.columns if col not in only_V]

df_train_v_pca = pd.DataFrame(pca_fit_final, columns=[f'PCA_V_{i+1}' for i in range(32)])
df_test_v_pca = pd.DataFrame(pca_fit_test_final, columns=[f'PCA_V_{i+1}' for i in range(32)])

df_train = pd.concat([df_train[others_train], df_train_v_pca],axis=1)
df_test = pd.concat([df_test[others_test], df_test_v_pca],axis=1)

df_train.shape
df_test.shape


#// ############ Yeni Değişkenler Oluşturma Zamanı

# Hafta içi/Hafta sonu
df_train['new_is_weekend'] = df_train['day'].apply(lambda x: 1 if x in [6, 7] else 0)
df_test['new_is_weekend'] = df_test['day'].apply(lambda x: 1 if x in [6, 7] else 0)

#Kart ile ilgili özetler
card_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
for col in card_cols:
    df_train[f'new_{col}_transaction_count'] = df_train.groupby(col)['TransactionAmt'].transform('count')
    df_train[f'new_{col}_transaction_mean'] = df_train.groupby(col)['TransactionAmt'].transform('mean')
    df_test[f'new_{col}_transaction_count'] = df_test.groupby(col)['TransactionAmt'].transform('count')
    df_test[f'new_{col}_transaction_mean'] = df_test.groupby(col)['TransactionAmt'].transform('mean')

# Email Domain Aggregates
email_cols = ['P_emaildomain', 'R_emaildomain']
for col in email_cols:
    df_train[f'new_{col}_transaction_count'] = df_train.groupby(col)['TransactionAmt'].transform('count')
    df_train[f'new_{col}_transaction_mean'] = df_train.groupby(col)['TransactionAmt'].transform('mean')
    df_test[f'new_{col}_transaction_count'] = df_test.groupby(col)['TransactionAmt'].transform('count')
    df_test[f'new_{col}_transaction_mean'] = df_test.groupby(col)['TransactionAmt'].transform('mean')

# Device and Browser Aggregates
df_train['new_DeviceType_Transaction_count'] = df_train.groupby('DeviceType')['TransactionAmt'].transform('count')
df_train['new_DeviceType_Transaction_mean'] = df_train.groupby('DeviceType')['TransactionAmt'].transform('mean')
df_test['new_DeviceType_Transaction_count'] = df_test.groupby('DeviceType')['TransactionAmt'].transform('count')
df_test['new_DeviceType_Transaction_mean'] = df_test.groupby('DeviceType')['TransactionAmt'].transform('mean')

# Fraud Oranı Yüksek Kategorilerden Bayrak Değişkenleri
df_train['new_is_fraud_high_ProductCD'] = df_train['ProductCD'].apply(lambda x: 1 if x in ['C', 'H', 'R'] else 0)
df_test['new_is_fraud_high_ProductCD'] = df_test['ProductCD'].apply(lambda x: 1 if x in ['C', 'H', 'R'] else 0)

# Oran Değişkenleri
df_train['new_TransactionAmt_card1_ratio'] = df_train['TransactionAmt'] / (df_train['card1'] + 1e-5)
df_train['new_TransactionAmt_card4_ratio'] = df_train['TransactionAmt'] / (df_train['card4'].factorize()[0] + 1e-5)
df_test['new_TransactionAmt_card1_ratio'] = df_test['TransactionAmt'] / (df_test['card1'] + 1e-5)
df_test['new_TransactionAmt_card4_ratio'] = df_test['TransactionAmt'] / (df_test['card4'].factorize()[0] + 1e-5)

# DeviceInfo ve DeviceType ile yapılan toplam işlem sayısı:
device_aggregates = df_train.groupby(['DeviceType', 'DeviceInfo'])['TransactionID'].nunique()
df_train = df_train.merge(device_aggregates.rename('new_Device_Transaction_Count'), on=['DeviceType', 'DeviceInfo'], how='left')

device_aggregates_test = df_test.groupby(['DeviceType', 'DeviceInfo'])['TransactionID'].nunique()
df_test = df_test.merge(device_aggregates_test.rename('new_Device_Transaction_Count'), on=['DeviceType', 'DeviceInfo'], how='left')

# Device bilgileri: Kullanıcı cihazının tipi ve detayları
df_train["new_DeviceInfo_group"] = df_train["DeviceInfo"].apply(lambda x: x.split()[0] if pd.notnull(x) and x != -9999 else "Unknown")
df_train["new_DeviceType_flag"] = df_train["DeviceType"].apply(lambda x: 1 if x == 'desktop' else 0)

df_test["new_DeviceInfo_group"] = df_test["DeviceInfo"].apply(lambda x: x.split()[0] if pd.notnull(x) and x != -9999 else "Unknown")
df_test["new_DeviceType_flag"] = df_test["DeviceType"].apply(lambda x: 1 if x == 'desktop' else 0)

# Address ile ilgili özellikler: addr1 ve addr2 kolonlarını birleştirip risk grubu oluşturma
df_train["new_addr_combined"] = df_train.apply(lambda row: str(row["addr1"]) + "_" + str(row["addr2"]) if row["addr1"] != -9999 and row["addr2"] != -9999 else "Unknown", axis=1)
df_train["new_addr1_group"] = df_train["addr1"].apply(lambda x: "High Risk" if x in [264, 325] else "Low Risk")

df_test["new_addr_combined"] = df_test.apply(lambda row: str(row["addr1"]) + "_" + str(row["addr2"]) if row["addr1"] != -9999 and row["addr2"] != -9999 else "Unknown", axis=1)
df_test["new_addr1_group"] = df_test["addr1"].apply(lambda x: "High Risk" if x in [264, 325] else "Low Risk")

# Hour değişkenini belirli saat aralıklarına göre gruplandırma
df_train["new_hour_group"] = pd.cut(df_train["hour"], bins=[0, 6, 12, 18, 24], labels=["Night", "Morning", "Afternoon", "Evening"])
df_test["new_hour_group"] = pd.cut(df_test["hour"], bins=[0, 6, 12, 18, 24], labels=["Night", "Morning", "Afternoon", "Evening"])

# E-mail domain özellikleri: P_emaildomain ve R_emaildomain üzerinden sağlayıcı ve tür bilgisi
df_train["P_emaildomain_provider"] = df_train["P_emaildomain"].apply(lambda x: x.split('.')[0] if pd.notnull(x) and x != -9999 else "Unknown")
df_train["P_emaildomain_type"] = df_train["P_emaildomain"].apply(lambda x: "commercial" if x in ["gmail.com", "yahoo.com", "outlook.com"] else "other")
df_train["R_emaildomain_provider"] = df_train["R_emaildomain"].apply(lambda x: x.split('.')[0] if pd.notnull(x) and x != -9999 else "Unknown")
df_train["R_emaildomain_type"] = df_train["R_emaildomain"].apply(lambda x: "commercial" if x in ["gmail.com", "yahoo.com", "outlook.com"] else "other")

df_test["P_emaildomain_provider"] = df_test["P_emaildomain"].apply(lambda x: x.split('.')[0] if pd.notnull(x) and x != -9999 else "Unknown")
df_test["P_emaildomain_type"] = df_test["P_emaildomain"].apply(lambda x: "commercial" if x in ["gmail.com", "yahoo.com", "outlook.com"] else "other")
df_test["R_emaildomain_provider"] = df_test["R_emaildomain"].apply(lambda x: x.split('.')[0] if pd.notnull(x) and x != -9999 else "Unknown")
df_test["R_emaildomain_type"] = df_test["R_emaildomain"].apply(lambda x: "commercial" if x in ["gmail.com", "yahoo.com", "outlook.com"] else "other")

# TransactionAmt ile card1 arasındaki oran
df_train["new_TransactionAmt_to_card1_ratio"] = df_train.apply(lambda x: x["TransactionAmt"] / x["card1"] if x["card1"] != -9999 else -9999, axis=1)
df_test["new_TransactionAmt_to_card1_ratio"] = df_test.apply(lambda x: x["TransactionAmt"] / x["card1"] if x["card1"] != -9999 else -9999, axis=1)

# TransactionDT'de zamansal fark hesaplama
df_train["new_TransactionDT_Diff"] = df_train["TransactionDT"].diff().fillna(0)
df_test["new_TransactionDT_Diff"] = df_test["TransactionDT"].diff().fillna(0)

# Card ve Email frekans bilgisi
df_train["new_card_email_freq"] = df_train.groupby(["card1", "P_emaildomain"])["TransactionID"].transform("count")
df_test["new_card_email_freq"] = df_test.groupby(["card1", "P_emaildomain"])["TransactionID"].transform("count")

df_train.shape
df_test.shape


df_train_to_Model = df_train.to_csv("dataset/df_train_to_Model.csv")
df_test_to_Model = df_test.to_csv("dataset/df_test_to_Model.csv")

