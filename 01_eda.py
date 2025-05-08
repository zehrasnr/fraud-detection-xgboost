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

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)



#Veri Setlerinin içeri alınması (Importing Datasets)

dfc_train_identity = pd.read_csv("datasets/train_identity.csv")
dfc_train_transaction = pd.read_csv("datasets/ieee-fraud-detection/train_transaction.csv")

dfc_test_identity = pd.read_csv("datasets/ieee-fraud-detection/test_identity.csv")
dfc_test_transaction = pd.read_csv("datasets/ieee-fraud-detection/test_transaction.csv")

dfc_train_identity
dfc_test_identity

dfc_train_identity.columns
dfc_test_identity.columns

#Train Identity ve Test Identity içerisinde bulunan ID kolonlarının aynı şekilde isimlendirilmesini sağlamak adına

test_id_cols = [col for col in dfc_test_identity.columns if col.startswith("id")]
test_rename_cols = {i:'id_'+str(i[-2]+i[-1]) for i in test_id_cols}
dfc_test_identity = dfc_test_identity.rename(columns = test_rename_cols)

#Veri Setlerinin birleştirilmesi

dfc_train_identity.describe().T
dfc_train_transaction.describe().T
dfc_train_identity.loc[0:4,]


df_train = dfc_train_transaction.merge(dfc_train_identity,on = ["TransactionID"], how = "left")
df_test = dfc_test_transaction.merge(dfc_test_identity,on = ["TransactionID"], how = "left")

df_test.shape
df_train.shape

######
#-- EDA Başlangıç
#####

#Fonskiyonların Girdisi
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    #bağımlı bir değişkeni sayısal değişken ile özetlemek için
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    # bağımlı bir değişkeni kategorik bir değişken ile özetlemek için
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")
    targetdf = pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()})
    return targetdf

def correlation_matrix(df, cols):
    #veri seti içerisindeki sayısal değişkenlerin birbirler
    # i arasındaki ilişkiye, korelasyona bakıyor olacak ve bunu görselleştiriyor olacak bir ısı haritası yardımıyla
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

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

#Genel bir inceleme
check_df(df_train)

#Kategorik kolonlara ayırma
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df_train, cat_th = 10, car_th=20)

#// Train veri setinin sunum için kısımları
len(cat_cols)
len(num_cols)
len(cat_but_car)
len(num_but_cat)
df_train.shape

#// Önce Hedef Değişken Dağılımı ile başlayalım:

df_train["isFraud"].value_counts()
print(df_train['isFraud'].value_counts(normalize=True) * 100)

sns.set_palette("Set2")

plt.figure(figsize=(6, 4))
sns.countplot(x='isFraud', data=df_train, hue="isFraud")
plt.title("isFraud Hedef Değişken Dağılımı")
plt.xlabel("isFraud")
plt.ylabel("Frekans")
plt.show()

#// Kategorik Değişkenlere Bakalım Bi
for col in cat_cols:
    cat_summary(df_train, col)

#-- ######// Her bir kategorik değişkenin unique değer sayısı ve 0.01%'den düşük orana sahip olanların sayısını hesaplayacağız
#Train
summary_df = pd.DataFrame(columns=["Name", "UniqueValue", "SmallerThanValue"])

for cat_var in cat_cols:
    # Yüzdesel dağılımı hesaplayalım
    percentage_distribution = df_train[cat_var].value_counts(normalize=True) * 100

    # Toplam unique değer sayısı
    unique_count = percentage_distribution.size

    # 0.1%'den küçük oran sayısı
    low_ratio_count = (percentage_distribution < 0.01).sum()

    # Sonuçları DataFrame'e ekleyelim
    summary_df = summary_df._append({
        "Name": cat_var,
        "UniqueValue": unique_count,
        "SmallerThanValue": low_ratio_count
    }, ignore_index=True)

summary_df["Ratio"] = (summary_df["SmallerThanValue"] / summary_df["UniqueValue"])
summary_df["Ratio"] = summary_df["Ratio"].apply(lambda x: round(x,1))
num_summary(summary_df,"Ratio")

#Histogram Grafiği
plt.figure(figsize=(8, 5))
ax = sns.histplot(summary_df['Ratio'], bins=10, kde=True)
plt.title("Kategori Bazında Düşük Oranlı Değerlerin Unique Değerlere Oranı (Ratio)")
plt.xlabel("Ratio (Düşük Oranlı Değerlerin Unique Değerlere Oranı)")
plt.ylabel("Frekans")

# Her çubuğun üstüne değerleri yazdır
for p in ax.patches:
    height = p.get_height()
    if height > 0:  # Sadece frekansı sıfırdan büyük olan çubuklara yazı ekleyelim
        ax.text(p.get_x() + p.get_width() / 2, height + 0.5,  # X ve Y koordinatları
                f'{int(height)}', ha="center")  # Değeri yazdır ve ortala

plt.show()


# Çift barlı bir grafik oluşturma
summary_df[summary_df["Ratio"]>0.5].drop("Ratio",axis=1).plot(x='Name', kind='bar', stacked=False, figsize=(12, 6))


filtered_df = summary_df[summary_df["Ratio"] >= 0.6]
ax = filtered_df.plot(
    x='Name', y=['UniqueValue', 'SmallerThanValue'], kind='bar', stacked=False, figsize=(7, 6)
)
# Başlık ve eksen isimlerini ayarla

# Başlık ve eksen isimlerini ayarla
plt.title("Ratio > 0.6 Olan Kategorik Değişkenlerde Unique Değer ve Düşük Oranlı Değer Sayısı")
plt.xlabel("Kategorik Değişkenler")
plt.ylabel("Değer Sayısı")
plt.legend(["Unique Value Count", "Values < 0.01%"])
plt.xticks(rotation=90)

# Her çubuğun üzerine değerleri yazdır
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f')  # fmt='%d' veya '%.2f' ile format ayarlayabilirsin

plt.show()

#-- ######// Missing Value ile Devam Edelim
#1. Veri Seti Özeti ve Eksik Değer Analizi Başlık ile tekrark geri dönebilirsin

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

na_columns = missing_values_table(df_train,na_name = True)
len(na_columns)
df_train.shape
missing_vs_target(df_train,"isFraud", na_columns)

## Eksik değer oranlarını hesaplayalım
missing_values = df_train.isnull().sum() / len(df_train) * 100
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

id_missing_min = missing_values[missing_values.index.str.startswith("id")].idxmin()
min_missing_value = missing_values[id_missing_min]

##Eksik değerlere göre kümelemek
missing_value_bins = pd.cut(
    missing_values,
    bins=[0, 25, 50,75, 90, 100],
    labels=["0-25%", "25-50%", "50-75%","50-90%", "90%+"])
# Kümeleri sayalım
missing_value_groups = missing_value_bins.value_counts().sort_index()
#Pie Chart Çıkartırsak
plt.figure(figsize=(8, 8))
plt.pie(missing_value_groups, labels=missing_value_groups.index, autopct="%.1f%%", startangle=140)
plt.title("Eksik Değer Oranı Kümeleri")
plt.show()

##Yüzde 90 üzerinde olanlar incelenirse:

missing_df = pd.DataFrame(missing_values).reset_index()

missing_90 = missing_df[missing_df[0]>90]
na_columns2 = missing_90["index"]
missing_vs_target(df_train,"isFraud",missing_90["index"])

plt.figure(figsize=(10, 6))
ax = sns.barplot(y=missing_values.index[:12], x=missing_values.values[:12], hue =missing_values[:12])

# Çubukların üstüne değerleri yazdırma
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%")  # Değerleri yüzde formatında yazdırıyoruz

plt.xticks(rotation=0)
plt.title("%90 Eksik Değer Oranı Olan Değişkenler")
plt.xlabel("Eksik Değer Oranı (%)")
plt.ylabel("Değişkenler")
plt.show()

df_train["addr2"].unique()
cat_summary(df_train, "addr2")


#-- ######// Değişkenler ile devam edelim

# TransactionDT – Numerik
# TransactionAMT – Numerik
# Addr1 – Kardinal
# P_emaildomain, R_emaildomain – Kardinal
# ProductCD – Kategorik
# CARD4, CARD6 – Kategorik
# V12, V13, V36, V53 – Kategorik

###TransactionDT

#İlk Beraber Görselleştirme
plt.figure(figsize=(20, 5))

# Eğitim ve test veri setlerindeki TransactionDT dağılımı
sns.histplot(df_train['TransactionDT'], bins=60, kde=False, color="skyblue", label="Train")
sns.histplot(df_test['TransactionDT'], bins=60, kde=False, color="salmon", label="Test")

# Başlık, eksen adları ve açıklamalar
plt.legend()
plt.ylabel('Frekans')
plt.xlabel('TransactionDT')
plt.title('TransactionDT Özelliğinin Eğitim ve Test Veri Setlerindeki Dağılımı')
plt.show()

#Transaction içine yeni iki değişken eklenmesi:

df_train['day'] = ((df_train['TransactionDT'] //(3600*24)-1)%7)+1
df_test['day'] = ((df_test['TransactionDT'] //(3600*24)-1)%7)+1

df_train['hour'] = ((df_train['TransactionDT']//3600)%24)+1
df_test['hour'] = ((df_test['TransactionDT']//3600)%24)+1


df_train[['day','TransactionDT']].describe()

df_test['day'].describe()


plt.figure(figsize=(10, 5))
ax = sns.countplot(data=df_train, x='day', hue='isFraud', palette=['skyblue', 'salmon'])
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type="edge", padding=3)
# Başlık ve eksen adları
plt.title("Haftanın Günlerine Göre İşlem Türü Dağılımı (isFraud)")
plt.xlabel("Haftanın Günü")
plt.ylabel("İşlem Sayısı")
plt.legend(title="isFraud", labels=["0", "1"])
plt.show()


plt.figure(figsize=(15, 5))
ax = sns.countplot(data=df_train, x='hour', hue='isFraud', palette=['skyblue', 'salmon'])
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type="edge", padding=3)
plt.title("Saatlere Göre İşlem Türü Dağılımı (isFraud)")
plt.xlabel("Saat")
plt.ylabel("İşlem Sayısı")
plt.legend(title="isFraud", labels=["0", "1"])
plt.show()

###TransactionAMT

#Train
num_summary(df_train, "TransactionAmt", False)

plt.figure(figsize=(12, 6))

ax = sns.histplot(df_train['TransactionAmt'], bins=50, kde=False, color="skyblue")
# Çubukların üstüne değerleri ekleme
for p in ax.patches:
    if p.get_height() > 0:  # Yüksekliği 0'dan büyük olanları yazdır
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=8, color='black')
# Başlık ve eksen adları
plt.title("TransactionAmt Train Dağılımı")
plt.xlabel("TransactionAmt Train (İşlem Miktarı)")
plt.ylabel("Frekans")
plt.show()

#Test
num_summary(df_test, "TransactionAmt", False)

plt.figure(figsize=(12, 6))

ax = sns.histplot(df_test['TransactionAmt'], bins=50, kde=False, color="skyblue")
# Çubukların üstüne değerleri ekleme
for p in ax.patches:
    if p.get_height() > 0:  # Yüksekliği 0'dan büyük olanları yazdır
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=8, color='black')
# Başlık ve eksen adları
plt.title("TransactionAmt Test Dağılımı")
plt.xlabel("TransactionAmt Test (İşlem Miktarı)")
plt.ylabel("Frekans")
plt.show()

##########ProductCD – Kategorik

cat_summary(df_train,"ProductCD")
cat_summary(df_test,"ProductCD")
target_summary_with_cat(df_train,"isFraud","ProductCD")

plt.figure(figsize=(18, 5))

# 1. Eğitim veri setinde ProductCD dağılımı
plt.subplot(1, 3, 1)
ax1 = sns.countplot(x='ProductCD', data=df_train, palette="pastel")
for p in ax1.patches:
    ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha='center', va='bottom', fontsize=10)
plt.title("Eğitim Setinde ProductCD Dağılımı")
plt.xlabel("ProductCD")
plt.ylabel("Frekans")

# 2. Test veri setinde ProductCD dağılımı
plt.subplot(1, 3, 2)
ax2 = sns.countplot(x='ProductCD', data=df_test, palette="muted")
for p in ax2.patches:
    ax2.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha='center', va='bottom', fontsize=10)
plt.title("Test Setinde ProductCD Dağılımı")
plt.xlabel("ProductCD")
plt.ylabel("Frekans")

# 3. Eğitim veri setinde hedef değişken (isFraud) bazında ProductCD dağılımı
plt.subplot(1, 3, 3)
ax3 = sns.countplot(x='ProductCD', data=df_train, hue='isFraud', palette=["skyblue", "salmon"])
for p in ax3.patches:
    ax3.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha='center', va='bottom', fontsize=10)
plt.title("Eğitim Setinde isFraud'a Göre ProductCD Dağılımı")
plt.xlabel("ProductCD")
plt.ylabel("Frekans")
plt.legend(title="isFraud", labels=["Dolandırıcılık Değil", "Dolandırıcılık"])

plt.tight_layout()
plt.show()

num_cols

######Addr1, Addr2 – Numerik
cat_summary(df_train,"addr2")
cat_summary(df_test,"addr2")

cat_summary(df_test,"addr1")


plt.figure(figsize = (25,8))

plt.subplot(1,2,1)
sns.distplot(df_train[(df_train['isFraud'] == 0) & (~df_train['addr1'].isnull())]['addr1'])
sns.distplot(df_train[(df_train['isFraud'] == 1) & (~df_train['addr1'].isnull())]['addr1'])
plt.legend(['0','1'])
plt.xticks(np.arange(0, 580, 20))
plt.ylabel('İşlem Yoğunluğu')
plt.title('Eğitim Veri Setinde Purchaser Region (addr1) Dağılımı')

plt.subplot(1,2,2)
sns.distplot(df_test[~df_test['addr1'].isnull()]['addr1'])
plt.xticks(np.arange(0, 580, 20))
plt.ylabel('İşlem Yoğunluğu')
plt.title('Test Veri Setinde Purchaser Region (addr1) Dağılımı')

plt.show()

####P_emaildomain, R_emaildomain – Yüksek Kardinalite

cat_summary(df_train,"P_emaildomain")
cat_summary(df_test,"P_emaildomain")

p_target = target_summary_with_cat(df_train,"isFraud","P_emaildomain")
p_target = p_target.reset_index()
p_target = p_target.sort_values(by="TARGET_MEAN",ascending = False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(y=p_target["P_emaildomain"][:12], x=p_target["TARGET_MEAN"][:12], hue =p_target["TARGET_MEAN"][:12])
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f")
plt.xticks(rotation=0)
plt.title("isFraud Hedef Değişkenin P_emaildomain Kategorik Değişkene Göre Dağılımı")
plt.xlabel("Domain İsimleri")
plt.ylabel("Ortalama Değerler")
plt.show()

cat_summary(df_train,"R_emaildomain")
cat_summary(df_test,"R_emaildomain")

r_target = target_summary_with_cat(df_train,"isFraud","R_emaildomain")
r_target = r_target.reset_index()
r_target = r_target.sort_values(by="TARGET_MEAN",ascending = False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(y=r_target["R_emaildomain"][:12], x=r_target["TARGET_MEAN"][:12], hue =r_target["TARGET_MEAN"][:12])
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f")
plt.xticks(rotation=0)
plt.title("isFraud Hedef Değişkenin R_emaildomain Kategorik Değişkene Göre Dağılımı")
plt.xlabel("Domain İsimleri")
plt.ylabel("Ortalama Değerler")
plt.show()

####DeviceType - Kategorik
missing_values["DeviceType"]

cat_summary(df_train,"DeviceType")

# 1. Eğitim veri setinde ProductCD dağılımı
plt.subplot(1, 3, 1)
ax1 = sns.countplot(x='DeviceType', data=df_train, palette="pastel")
for p in ax1.patches:
    ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha='center', va='bottom', fontsize=10)
plt.title("Eğitim Setinde DeviceType Dağılımı")
plt.xlabel("DeviceType")
plt.ylabel("Frekans")

# 2. Test veri setinde ProductCD dağılımı
plt.subplot(1, 3, 2)
ax2 = sns.countplot(x='DeviceType', data=df_test, palette="muted")
for p in ax2.patches:
    ax2.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha='center', va='bottom', fontsize=10)
plt.title("Test Setinde DeviceType Dağılımı")
plt.xlabel("ProductCD")
plt.ylabel("Frekans")

# 3. Eğitim veri setinde hedef değişken (isFraud) bazında ProductCD dağılımı
plt.subplot(1, 3, 3)
ax3 = sns.countplot(x='DeviceType', data=df_train, hue='isFraud', palette=["skyblue", "salmon"])
for p in ax3.patches:
    ax3.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha='center', va='bottom', fontsize=10)
plt.title("Eğitim Setinde isFraud'a Göre DeviceType Dağılımı")
plt.xlabel("ProductCD")
plt.ylabel("Frekans")
plt.legend(title="isFraud", labels=["0", "1"])

plt.tight_layout()
plt.show()

####DeviceInfo – Yüksek Kardinalite
missing_values["DeviceInfo"]

cat_summary(df_train,"DeviceInfo")
cat_summary(df_test,"DeviceInfo")

device_target = target_summary_with_cat(df_train,"isFraud","DeviceInfo")
device_target = device_target.reset_index()
device_target = device_target.sort_values(by="TARGET_MEAN",ascending = False)
device_target = device_target.reset_index()

plt.figure(figsize=(10, 6))
ax = sns.barplot(y=device_target["DeviceInfo"][410:420], x=device_target["TARGET_MEAN"][410:420], hue =device_target["TARGET_MEAN"][410:420])
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f")
plt.xticks(rotation=0)
plt.title("isFraud Hedef Değişkenin DeviceInfo Kategorik Değişkene Göre Dağılımı")
plt.xlabel("DeviceInfo")
plt.ylabel("Ortalama Değerler")
plt.show()

device_target[device_target["TARGET_MEAN"] ==1].index.min()

device_target.loc[417,:]

#########CARD4, CARD6 – Kategorik
card4_df = pd.DataFrame({"TARGET_MEAN": df_train.groupby("card4")["isFraud"].mean(),
              "COUNT": df_train.groupby("card4")["isFraud"].count()}).reset_index()
cat_summary(df_test,"card4")

# Grafik boyutu
fig, ax1 = plt.subplots(figsize=(10, 6))

# Sol eksen için bar grafiği (COUNT)
bars = ax1.bar(card4_df['card4'], card4_df['COUNT'], color='skyblue', label="COUNT")
ax1.set_xlabel("card4")
ax1.set_ylabel("COUNT", color="skyblue")
ax1.tick_params(axis='y', labelcolor="skyblue")

# Bar grafikteki değerleri üstüne yazdırma
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=10)

# Sağ eksen için çizgi grafiği (TARGET_MEAN)
ax2 = ax1.twinx()
line = ax2.plot(card4_df['card4'], card4_df['TARGET_MEAN'], color="salmon", marker="o", label="TARGET_MEAN")
ax2.set_ylabel("TARGET_MEAN", color="salmon")
ax2.tick_params(axis='y', labelcolor="salmon")

# Çizgi grafikteki değerleri üstüne yazdırma
for i, value in enumerate(card4_df['TARGET_MEAN']):
    ax2.text(i, value, f"{value:.4f}", ha='center', va='bottom', fontsize=10, color="salmon")

# Gösterge (legend) ekleme
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Başlık
plt.title("Card4 - COUNT ve TARGET_MEAN")
fig.tight_layout()
plt.show()

#CARD6
card6_df = pd.DataFrame({"TARGET_MEAN": df_train.groupby("card6")["isFraud"].mean(),
              "COUNT": df_train.groupby("card6")["isFraud"].count()}).reset_index()
cat_summary(df_test,"card6")


# Grafik boyutu
fig, ax1 = plt.subplots(figsize=(10, 6))

# Sol eksen için bar grafiği (COUNT)
bars = ax1.bar(card6_df['card6'], card6_df['COUNT'], color='skyblue', label="COUNT")
ax1.set_xlabel("card6")
ax1.set_ylabel("COUNT", color="skyblue")
ax1.tick_params(axis='y', labelcolor="skyblue")

# Bar grafikteki değerleri üstüne yazdırma
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=10)

# Sağ eksen için çizgi grafiği (TARGET_MEAN)
ax2 = ax1.twinx()
line = ax2.plot(card6_df['card6'], card6_df['TARGET_MEAN'], color="salmon", marker="o", label="TARGET_MEAN")
ax2.set_ylabel("TARGET_MEAN", color="salmon")
ax2.tick_params(axis='y', labelcolor="salmon")

# Çizgi grafikteki değerleri üstüne yazdırma
for i, value in enumerate(card6_df['TARGET_MEAN']):
    ax2.text(i, value, f"{value:.4f}", ha='center', va='bottom', fontsize=10, color="salmon")

# Gösterge (legend) ekleme
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Başlık
plt.title("Card6 - COUNT ve TARGET_MEAN")
fig.tight_layout()
plt.show()

#########V12, V36, V53 – Kategorik

cat_summary(df_test,"V12")
cat_summary(df_test,"V36")
cat_summary(df_test,"V53")

def cat_targetmean_plot(dataframe,analysiscol,target):
    card6_df = pd.DataFrame({"TARGET_MEAN": dataframe.groupby(analysiscol)[target].mean(),
                             "COUNT": dataframe.groupby(analysiscol)[target].count()}).reset_index()

    # Grafik boyutu
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Sol eksen için bar grafiği (COUNT)
    bars = ax1.bar(card6_df[analysiscol], card6_df['COUNT'], color='skyblue', label="COUNT")
    ax1.set_xlabel(analysiscol)
    ax1.set_ylabel("COUNT", color="skyblue")
    ax1.tick_params(axis='y', labelcolor="skyblue")

    # Bar grafikteki değerleri üstüne yazdırma
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=10)

    # Sağ eksen için çizgi grafiği (TARGET_MEAN)
    ax2 = ax1.twinx()
    line = ax2.plot(card6_df[analysiscol], card6_df['TARGET_MEAN'], color="salmon", marker="o", label="TARGET_MEAN")
    ax2.set_ylabel("TARGET_MEAN", color="salmon")
    ax2.tick_params(axis='y', labelcolor="salmon")

    # Çizgi grafikteki değerleri üstüne yazdırma
    for i, value in enumerate(card6_df['TARGET_MEAN']):
        ax2.text(i, value, f"{value:.4f}", ha='center', va='bottom', fontsize=10, color="salmon")

    # Gösterge (legend) ekleme
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Başlık
    plt.title(f"{analysiscol} - COUNT ve TARGET_MEAN")
    fig.tight_layout()
    plt.show()

cat_targetmean_plot(df_train,"V12","isFraud")
cat_targetmean_plot(df_train,"V36","isFraud")
cat_targetmean_plot(df_train,"V53","isFraud")

#####Outlier Analizi

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


#Train Section
for col in num_cols:
    print(col, check_outlier(df_train, col, 0.01, 0.99))

for col in num_cols:
    print(col, check_outlier(df_test, col, 0.01, 0.99))



outlier_percentages = {}
for col in num_cols:
    Q1 = df_train[col].quantile(0.01)
    Q3 = df_train[col].quantile(0.99)
    IQR = Q3 - Q1
    outliers = ((df_train[col] < (Q1 - 1.5 * IQR)) | (df_train[col] > (Q3 + 1.5 * IQR)))
    outlier_percentages[col] = 100 * outliers.sum() / len(df_train)

# En çok outlier olan ilk 20 sütun
sorted_outliers = dict(sorted(outlier_percentages.items(), key=lambda item: item[1], reverse=True)[:20])

plt.figure(figsize=(15, 6))
sns.barplot(x=list(sorted_outliers.keys()), y=list(sorted_outliers.values()))
plt.title("Train Veri Setine Göre En Fazla Aykırı Değer İçeren İlk 20 Sayısal Kolon")
plt.xticks(rotation=90)
plt.ylabel("Outlier Yüzdesi (%)")
plt.show()


df_train_to_Feature = df_train.to_csv("dataset/df_train_to_Feature.csv")
df_test_to_Feature = df_test.to_csv("dataset/df_test_to_Feature.csv")



