import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap #Çıkan sonuçları görselleştirmek için.

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV #KNN ile ilgili best parametreli bulurken kullanılır.
from sklearn.metrics import accuracy_score, confusion_matrix #matrisler kullanılacak nerede hata olduğunu anlamak için
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

#warning library
import warnings
warnings.filterwarnings("ignore")  # warningleri görmezden gelmek için.

# veri setinin içeriği eğitime uygun hale getirilmesi
data = pd.read_csv("cancer.csv")        #Pandas kütüphanesi ile csv dosyası okunur.
data.drop(['Unnamed: 32','id'], inplace = True, axis = 1 )
#20.Satırda veri seti kolonlarının sonunda bir tane fazlalık silindi. 569,33'ten 569,31'e indirgendi
#İstenmeyen featurelar çıkarıldı.  
data= data.rename(columns = {"diagnosis" : "target"}) # Diagnosis kolonun ismi target olarak değiştirildi.

sns.countplot(data["target"]) # Verideki iyi huylu ve kötü huylu hücre sayısını grafiğe döker.
print(data.target.value_counts())  #data.target.value_counts ile grafik ve konsola değerleri yazdırdık.
data["target"]= [1 if i.strip() == "M" else 0 for i in data.target] # Target kısmındaki iyi huylulara 0 kötü huylulara 1 atandı. Strip ile başta boşluk olmasına karşı önlem alındı.

# Veri setinin özet olarak konsolda incelenmesi
print(len(data))
print(data.head())
print("Data shape ",data.shape) #Veri uzunluğu, verinin ilk 5 satırı ve satır sütun sayıları konsola yazdırıldı.
data.info() # Mising value tespiti için kullanılır. Bizim datamızda mising value bulunmamaktadır.
describe = data.describe()

# Verinin detaylı analizi EDA 

# Korelasyon
corr_matrix = data.corr() # Verilerin içierisindeki nümerik değerler arasındaki korelasyona bakılır. String olsaydı discard edilecekti ancak verimizde string bulunmamakta.
sns.clustermap(corr_matrix, annot = True, fmt = ".2f") # Taşan float kısımlarından sadece 2 tanesi gösterildi.
plt.title("Featurelar Arasındaki Korelasyon")
plt.show()

threshold = 0.5
filtre = np.abs(corr_matrix["target"]) > threshold  # 0.75'ten büyük değerleri almak istiyoruz, targetler ile karşılaştırdı.
corr_features = corr_matrix.columns[filtre].tolist() # Matriksteki kolonları al filtreyi uygula ve listeye çevir.
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f") # Fazlılıklar korelasyondaki özellikler içinde uygulanır.
plt.title("Değeri 0.5'ten büyük olan korelasyon featureları")


# Box plot çizimi
data_melted = pd.melt(data, id_vars = "target",
                      var_name = "features",
                      value_name = "value")
plt.figure()
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90) # Feature isimleri 90 derece açıyla yazdırılır.
plt.show()


# Standardization ve normalization
# Pait plot çizimi
sns.pairplot(data[corr_features], diag_kind = "kde", markers = "+", hue= "target") 
plt.show()  #  Çizim yapıldı ve skewneslar tespit edildi. 


#Skewness 
#Outlier

y = data.target # x feature y ise target.
x = data.drop(["target"],axis = 1 ) 
columns = x.columns.tolist()

clf = LocalOutlierFactor()
y_pred = clf.fit_predict(x) # outlier için -1 , inlier için 1 değeri verilir.
X_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()
outlier_score["score"] = X_score # Score değerini görebilmek için.

#threshold  
threshold = -2.5
filtre = outlier_score["score"] < threshold # -2.5'dan büyükleri almayarak filtreleme yaptık.
outlier_index = outlier_score[filtre].index.tolist() # filtre uygulanmıs hali listeye cevirildi.

plt.figure()
plt.scatter(x.iloc[outlier_index,0], x.iloc[outlier_index,1], color = "blue", s = 50, label = "Outliers" )
plt.scatter(x.iloc[:,0], x.iloc[:,1], color = "k", s = 3, label = "Data Points")

radius = (X_score.max() - X_score)/(X_score.max() - X_score.min())
outlier_score["radius"] = radius # Radius değerini görebilmek için.
plt.scatter(x.iloc[:,0], x.iloc[:,1], s = 1000*radius, edgecolors = "r", facecolors = "none", label = "Outlier Skorları")
plt.legend()
plt.show() # Büyük çemberler outlier olmaya en yakın adaylar, -3 değerimiz en sağdaki büyük çember gibi ..

#drop outliers
x = x.drop(outlier_index) # x data frame
y = y.drop(outlier_index).values # y ise arraye çevrildi.

#Train test split (içerisinde default olarak shuffle var. Veri shuffle edilerek train ve test olarak ayrılır.)
test_size = 0.3 # Ne kadar train ne kadar test yapılacağını verir, x ve y için.
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = test_size, random_state = 42) # Bu fonksiyon baştaki dörtlüyü return eder.

#Standardization (Datadaki veriler arasında çok fazla fark olduğundan yapıldı. Ör: 1000 ve 0.01 değerleri gibi)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) #Train verisi fit edilip eğitilmiş scale edilir
X_test = scaler.transform(X_test) # Hali hazırda eğitilmiş scaler olduğu için fit edilmeye gerek yok, sadece transform edilir.

X_train_df = pd.DataFrame(X_train, columns = columns)  # Standardization işlemi tamamlandı, Örneğin 600 olan veri 3'e indirgendi.
X_train_df_describe = X_train_df.describe()
X_train_df["target"] = Y_train  #Standardization işlemi sonucunda featurların meanı 0, standart sapması ise 1 olur.

# Box plot çizimi
data_melted = pd.melt(X_train_df, id_vars = "target",
                      var_name = "features",
                      value_name = "value")
plt.figure()
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90) # Feature isimleri 90 derece açıyla yazdırılır.
plt.show()

# Pait plot çizimi
sns.pairplot(X_train_df[corr_features], diag_kind = "kde", markers = "+", hue= "target") 
plt.show()  #  Çizim yapıldı ve skewneslar tespit edildi. 



# Temel KNN Metodu 

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test) # Xtest üzerinde knn algoritmasını kullanarak prediction gerçekleştirir.
cm = confusion_matrix(Y_test, y_pred) #Ytest ve Ypred'i karşılaştırarak bir confusion matrixi ortaya çıkarır.
acc = accuracy_score(Y_test, y_pred)
# print("Confusion matrix: ", cm)
# print ("Knn algoritmasının accuracy : ", acc) # Doğruluk yüzdesi sonucu ekrana basılır.


# En iyi parametreyi seçme

def KNN_Best_Params(x_train, x_test, y_train, y_test):
    
    k_range = list(range(1,31))
    weight_options = ["uniform","distance"] # Bu iki parametreden en uygun olanını bulmaya calısıyoruz.
    print()
    param_grid = dict(n_neighbors = k_range, weights = weight_options)  # Grid search için gerekli parametreleri dictionary içerisine koyduk
    
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv = 10, scoring = "accuracy") # ML olarak knn seçildi parametre olarak ise param_grid kullan, cross val over fitting engellemek için.
    grid.fit(x_train, y_train) # fit etmek için x ve y train kullanılıyor
    
    # print("En iyi egitim skoru : {} parametreler : {}".format(grid.best_score_, grid.best_params_))
    print()
    
    knn = KNeighborsClassifier(**grid.best_params_) # Yukarıda elde edilen en iyi parametrelerin kullanılması istendi.
    knn.fit (x_train, y_train)  # Burada knn.fit yerinde direk grid.fit yazabilirdik.
    
    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    
    acc_test  = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    
    print ("Test skoru : {}, Egitim skoru : {}".format(acc_test, acc_train))
    print()
    print("CM Test : ", cm_test)
    print("CM Egitim : ", cm_train)
    
    
    return grid
    
grid = KNN_Best_Params(X_train, X_test, Y_train, Y_test)

