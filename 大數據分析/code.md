## 0601
```
import statistics
data1=[1,3,4,5,7,9,2]

x=statistics.mean(data1)
y=statistics.variance(data1)
z=statistics.stdev(data1)


print("Mean is:",x)
print("variance is:",y)
print("standard deviation of data:",z)
```
```
Mean is: 4.428571428571429
variance is: 7.9523809523809526
standard deviation of data: 2.819996622760558
```
## 0602
```
import numpy as np
import statistics

y=[1,3,4,5,7,9,2]

total=sum((y-np.mean(y))**2)


print(y)
print("平均數",statistics.mean(y))
print("離差平方和 :",total)
```
```
[1, 3, 4, 5, 7, 9, 2]
平均數 4.428571428571429
離差平方和 : 47.714285714285715
```
## 0603
```


import numpy as np
import statistics as sts

score=[31,24,23,25,14,25,13,12,14,23,
       32,34,43,41,21,23,26,26,34,42,
       43,25,24,23,24,44,23,14,52,32,
       42,44,35,28,17,21,32,42,12,34]

print('求和:',np.sum(score))
print('個數:',len(score))
print('平均數:',np.mean(score))
print('中位數:',np.median(score))
print('眾數:',sts.mode(score))
print('上四分位數:',sts.quantile(score,p=0.25))
print('下四分位數:',sts.quantile(score,p=0.75))

print('最大值:',sts.max(score))
print('最小值:',sts.min(score))
print('極差:',np.max(score)-np.min(score))
print('四分位差:',sts.quantitle(score,p=0.75)-sts.quantitle(score,p=0.25))
print('標準差:',np.std(score))
print('方差:',np.var(score))
print('離散係數:',np.std(score)/np.mean(score))

print('偏度:',sts.skewness(score))
print('緯度:',sts.kurtosis(score))
```
![擷取](https://user-images.githubusercontent.com/71476327/226266085-192f92be-80ac-4ebe-93e8-1b7b08fcbeac.PNG)

## 1505
```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston

boston_dataset=load_boston()

boston=pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)
boston['MEDV']=boston_dataset.target
boston
```
![擷取](https://user-images.githubusercontent.com/71476327/224634842-9609bec1-a0be-4578-8f6a-fd81a55e7904.PNG)
```
plt.figure(figsize=(2,5))
plt.boxplot(boston['LSTAT'],showmeans=True)
plt.title('LSTAT')
plt.show
```
![擷取](https://user-images.githubusercontent.com/71476327/224641505-338571de-2582-43e4-9d38-8869a05cf73a.PNG)
## DA_ex1602
```
import numpy as np

points= np.random.normal(27000,15000,10000)
np.mean(points)

import matplotlib.pyplot as plt
plt.hist(points, 50)
plt.show()

np.mean(points)
np.median(points)

points2=np.append(points,[1000000000])
np.mean(points2)
np.median(points2)
plt.hist(points2,50)
plt.show()
```
![擷取](https://user-images.githubusercontent.com/71476327/226263896-eaf0e5a6-956b-4143-89ff-3a3c714f4371.PNG)

## 5305
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import StanderScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris_df = pd.read_csv('https://archive.ics.uci.edu/ml/mac')

X = iris_df.iloc[:,:-1].values
y = iris_df.iloc[:,-1].values

def calc_cod(data):
    return np.std(data)/np.mean(data)

cod1 = calc_cod(X[:,0])
cod2 = calc_cod(X[:,1])

print("第一個屬性的COD:",cod1)
print("第二個屬性的COD:",cod2)

scaler = StanderScaler()
X_norm = scaler.fit_transform(X)

X_train,X_test,y_test = train_test_spilt(X_norm,y,test_size=0.2,random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

score = knn.score(X_test,y_test)
print("準確率得分:",score)
```
