
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_spilt
from sklearn.linear_model  import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

data = {
        'house_area':np.random.randint(50, 200, size=100),
        'bedroom':np.random.randint(1, 5, size=100),
        'price':np.random.randint(100000, 1000000, size=100),
}

df = pd.DataFrame(data)

cov_matrix = df.cov()
print("Covariance Matrix:\n", cov_matrix)

x = df[['house_area', 'bedroom']]
y=df['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

plt.scatter(x_test['house_area'], y_test, color='blue', lable='Actual Price')
plt.scatter(x_test['house_area'], y_pred, color='red', lable='Predicted Price')
plt.xlabel('House Area')
plt.ylabel('House Price')
plt.title('House Price vs House Area')
plt.legend()
plt.show
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

wine = datasets.load_wine()
df =pd.DataFrame(data=wine.data, columns=feature_names)
df["target"] = wine.target

df.head()
df.tail()
df.shape()
df.describe()
df.info()
df.dtypes()
df.isna()
df.duplicated().sum()

missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values)
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

wine = datasets.load_wine()
df = pd.DataFrame(data=wine.data,columns=wine.feature_names)
df["target"] = wine.target
df.rename(columns={"od280/od315_of_diluted_wines":"protein_concentration"},inplace=True)



df['target'].value_counts().plot(kind="bar",color="#FF5809",alpha=1)
plt.grid(axis="both", linestyle="-", alpha=0.5)
plt.title("Value counts of the target variable")
plt.xlabel("Wine type",fontstyle='italic')
plt.xticks(rotation=0)
plt.ylabel("Count")


plt.show()

```
```
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

wine = datasets.load_wine()
df = pd.DataFrame(data=wine.data,columns=wine.feature_names)
df["target"] = wine.target
df.rename(columns={"od280/od315_of_diluted_wines":"protein_concentration"},inplace=True)

magnesium_data = df['magnesium']
magnesium_data.hist()

print(f"Skewness::{df['magnesium'].skew()}")
print(f"Skewness::{df['magnesium'].kurt()}")

print(f"原始偏態:{df['magnesium'].skew()}")

df['magnesium_log'] = np.log1p(df['magnesium'])

print(f"log變換後偏態:{df['magnesium_log'].skew()}")

plt.figure(figsize=(12,6))

plt.subplot(1, 2, 1)
sns.histplot(df['magnesium'], bins=30, color='blue')
plt.title('Original Magnesium')

plt.subplot(1, 2, 2)
sns.histplot(df['magnesium_log'], bins=30, color='green')
plt.title('Log Transformed Magnesium')
plt.show()
```
```
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from matplotlib import rcParams

wine = load_wine()
df = pd.DataFrame(data=wine.data,columns=wine.feature_names)
df["target"] = wine.target

df.target.value_counts()
df.target.value_counts(normalize=True)

x = df.drop('target', axis=1)
y =df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print(classification_report(y_test, predictions))


```
```
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from matplotlib import rcParams

wine = load_wine()
df = pd.DataFrame(data=wine.data,columns=wine.feature_names)
df["target"] = wine.target

df.head()
df.tail()
df.shape
df.describe()
df.info()
df.dtypes
df.isna()
df.duplicated().sum()

df.rename(columns={"od280/od315_of_diluted_wines":"protein_connection"})

df.target.value_counts()
df.target.value_counts(normalize=True)


Q1=df.quantile(0.25)
Q3=df.quantile(0.75)
IQR =Q3-Q1

lower_bound=Q1-1.5*IQR
upper_bound=Q1+1.5*IQR

outlier_counts=((df<lower_bound)|(df>upper_bound)).sum()

print(outlier_counts)
```
```
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from matplotlib import rcParams

wine = load_wine()
df = pd.DataFrame(data=wine.data,columns=wine.feature_names)
df["target"] = wine.target

df=df.drop('magnesium',axis=1)
sns.pairplot(df)
```
```
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from matplotlib import rcParams

wine = load_wine()
df = pd.DataFrame(data=wine.data,columns=wine.feature_names)
df["target"] = wine.target



corrmat=df.corr()
hm=sns.heatmap(corrmat, cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':6},
               yticklabels=df.columns,xticklabels=df.columns,cmap="Spectral_r")
plt.show()
```
```
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from matplotlib import rcParams

wine = load_wine()
df = pd.DataFrame(data=wine.data,columns=wine.feature_names)
df["target"] = wine.target


df=df.drop('magnesium',axis=1)

corrmat=df.corr()
hm=sns.heatmap(corrmat, cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':6},
               yticklabels=df.columns,xticklabels=df.columns,cmap="Spectral_r")
plt.show()

sns.catplot(x="target",y="proline",data=df,kind="box",aspect=1.5)
plt.title("Boxplot for target vs proline")
plt.show()
```
