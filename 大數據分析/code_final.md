
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
