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
