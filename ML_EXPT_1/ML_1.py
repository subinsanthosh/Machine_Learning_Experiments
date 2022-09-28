
#EXPERIMENT NO: 1
#Boston_Housing
#Multiple Linear Regression


# Using all the features:

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BostonTrain = pd.read_csv("./BostonHousing.csv")
BostonTrain.head()

	crim	  zn	indus  chas	 nox	 rm	    age	    dis   rad	tax	 ptratio   b	 lstat	medv
0	0.00632	18.0	2.31	0	0.538	6.575	65.2  4.0900   1	296	  15.3	 396.90  4.98	24.0
1	0.02731	 0.0	7.07	0	0.469	6.421	78.9  4.9671   2	242	  17.8	 396.90	 9.14	21.6
2	0.02729	 0.0	7.07	0	0.469	7.185	61.1  4.9671   2	242	  17.8	 392.83	 4.03	34.7
3	0.03237	 0.0	2.18	0	0.458	6.998	45.8  6.0622   3	222	  18.7	 394.63	 2.94	33.4
4	0.06905	 0.0	2.18	0	0.458	7.147	54.2  6.0622   3	222	  18.7	 396.90	 5.33	36.2

BostonTrain.info()
BostonTrain.describe()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 506 entries, 0 to 505
Data columns (total 14 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   crim     506 non-null    float64
 1   zn       506 non-null    float64
 2   indus    506 non-null    float64
 3   chas     506 non-null    int64  
 4   nox      506 non-null    float64
 5   rm       506 non-null    float64
 6   age      506 non-null    float64
 7   dis      506 non-null    float64
 8   rad      506 non-null    int64  
 9   tax      506 non-null    int64  
 10  ptratio  506 non-null    float64
 11  b        506 non-null    float64
 12  lstat    506 non-null    float64
 13  medv     506 non-null    float64
dtypes: float64(11), int64(3)
memory usage: 55.5 KB

	      crim	        zn	        indus	    chas    	 nox	      rm	     age	  
count  506.000000   506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	
mean     3.613524	11.363636	 11.136779	  0.069170	  0.554695	  6.284634	 68.574901	 
std	     8.601545	23.322453	  6.860353	  0.253994	  0.115878	  0.702617	 28.148861	  
min	     0.006320	 0.000000	  0.460000    0.000000	  0.385000	  3.561000	  2.900000
25%	     0.082045	 0.000000	  5.190000	  0.000000	  0.449000	  5.885500	 45.025000 
50%	     0.256510	 0.000000	  9.690000	  0.000000	  0.538000	  6.208500	 77.500000	
75%	     3.677083	12.500000	 18.100000	  0.000000	  0.624000	  6.623500	 94.075000
max	   88.976200   100.000000	 27.740000	  1.000000	  0.871000	  8.780000	100.000000

             dis	      rad	      tax	      ptratio	     b	        lstat     	medv
count    506.000000	 506.000000   506.000000	506.000000	506.000000	506.000000	506.000000
mean       3.795043	   9.549407	  408.237154	 18.455534	356.674032	 12.653063	 22.532806
std        2.105710	   8.707259	  168.537116	  2.164946	 91.294864	  7.141062	  9.197104
min	       1.129600	   1.000000	  187.000000	 12.600000	  0.320000	  1.730000    5.000000
25%        2.100175	   4.000000	  279.000000	 17.400000	375.377500	  6.950000	 17.025000
50%        3.207450	   5.000000	  330.000000	 19.050000	391.440000	 11.360000	 21.200000
75%  	   5.188425	  24.000000	  666.000000	 20.200000	396.225000	 16.955000	 25.000000
max 	  12.126500	  24.000000	  711.000000	 22.000000	396.900000	 37.970000	 50.000000

BostonTrain.plot.scatter('rm', 'medv')

plt.subplots(figsize=(12,8))
sns.heatmap(BostonTrain.corr(), cmap = 'RdGy')

sns.pairplot(BostonTrain, vars = ['lstat', 'ptratio', 'indus', 'tax', 'crim', 'nox', 'rad', 'age', 'medv'])

sns.pairplot(BostonTrain, vars = ['rm', 'zn', 'b', 'dis', 'chas','medv'])

X = BostonTrain[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'b', 'lstat']]
y = BostonTrain['medv']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lm = LinearRegression()
lm.fit(X_train,y_train)

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

Text(0, 0.5, 'Predicted Y')

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

MAE: 3.1292691433488975
MSE: 18.8981061529637
RMSE: 4.347195205297744


#By droping irrelevant columns (Feature Engineering)

X1 = BostonTrain[['indus', 'chas', 'nox', 'rm', 'dis', 'rad', 'tax',
       'ptratio', 'b', 'lstat']]
y1 = BostonTrain['medv']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3)
lm = LinearRegression()
lm.fit(X1_train,y1_train)

LinearRegression()
predictions = lm.predict(X1_test)

plt.scatter(y1_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

Text(0, 0.5, 'Predicted Y')

print('MAE:', metrics.mean_absolute_error(y1_test, predictions))
print('MSE:', metrics.mean_squared_error(y1_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y1_test, predictions)))

MAE: 3.3560066685592607
MSE: 22.960618010410197
RMSE: 4.791723907990756