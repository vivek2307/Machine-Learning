import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Car_sales.csv')

data.head()
data.info()

data_obj=[]
data_float_int=[]
for i in data:
    if((data[i]).dtypes =='object' ):
        data_obj.append(i)
    else:
        data_float_int.append(i)
print(data_obj)
print(data_float_int)

data.isnull().sum()

#distribution
data["__year_resale_value"].plot.hist()
#car.hist()
data["__year_resale_value"].fillna(data["__year_resale_value"].median(),inplace=True)

data.isnull().sum()


#####################################################################################################

"""to predict the car sales amount.."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('Car_sales.csv')

data.head()
data.info()
data.columns
data.isnull().sum()

data['__year_resale_value'].fillna(data['__year_resale_value'].median(),inplace=True)

data['Price_in_thousands'].fillna(data['Price_in_thousands'].median(),inplace=True)

data['Engine_size'].fillna(data['Engine_size'].median(),inplace=True)

data['Horsepower'].fillna(data['Horsepower'].median(),inplace=True)

data['Wheelbase'].fillna(data['Wheelbase'].median(),inplace=True)

data['Length'].fillna(data['Length'].median(),inplace=True)

data['Width'].fillna(data['Width'].median(),inplace=True)

data['Curb_weight'].fillna(data['Curb_weight'].median(),inplace=True)

data['Fuel_capacity'].fillna(data['Fuel_capacity'].median(),inplace=True)

data['Fuel_efficiency'].fillna(data['Fuel_efficiency'].median(),inplace=True)

data['Power_perf_factor'].fillna(data['Power_perf_factor'].median(),inplace=True)

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

data['Manufacturer'].value_counts
data['Manufacturer']= lb.fit_transform(data['Manufacturer'])

data['Model'].value_counts
data['Model']= lb.fit_transform(data['Model'])

data['Vehicle_type'].value_counts
data['Vehicle_type']= lb.fit_transform(data['Vehicle_type'])

#data['Latest_Launch'].value_counts
#data['Latest_Launch']= lb.fit_transform(data['Latest_Launch'])

data['Latest_Launch'] = pd.to_datetime(data['Latest_Launch'])
data = data.set_index(data['Latest_Launch'])
data = data.sort_index()


train = data['2008-02-13':'2012-05-30']
test  = data['2012-06-01':]
print('Train Dataset:',train.shape)
print('Test Dataset:',test.shape)

train.drop('Latest_Launch',axis=1,inplace=True)
test.drop('Latest_Launch',axis=1,inplace=True)

x_1 = train.iloc[:, [0,1,2,4,5,6,7,8,9,10,11,12,13,14]]
y_1 = train.iloc[:,3]

#x_test = test.iloc[:, [0,1,2,4,5,6,7,8,9,10,11,12,13,14]]
#y_test = test.iloc[:,3]


#x_train.isnull().sum()
#y_train.isnull().sum()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_1,y_1)

from sklearn.preprocessing import StandardScaler
sd=StandardScaler()
ft=sd.fit_transform(x_train,y_train)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

lr.score(x_train,y_train)


predict = lr.predict(x_train)






































