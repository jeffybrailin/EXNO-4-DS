## Name : Jeffy Brailin T
## Reg.No: 212223040076

# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df

```
<img width="369" height="515" alt="image" src="https://github.com/user-attachments/assets/c503668c-90fd-44a9-9159-cde0b83c475f" />

```
df.head()

```
<img width="340" height="261" alt="image" src="https://github.com/user-attachments/assets/18a4f085-d286-4ecf-991d-e94afd5f1235" />

```
df.dropna()

```
<img width="352" height="512" alt="image" src="https://github.com/user-attachments/assets/36f7619a-6ee9-42e8-b040-8ba044e9f4ba" />

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals

```
<img width="117" height="42" alt="image" src="https://github.com/user-attachments/assets/d3f0caf5-94e7-4e5b-aa85-352a6b433ebf" />

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)

```
<img width="383" height="440" alt="image" src="https://github.com/user-attachments/assets/d6dc3f46-1c32-4be1-b30d-af42e30e7be3" />

```

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="386" height="445" alt="image" src="https://github.com/user-attachments/assets/b8ef312d-ecac-41d3-b8c7-19002f770090" />

```
from sklearn.preprocessing import Normalizer
scale=Normalizer()
df[['Height','Weight']]=scale.fit_transform(df[['Height','Weight']])
df.head(10)

```

<img width="366" height="441" alt="image" src="https://github.com/user-attachments/assets/cbc7518f-0598-458b-b754-5398be175c04" />

```
from sklearn.preprocessing import MaxAbsScaler
scalen=MaxAbsScaler()
df[['Height','Weight']]=scalen.fit_transform(df[['Height','Weight']])
df.head(10)

```

<img width="370" height="452" alt="image" src="https://github.com/user-attachments/assets/316cddc6-5c59-40e7-ba7c-8f964492cf8d" />

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)

```

<img width="390" height="438" alt="image" src="https://github.com/user-attachments/assets/0a361975-2c03-494d-9d16-a1bf267eee1d" />

```

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data

```
<img width="1700" height="529" alt="image" src="https://github.com/user-attachments/assets/71837269-8b47-4f12-b09d-758b3ca96c92" />

```
data.isnull().sum()

```

<img width="236" height="326" alt="image" src="https://github.com/user-attachments/assets/b6a08ed9-c1c4-4839-9491-643eaadacb01" />

```
missing=data[data.isnull().any(axis=1)]
missing

```

<img width="1685" height="530" alt="image" src="https://github.com/user-attachments/assets/c8ce4795-b2c4-41c4-8785-96fd38dc0297" />

```
data2=data.dropna(axis=0)
data2

```
<img width="1719" height="509" alt="image" src="https://github.com/user-attachments/assets/c9257165-bcde-4f43-812a-2810b9f791a1" />

```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

```
<img width="1380" height="416" alt="image" src="https://github.com/user-attachments/assets/557f4b91-d1bb-457e-92ac-45090e2b4f84" />

```

sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs

```

<img width="414" height="507" alt="image" src="https://github.com/user-attachments/assets/0c187375-c39f-4ac4-9ed1-6429105dad02" />

```
new_data=pd.get_dummies(data2,drop_first=True)
new_data

```
<img width="1794" height="392" alt="image" src="https://github.com/user-attachments/assets/c76bc727-30f3-4af5-b4cf-373ca15954dd" />

```
columns_list=list(new_data.columns)
print(columns_list)

```
<img width="1797" height="52" alt="image" src="https://github.com/user-attachments/assets/65409e82-df02-4f43-a709-28ac71bd217d" />

```
y=new_data['SalStat'].values
print(y)

```

<img width="212" height="44" alt="image" src="https://github.com/user-attachments/assets/864b3a25-97e5-4bcd-8ba1-c643eb3d4e8a" />

```
x=new_data[features].values
x

```

<img width="688" height="164" alt="image" src="https://github.com/user-attachments/assets/8b9321e5-2f54-4d7b-abbc-cae69ad09559" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)

```
<img width="324" height="88" alt="image" src="https://github.com/user-attachments/assets/190aeafa-fe4a-400b-aa88-a63d63f56c85" />

```
prediction=KNN_classifier.predict(test_x)
confusionMmatrix=confusion_matrix(test_y,prediction)
print(confusionMmatrix)

```
<img width="184" height="62" alt="image" src="https://github.com/user-attachments/assets/bfe8d4b8-282a-4655-abf0-889964f309de" />

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

```
<img width="221" height="31" alt="image" src="https://github.com/user-attachments/assets/1de28b20-b085-4064-8dd9-3d757d6d9895" />

```
print('Misclassified samples: %d' % (test_y != prediction).sum())

```
<img width="294" height="36" alt="image" src="https://github.com/user-attachments/assets/775d38f8-d190-4869-a2e5-af3187660154" />

```
data.shape
```

<img width="134" height="51" alt="image" src="https://github.com/user-attachments/assets/1eb45c49-c2b9-4b59-a0c1-dd5636da8f87" />


```

import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

```

<img width="366" height="65" alt="image" src="https://github.com/user-attachments/assets/358a5b00-43a9-4dbd-b565-5683b8e8ea28" />


```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

```
<img width="532" height="245" alt="image" src="https://github.com/user-attachments/assets/13fceba4-7463-4702-ae8f-14bc7539c3e4" />


```
contigency_table=pd.crosstab(tips['sex'],tips['time'])
print(contigency_table)

```
<img width="246" height="113" alt="image" src="https://github.com/user-attachments/assets/2d48d32c-6d43-454c-8840-f900ca20915e" />

```
chi2,p, _, _ =chi2_contingency(contigency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")

```
<img width="412" height="61" alt="image" src="https://github.com/user-attachments/assets/b84ea04e-6b15-4d94-ae7f-f06bcd62ab0c" />


# RESULT:
Thus perform Feature Scaling and Feature Selection process and save the data to a file successfully.

## Inference:

In this experiment, I done cleaning and transforming raw data through a series of well chosen preprocessing steps to build robust machine learning models. The workflow incorporates feature scaling methods—StandardScaler, MinMaxScaler, RobustScaler, and others to harmonize the range and distribution of features, minimizing bias and improving training stability. Feature selection techniques, including the use of the chi-square test, help identify the most relevant variables, which streamlines the model and enhances interpretability. Chi-square specifically measures the association between categorical variables, guiding the inclusion of features that have the greatest impact on predictions. Incorporating these processes ensures that the final model is not only more accurate but also computationally efficient, making it well suited for real-world data science tasks.
