
# data imported from https://www.kaggle.com/uciml/pima-indians-diabetes-database




#@title
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn


# In[ ]:

# ## LOAD CSV DATA 
#@title
from google.colab import files

uploaded = files.upload()


# 

# 


df = pd.read_csv("diabetes.csv")


# ## SHOW 5 FIRST RECORD

# In[ ]:


df.head(5)


# ## SHOW 5 LAST RECORD

# In[ ]:


df.tail(5)


# ## CHANGE COLUMN EACH FEATURES

# In[ ]:


df.rename(columns={'Pregnancies':'pregnancies', 
                         'Glucose':'glucose',
                         'BloodPressure':'blood_pressure',
                         'SkinThickness':'skin_thickness', 'Insulin':'insulin',
                         'BMI':'bmi',
                         'DiabetesPedigreeFunction':'dpf',
                         'BMI':'bmi',
                         'Age':'age',
                         'Outcome':'target'}, inplace = True)


# ## CHANGE VALUE TO HEALTHY -> 0 AND DIABETIC -> 0 EACH DATA

# In[ ]:


target = df['target']
diabetes_baru = []
for item in target:
    if item == 0:
        diabetes_baru.append('Healthy')
    else:
        diabetes_baru.append('Diabetic')
diabetes_baru = pd.DataFrame(diabetes_baru)
df['diagnosis'] = diabetes_baru
df.loc[0:,'diagnosis':]


# In[ ]:


df.head()


# ## SHOW SUMMARY OF DATA

# In[ ]:


df.info()


# ## SHOW STATISTIC

# In[ ]:


df.describe()


# ## VISUALIZE WITH SCATTER PLOT

# In[ ]:


D = df[df.diagnosis == "Diabetic"]
H = df[df.diagnosis == "Healthy"]

plt.title("scatter plot")
plt.xlabel("glucose")
plt.ylabel("blood pressure")
plt.scatter(D.glucose, D.blood_pressure, color = "red", label = "Diabetic")
plt.scatter(H.glucose, H.blood_pressure, color = "lime", label = "Healthy")
plt.legend()
plt.show()


# ## SHOW DATA DISTRIBUTION WITH HISTOGRAM

# In[ ]:


import numpy as np

df.hist()


# ## CREATE NEW COLUMN ACCORDING TO "diabetic_category"

# In[ ]:


diabetes_kategori = df['glucose']
df_baru = []
for item in diabetes_kategori:
    if(item in range(0,141)):
        df_baru.append('Normal')
    elif(item in range(141,201)):
        df_baru.append('Prediabetic')
    elif(item > 200):
        df_baru.append('Diabetic')
df['diabetic_category'] = df_baru
df.loc[:,"diabetic_category"]


# In[ ]:


df


# ## GET COUNT OF EACH DIAGNOSIS AND DIABETIC_CATEGORY

# In[45]:



df.value_counts(["diagnosis", "diabetic_category"])


# ## CHECK MISSING VALUE

# In[ ]:


df.isnull()


# ## REMOVE OUTLIER WITH INTERQUARTILE APPROACH

# In[ ]:


def remove_outlier_IQR(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    df_final=df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR)))]
    return df_final
 
df_outlier_removed=remove_outlier_IQR(df)
df_outlier_removed=pd.DataFrame(df_outlier_removed)
ind_diff=df.index.difference(df_outlier_removed.index)

for i in range(0, len(ind_diff),1):
    df_final=df.drop([ind_diff[i]])
    df=df_final


# In[ ]:


df.describe()


# ## NORMALIZATION WITH MINMAX

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df_scaled = scaler.fit_transform(df[['pregnancies',
                                    'glucose',
                                    'blood_pressure',
                                    'skin_thickness',
                                    'insulin',
                                    'bmi',
                                    'dpf',
                                    'age',
                                    'target']].to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=['pregnancies',
                                             'glucose',
                                             'blood_pressure',
                                             'skin_thickness',
                                             'insulin',
                                             'bmi',
                                             'dpf',
                                             'age',
                                             'target'])
 
print("Scaled Dataset Using MinMaxScaler")
df_scaled.head()


# In[ ]:


df_scaled


# ## SPLIT DATA TO 80% AND 20% FOR TRAINING AND TEST DATA

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklear.metrics import confusion_matrix
X = df_scaled
y = df_scaled.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[ ]:


y_train


# ## CLASSIFY WITH NAIVE BAYES GAUSSIAN

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklear.metrics import confusion_matrix
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)*100
print(accuracy,'%')

