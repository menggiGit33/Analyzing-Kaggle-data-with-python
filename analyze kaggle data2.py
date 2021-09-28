
 ## data imported from https://www.kaggle.com/ishandutta/early-stage-diabetes-risk-prediction-dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from google.colab import files

uploaded = files.upload()

"""## LOAD FILES"""

df = pd.read_csv('diabetes_data_upload.csv')
df

"""## DISPLAY DATA"""

df.info()

"""## DISPLAY SUMMARY"""

df.describe()

"""## COUNT VALUE EACH CLASS"""

df['class'].value_counts()

df['Gender'].value_counts()

df.head()

"""## CREATE BAR CHART VISUALIZATION"""

df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})
df['Polyuria'] = df['Polyuria'].map({'Yes':1, 'No':0})
df['Polydipsia'] = df['Polydipsia'].map({'Yes':1, 'No':0})
df['sudden weight loss'] = df['sudden weight loss'].map({'Yes':1, 'No':0})
df['weakness'] = df['weakness'].map({'Yes':1, 'No':0})
df['Polyphagia'] = df['Polyphagia'].map({'Yes':1, 'No':0})
df['visual blurring'] = df['visual blurring'].map({'Yes':1, 'No':0})
df['Genital thrush'] = df['Genital thrush'].map({'Yes':1, 'No':0})
df['Itching'] = df['Itching'].map({'Yes':1, 'No':0})
df['Irritability'] = df['Irritability'].map({'Yes':1, 'No':0})
df['delayed healing'] = df['delayed healing'].map({'Yes':1, 'No':0})
df['partial paresis'] = df['partial paresis'].map({'Yes':1, 'No':0})
df['muscle stiffness'] = df['muscle stiffness'].map({'Yes':1, 'No':0})
df['Alopecia'] = df['Alopecia'].map({'Yes':1, 'No':0})
df['Obesity'] = df['Obesity'].map({'Yes':1, 'No':0})
df['class'] = df['class'].map({'Positive':1, 'Negative':0})

df.groupby('Gender')['Polyuria'].value_counts().unstack(0).plot.bar()

df.groupby('Gender')['class'].value_counts().unstack(0).plot.bar()

df.groupby('class')['partial paresis'].count().plot.bar(color=['C0', 'C1'])

df.groupby('class')['Genital thrush'].count().plot.bar(color=['C0', 'C1'])

"""## MAKE HEATMAP VISUALIZATION"""

## Correlation matrix
plt.figure(figsize=(20, 10))
heatmap = sns.heatmap(df.corr(),annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);

"""## MODELING DATA WITH KNN"""

# a. splitting data

from sklearn.model_selection import train_test_split
X = df
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

X_test

X_train

y_test

y_train

#b. feature scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#c. KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(metric='euclidean')
classifier.fit(X_train, y_train)

#d. Get y prediction
y_pred = classifier.predict(X_test)

#e. Evaluate
from sklearn.metrics import classification_report, confusion_matrix

# confusion matrix
print(confusion_matrix(y_test, y_pred))

# classification report
print(classification_report(y_test, y_pred))

"""## SEARCHING BEST K VALUE"""

#a. comparing error rate with k value
error = []

for i in range(1, 14):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

error

#b. input best k value and get kkn score
plt.figure(figsize=(12, 6))
plt.plot(range(1, 14), 
         error, 
         color='red', 
         linestyle='dashed', 
         marker='o',
         markerfacecolor='blue', 
         markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

knn.score(X_test, y_test)