import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

breast_csv = pd.read_csv('../../datasets/breast-cancer.csv')

# print(breast_csv.head())
print(breast_csv['diagnosis'].value_counts())

# #1. Chech shape
# print(breast_csv.shape)
# #2. Check null value
# print(breast_csv.isnull().sum())
# #3. Check. duplicate
# print(breast_csv.duplicated().sum())
# #4. check info
# print(breast_csv.info())


#Now Encode the targeted colum.
#Encoding is needed because the targated column means diagnosis is in string and need to convert it in to numeric. 
# Because this is categorical encoding, we can encode by scikitlearn Labelencoder, but we will use pandas logic here.
breast_csv['diagnosis'] = breast_csv['diagnosis'].map({
    'M': 1,
    'B': 0
})

# print(breast_csv['diagnosis'].value_counts())

#Now splitting data into Training and testing set

X = breast_csv.drop('diagnosis', axis=1)
y = breast_csv['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(X_train.shape)
# print(X_test.shape)

#Now we need to do feature scaling

scalar = StandardScaler()

scalar.fit(X_train)

X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

print(X_train[10])


#Train the model usiing logiistic regretion

from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()

logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)

#Now chec the score

print(accuracy_score(y_test, y_pred))

imput_text =(  -0.23711093, -0.4976419,   0.61365274, -0.49813131, -0.53102815, -0.57694824,
                -0.17494424, -0.36215622, -0.284859,    0.43345165,  0.17818232, -0.36844966,
                0.55310406, -0.31671104, -0.40524636,  0.04025752, -0.03795529, -0.18043065,
                0.16478901, -0.12170969,  0.23079329, -0.50044002,  0.81940367, -0.46922838,
                -0.53308833, -0.04910117, -0.04160193, -0.14913653,  0.09681787,  0.10617647,
                0.49035329,
            )

np_df = np.asarray(imput_text)
prediction = logistic.predict(np_df.reshape(1,-1))

# print(prediction[0])
if prediction[0] == 1 :
    print('Cancrous')
else:
    print('Not Cancorus')


pickle.dump(logistic, open('logistic_model.pkl', 'wb'))







