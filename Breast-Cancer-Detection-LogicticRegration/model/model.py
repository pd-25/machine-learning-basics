import pandas as pd  
import numpy as np

breast_csv = pd.read_csv('../../datasets/breast-cancer.csv')

# print(breast_csv.head())
# print(breast_csv['diagnosis'].value_counts())

#1. Chech shape
print(breast_csv.shape)
#2. Check null value
print(breast_csv.isnull().sum())
#3. Check. duplicate
print(breast_csv.duplicated().sum())
#4. check info
print(breast_csv.info())


#Now Encode the targeted colum.
#Encoding is needed because the targated column means diagnosis is in string and need to convert it in to numeric. 
# Because this is categorical encoding, we can encode by scikitlearn Labelencoder, but we will use pandas logic here.
