#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Importing Libraries


import pandas as pd
import glob
import os
import zipfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


#This code defines a function called read_zipfile that takes a single argument filepath.
#The function expects filepath to be a path to a zip file that contains several CSV files, and it reads the CSV files 
#inside the zip file using the pandas library.

#The function renames the columns of the CSV files to more descriptive names and joins the data frames using a left join 
#on their timestamps. The resulting data frame is returned.

#The code then defines a list of folders, and for each folder it finds all the zip files inside it using the glob library.
#It then applies the read_zipfile function to each zip file to get a list of data frames, and concatenates them using 
#pd.concat. The resulting data frame is stored in a variable called df.

#Finally, the code removes any rows where ibo1 column has the value of 99999.


def read_zipfile(filepath):
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall()
        accelerometer = pd.read_csv('ACC.csv', index_col=None, header=0)
        blood_vol = pd.read_csv('BVP.csv', index_col=None, header=0)
        eda = pd.read_csv('EDA.csv', index_col=None, header=0)
        heart_rate = pd.read_csv('HR.csv', index_col=None, header=0)
        ibi = pd.read_csv('IBI.csv', index_col=None, header=0) if os.path.getsize('IBI.csv') > 0 else None
        temp = pd.read_csv('TEMP.csv', index_col=None, header=0)

        accelerometer = accelerometer.rename(columns={accelerometer.columns[0]: 'accelerometer_X', accelerometer.columns[1]: 'accelerometer_Y', accelerometer.columns[2]: 'accelerometer_Z'})
        blood_vol = blood_vol.rename(columns={blood_vol.columns[0]: 'blood_vol'})
        eda = eda.rename(columns={eda.columns[0]: 'eda'})
        heart_rate = heart_rate.rename(columns={heart_rate.columns[0]: 'heart_rate'})
        if ibi is not None:
            ibi = ibi.rename(columns={ibi.columns[0]: 'ibo1', ibi.columns[1]: 'ibo2'})
        
        temp = temp.rename(columns={temp.columns[0]: 'temp'})

        if ibi is not None:
            joined = accelerometer.join(blood_vol).join(eda).join(heart_rate).join(ibi).join(temp).dropna()
        else:
            joined = accelerometer.join(blood_vol).join(eda).join(heart_rate).dropna()
            joined['ibo1'] = 99999
            joined['ibo2'] = 99999
            joined = joined.join(temp)
        
        os.remove('ACC.csv')
        os.remove('BVP.csv')
        os.remove('EDA.csv')
        os.remove('HR.csv')
        if ibi is not None:
            os.remove('IBI.csv')
        os.remove('TEMP.csv')
        return joined

folders = glob.glob(os.path.join("Data", "*"))
files = [glob.glob(os.path.join(f, '*.zip')) for f in folders]
dfs = [read_zipfile(file) for sublist in files for file in sublist]
df = pd.concat(dfs, ignore_index=True, sort=False)
df = df[~(df['ibo1'] == 99999)]


# In[8]:


df


# In[9]:


df.info()


# In[12]:


#This code filters a pandas DataFrame called df based on the values in the ibo2 column. The code selects all rows where 
#the value in the ibo2 column is not equal to the string ' IBI'. The resulting DataFrame is then assigned back to the 
#variable df.
#In other words, the code removes all rows from df where the ibo2 column has the value of ' IBI'. This is accomplished 
#using boolean indexing in pandas, where df['ibo2'] != ' IBI' creates a boolean mask of True and False values, and only the 
#rows corresponding to True values are kept in the resulting DataFrame.

df = df[df['ibo2'] != ' IBI']


# In[13]:


#This code converts the values in the ibo2 column of a pandas DataFrame called df from the object data type to the 
#float data type using the .astype() method.

df['ibo2'] = df['ibo2'].astype(float)


# In[14]:


df.info()


# In[15]:


df.describe()


# In[16]:


#This code defines a function called get_target that takes a row of data and three threshold values 
#(acc_quantile, bvp_quantile, hr_quantile) as input. The function returns a value of 1 if the accelerometer_X, 
#accelerometer_Y, accelerometer_Z, blood_vol, and heart_rate columns in the row are all greater than their corresponding 
#quantile values, and 0 otherwise.

def get_target(row, acc_quantile, bvp_quantile, hr_quantile):
    return 1 if (row['accelerometer_X'] > acc_quantile) and (row['accelerometer_Y'] > acc_quantile) and                 (row['accelerometer_Z'] > acc_quantile) and (row['blood_vol'] > bvp_quantile) and                 (row['heart_rate'] > hr_quantile) else 0

acc_quantile = np.quantile(df['accelerometer_X'], 0.30)
bvp_quantile = np.quantile(df['accelerometer_Y'], 0.30)
hr_quantile = np.quantile(df['heart_rate'], 0.30)

df['target'] = [get_target(row, acc_quantile, bvp_quantile, hr_quantile) for index, row in df.iterrows()]


# In[17]:


#code concatenates the zero_values and one_values DataFrames using the pd.concat method, 
#creating a new DataFrame called df_after_downsampling. The ignore_index argument is set to True to reset the index of the 
#concatenated DataFrame. This results in a new DataFrame that has roughly equal numbers of rows where the target 
#column is 0 or 1.


zero_values = df[df['target'] == 0]
one_values = df[df['target'] == 1]

zero_values = zero_values.sample(227000)

df_after_downsampling = pd.concat([zero_values,one_values],ignore_index = True)


# In[18]:


#This code returns a tuple containing the shapes of two DataFrames zero_values and one_values. The shape attribute of a 
#DataFrame returns a tuple representing the number of rows and columns in the DataFrame.

zero_values.shape, one_values.shape


# In[19]:


#This code computes the correlation matrix for the downsampled DataFrame df_after_downsampling using the corr() method of a 
#pandas DataFrame. The transpose() method is then called on the result to obtain a transposed version of the correlation 
#matrix, which is a symmetric matrix of pairwise correlations between all pairs of features in the DataFrame.

#The resulting corr_metric DataFrame has the same number of rows and columns as df_after_downsampling, where each row and 
#column represents a feature, and each cell contains the correlation coefficient between the corresponding pair of features.
#The correlation coefficient is a measure of the strength and direction of the linear relationship between two variables, 
#and ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no correlation.

corr_metric = df_after_downsampling.corr().transpose()


# In[20]:


#This code visualizes the correlation matrix computed in the previous step using the heatmap function from the seaborn library
#The cmap argument specifies the color map to use for the heatmap, in this case 'plasma', which ranges from purple to yellow

sns.heatmap(corr_metric, cmap='plasma')
plt.show()


# In[21]:


#This code simply assigns the downsampled DataFrame df_after_downsampling to a new variable named final. The new variable 
#final now references the downsampled DataFrame with the target variable balanced between the two classes. 
#This code assigns the value of the "df_after_downsampling" dataframe to the variable "final".

final = df_after_downsampling


# In[22]:


final.info()


# In[23]:


#This code loads data into a pandas dataframe, splits it into training and testing sets, and trains a logistic regression 
#model to predict the target variable. The model's accuracy is evaluated using the accuracy score.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data into a pandas dataframe


# Split the data into features and target variable
data = final
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Logistic Regression classifier
lr = LogisticRegression(random_state=42)

# Train the model on the training data
lr.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = lr.predict(X_test)

# Evaluate the performance of the model using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# In[24]:


#This code imports the classification_report function from sklearn.metrics, which is used to generate a report that 
#displays the precision, recall, F1 score, and support for each class in the model's predictions on the test set
#The report helps evaluate the model's performance.

from sklearn.metrics import classification_report

# Print classification report for the model
print(classification_report(y_test, y_pred))


# In[ ]:




