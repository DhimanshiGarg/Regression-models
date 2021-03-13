#!/usr/bin/env python
# coding: utf-8

# ### Import packages

# In[287]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Set working directory and read data from directory

# In[288]:


os.chdir(r"E:\dekstop\python\Linear regression")
df=pd.read_excel(r"Assignment01 (1).xlsx",sheet_name='data')
df.head()


# In[289]:


# df.shape


# In[290]:


# df.info()


# #### Rename columns

# In[291]:


df1 = df.rename(columns={"Engine Fuel Type":"Engine_Fuel_Type","Engine HP":"Engine_HP","Engine Cylinders":"Engine_Cylinders",
                     "Transmission Type":"Transmission_Type","Number of Doors":"Number_of_Doors","Market Category":"Market_Category",
                     "Vehicle Size":"Vehicle_Size","Vehicle Style":"Vehicle_Style","highway MPG":"highway_MPG","city mpg":"city_mpg"})


# In[292]:


df1.columns


# #### Drop unnecessary columns

# In[293]:


df1.drop(columns=['Model','Make','Popularity','Market_Category','Year'], axis=1,inplace=True)


# In[294]:


df1.shape


# ### Missing value treatment

# In[295]:


df1.isnull().sum().sort_values(ascending=False)


# In[296]:


total_missing_value = df1.isnull().sum().sort_values(ascending=False)
percent_of_missign_value = (df1.isnull().sum()/df1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_missing_value, percent_of_missign_value], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[297]:


# Treating Engine HP and Engine Cylinders missing values as we can see they have so
# less percentage of missing values so we are using bfill and ffill metthod to treat missing values


# In[298]:


df1 = df1.fillna({
        'Engine_Cylinders' : df1['Engine_Cylinders'].ffill(),
        'Engine_HP' : df1['Engine_HP'].bfill()
    })


# In[299]:


# Number of doors have 6 values missing


# In[300]:


df1['Number_of_Doors']=df1['Number_of_Doors'].fillna(df1['Number_of_Doors'].mode()[0])


# In[301]:


# Engine Fuel Type


# In[302]:


df1['Engine_Fuel_Type']=df1['Engine_Fuel_Type'].fillna(df1['Engine_Fuel_Type'].mode()[0])


# In[303]:


# Market category


# In[304]:


# df1['Market_Category'] = df1.groupby(['Vehicle_Size'])['Market_Category'].apply(lambda x: x.fillna(x.mode().iloc[0]))


# In[305]:


total_missing_value = df1.isnull().sum().sort_values(ascending=False)
percent_of_missign_value = (df1.isnull().sum()/df1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_missing_value, percent_of_missign_value], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# #### splitting data into two dataframes numeric features and categorical features data

# In[306]:


df_numeric_features = df1.select_dtypes(include=[np.number])
df_numeric_features.columns


# In[307]:


df_categorical_features = df1.select_dtypes(include=[np.object])

df_categorical_features.columns


# In[308]:


df_numeric_features.shape


# In[309]:


df_categorical_features.shape


# In[310]:


# Drop duplicates values


# In[311]:


# df1.duplicated().sum()


# In[312]:


# df1.drop_duplicates(keep=False,inplace=True)


# In[313]:


df1.describe()


# ##  Exploratory Data Analysis

# ## Univariate Analysis

# In[314]:


# #creating overall profiling of the dataset
# pandas_profiling.ProfileReport(df1)


# In[315]:


df1['Engine_Fuel_Type'].value_counts()
fig, ax = plt.subplots()
df1['Engine_Fuel_Type'].value_counts().plot(ax=ax, kind='bar',figsize=(15,5))
plt.show()


# In[316]:


df1['Engine_HP'].describe()
sns.set(rc={'figure.figsize':(10,5)})
sns.kdeplot(df1['Engine_HP'],shade=True)
plt.show()


# In[317]:


df1['Engine_Cylinders'].value_counts()
fig, ax = plt.subplots()
df1['Engine_Cylinders'].value_counts().plot(ax=ax, kind='bar',figsize=(15,5))
plt.show()


# In[318]:


df1['Transmission_Type'].value_counts()
fig, ax = plt.subplots()
df1['Transmission_Type'].value_counts().plot(ax=ax, kind='bar',figsize=(10,5))
plt.show()


# In[319]:


df1['Driven_Wheels'].value_counts()
fig, ax = plt.subplots()
df1['Driven_Wheels'].value_counts().plot(ax=ax, kind='bar',figsize=(5,5))


# In[320]:


df1['Number_of_Doors'].value_counts()
fig, ax = plt.subplots()
df1['Number_of_Doors'].value_counts().plot(ax=ax, kind='bar',figsize=(5,5))


# In[321]:


# df1['Market_Category'].value_counts()
# fig, ax = plt.subplots()
# df1['Market_Category'].value_counts().plot(ax=ax, kind='bar', figsize=(25,5))
# plt.show()


# In[322]:


df1['Vehicle_Size'].value_counts()
fig, ax = plt.subplots()
df1['Vehicle_Size'].value_counts().plot(ax=ax, kind='bar', figsize=(5,5))
plt.show()


# In[323]:


df1['Vehicle_Style'].value_counts()
fig, ax = plt.subplots()
df1['Vehicle_Style'].value_counts().plot(ax=ax, kind='bar', figsize=(15,5))
plt.show()


# In[324]:


df1['highway_MPG'].describe()
sns.set(rc={'figure.figsize':(10,5)})
sns.kdeplot(df1['highway_MPG'],shade=True)
plt.show()


# In[325]:


df1['city_mpg'].describe()
sns.set(rc={'figure.figsize':(10,5)})
sns.kdeplot(df1['city_mpg'],shade=True)
plt.show()


# ## Bivariate analysis

# In[326]:


data=pd.concat([df1["Price"],df1["Engine_HP"]],axis=1)
data.plot.scatter(x='Engine_HP',y='Price',ylim=(0,800000));
plt.show()


# In[327]:


##  the plot shows that the horsepower and price are related to each other


# In[328]:


data=pd.concat([df1["Price"],df1["Engine_Cylinders"]],axis=1)
data.plot.scatter(x='Engine_Cylinders',y='Price',ylim=(0,800000));


# In[329]:


##  the plot shows that the engine cylinders and price are related to each other


# In[330]:



data=pd.concat([df1["Price"],df1["Number_of_Doors"]],axis=1)
data.plot.scatter(x='Number_of_Doors',y='Price',ylim=(0,800000));


# In[331]:


#plot shows no. of doors doesnot effect price


# In[332]:



data=pd.concat([df1["Price"],df1["highway_MPG"]],axis=1)
data.plot.scatter(x='highway_MPG',y='Price',ylim=(0,800000));


# In[333]:


##  the plot shows that the highway MPG  and price are related to each other


# In[334]:



data=pd.concat([df1["Price"],df1["city_mpg"]],axis=1)
data.plot.scatter(x='city_mpg',y='Price',ylim=(0,800000));


# In[335]:


# df1.boxplot(column="Price",        # Column to plot
#                  by= "Make",         # Column to split upon
#                  figsize= (8,8))


# In[336]:


df1.boxplot(column="Price",        # Column to plot
                 by= "Engine_Fuel_Type",         # Column to split upon
                 figsize= (8,8))


# In[337]:


df1.boxplot(column="Price",        # Column to plot
                 by= "Transmission_Type",         # Column to split upon
                 figsize= (8,8))


# In[338]:


df1.boxplot(column="Price",        # Column to plot
                 by= "Driven_Wheels",         # Column to split upon
                 figsize= (8,8))


# In[339]:


# df1.boxplot(column="Price",        # Column to plot
#                  by= "Market_Category",         # Column to split upon
#                  figsize= (8,8))
# plt.show()


# In[340]:


df1.boxplot(column="Price",        # Column to plot
                 by= "Vehicle_Size",         # Column to split upon
                 figsize= (8,8))
plt.show()


# In[341]:



df1.boxplot(column="Price",        # Column to plot
                 by= "Vehicle_Style",         # Column to split upon
                 figsize= (8,8))
plt.show()


# ## Multivariate analysis

# In[342]:


###### The main check points would be the correlation between the numeric variables and target variable with multicollinearity.


# In[343]:


#calculating correlation among numeric variable 
corr_matrix = df_numeric_features.corr() 

#filter correlation values above 0.5
filter_corr = corr_matrix[corr_matrix > 0.5]

#plot correlation matrix
plt.figure(figsize=(20,12))
sns.heatmap(filter_corr,
            cmap='coolwarm',
            annot=True);


# #### Based on the above correlation matrix, correlation among the variables been observed. For example, price is correlated with engine cylinders and engine Hp
# It also show the multicollinearity, for example the correlation between engine cylinders and engine hp is very high (0.78)

# In[344]:


#Finding outliers and treatment of outliers


# In[345]:


df_outlr=df1.describe(percentiles=[.01,.05,.1,.2,.25,.5,.75,.90,.91,.92,.93,.94,.95,.96,.97,.99]).T


# In[346]:


df_outlr


# In[347]:


# df_outlr.to_csv("outliers1.csv")


# In[348]:


df1['Engine_HP'][df1['Engine_HP']>510.0]=510.0
df1['Engine_Cylinders'][df1['Engine_Cylinders']>8]=8
df1['highway_MPG'][df1['highway_MPG']>40.0]=40.0
df1['city_mpg'][df1['city_mpg']>31.0]=31.0




# In[349]:


df1.describe(percentiles=[.01,.05,.1,.2,.25,.5,.75,.90,.91,.92,.93,.94,.95,.96,.97,.99]).T


# In[350]:


### Scale our numeric variables using min-max normalization


# In[351]:


df1.describe()


# In[352]:


df_numeric_features.shape


# In[353]:


df1.numeric_features=df1.select_dtypes(include=[np.number])
df1.numeric_features.shape


# In[354]:


from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler()
df_minmax = pd.DataFrame(min_max.fit_transform(df1.numeric_features.iloc[:,0:6]),
columns = df1.numeric_features.iloc[:,0:6].columns.tolist())


# In[355]:


df_minmax.head()


# #### ONE HOT ENCODING FOR CATEGORICAL FEATURES

# In[356]:


df1.categorial_features=df1.select_dtypes(include=np.object)


# In[357]:


for col in  df1.categorial_features.columns.values:
    one_hot_encoded_variables = pd.get_dummies(df1.categorial_features[col],prefix=col)
    df1.categorial_features = pd.concat([df1.categorial_features,one_hot_encoded_variables],axis=1)
    df1.categorial_features.drop([col],axis=1, inplace=True)


# In[358]:


# df1.columns


# #### Concatenate the numeric and encoded variables to the dataframe

# In[359]:


df_cardata = pd.concat([df_minmax,df1.categorial_features], axis=1)


# In[360]:


df_cardata.shape


# In[361]:


### Create training and testing datasets using the train_test_split


# In[362]:


# create feature and response variable set
# we create train and test sample from our dataset
from sklearn.model_selection import train_test_split
# create feature and response varibles
X = df_cardata.drop(['Price'], axis=1)
Y = df_cardata['Price']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,random_state=11)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# ### SGDRegressor() 

# In[363]:


import numpy as np
from sklearn.linear_model import SGDRegressor
lin_model = SGDRegressor()
# we fit our model with train data
lin_model.fit(x_train, y_train)
# we use predict() to predict our values
lin_model_predictions = lin_model.predict(x_test)
# we check the coefficient of determination with score()
print(lin_model.score(x_test,y_test))


# ### Mean Square Error

# In[364]:


# we check the root mean square error (RMSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, lin_model_predictions)
rmse = np.sqrt(mse)
print(rmse)


# ### Create a model with grid search

# In[365]:


# ignore the deprecation warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import GridSearchCV

# Grid search - this will take about 1 minute.
param_grid = {
    'alpha': 10.0 ** -np.arange(1, 7),
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'learning_rate': ['constant', 'optimal', 'invscaling'],
}
clf = GridSearchCV(lin_model, param_grid)
clf.fit(x_train, y_train)
print("Best score: " + str(clf.best_score_))


# In[366]:


pred_test = clf.predict(x_test)


# In[367]:


clf.best_estimator_


# ### Build a model using best parameters
# create a linear regression with sgd

# In[370]:


linreg_sgd=SGDRegressor(alpha=1e-06, learning_rate='constant',loss='huber',
             penalty='elasticnet')


# ### Training the model

# In[371]:


warnings.filterwarnings("ignore", category=DeprecationWarning)

linreg_sgd.fit(x_train,y_train)


# ### Predictions on test data

# In[372]:


pred_test = linreg_sgd.predict(x_test)


# ### Mean Square Error

# In[373]:


mse = mean_squared_error(y_test, pred_test)
rmse = np.sqrt(mse)
print(rmse)


# ### R square and adjusted r square with linear regressor

# In[374]:


yPrediction = linreg_sgd.predict(x_test)
SS_Residual = sum((y_test-yPrediction)**2)
SS_Total = sum((y_train-np.mean(y_train))**2)
#r_squared = 1 - (float(SS_Residual))/SS_Total
r_squared=(SS_Total-SS_Residual)/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-x_train.shape[1]-1)
print (r_squared, adjusted_r_squared)


# In[376]:


prediction=pd.DataFrame({'actual_price':y_test,'predicted_price':yPrediction})


# In[377]:


plot_pred=prediction.head(30)


# In[378]:


plot_pred.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# ### Training and testing score with decision tree regressor

# In[379]:


from sklearn import tree
dt3=tree.DecisionTreeRegressor(random_state=11,max_depth=6, min_samples_split=6)
dt3.fit(x_train, y_train)
dt3_score_train = dt3.score(x_train, y_train)
print("Training score: ",dt3_score_train)
dt3_score_test = dt3.score(x_test, y_test)
print("Testing score: ",dt3_score_test)


# ### R squre by decision regressor

# In[380]:


y_prediction_tree=dt3.predict(x_test)


# In[383]:


SS_Residual = sum((y_test-y_prediction_tree)**2)
SS_Total = sum((y_test-np.mean(y_test))**2)
#r_squared = 1 - (float(SS_Residual))/SS_Total
r_squared=(SS_Total-SS_Residual)/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-x_train.shape[1]-1)
print (r_squared, adjusted_r_squared)


# In[384]:


y_prediction_tree1=dt3.predict(x_train)


# In[385]:


SS_Residual = sum((y_train-y_prediction_tree1)**2)
SS_Total = sum((y_train-np.mean(y_train))**2)
#r_squared = 1 - (float(SS_Residual))/SS_Total
r_squared=(SS_Total-SS_Residual)/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-x_train.shape[1]-1)
print (r_squared, adjusted_r_squared)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[359]:


# for col in df_cardata.columns: 
#     print(col)


# In[360]:


# import statsmodels.api as sm  # Statical learning
# X = df_cardata[['Year', 'Engine_HP', 'Engine_Cylinders', 'Number_of_Doors',
#        'highway_MPG', 'city_mpg', 'Price', 'Engine_Fuel_Type_diesel',
#        'Engine_Fuel_Type_electric',
#        'Engine_Fuel_Type_flex-fuel (premium unleaded recommended/E85)',
#        'Engine_Fuel_Type_flex-fuel (premium unleaded required/E85)',
#        'Engine_Fuel_Type_flex-fuel (unleaded/E85)',
#        'Engine_Fuel_Type_flex-fuel (unleaded/natural gas)',
#        'Engine_Fuel_Type_natural gas',
#        'Engine_Fuel_Type_premium unleaded (recommended)',
#        'Engine_Fuel_Type_premium unleaded (required)',
#        'Engine_Fuel_Type_regular unleaded',
#        'Transmission_Type_AUTOMATED_MANUAL', 'Transmission_Type_AUTOMATIC',
#        'Transmission_Type_DIRECT_DRIVE', 'Transmission_Type_MANUAL',
#        'Transmission_Type_UNKNOWN', 'Driven_Wheels_all wheel drive',
#        'Driven_Wheels_four wheel drive', 'Driven_Wheels_front wheel drive',
#        'Driven_Wheels_rear wheel drive', 'Vehicle_Size_Compact',
#        'Vehicle_Size_Large', 'Vehicle_Size_Midsize',
#        'Vehicle_Style_2dr Hatchback', 'Vehicle_Style_2dr SUV',
#        'Vehicle_Style_4dr Hatchback', 'Vehicle_Style_4dr SUV',
#        'Vehicle_Style_Cargo Minivan', 'Vehicle_Style_Cargo Van',
#        'Vehicle_Style_Convertible', 'Vehicle_Style_Convertible SUV',
#        'Vehicle_Style_Coupe', 'Vehicle_Style_Crew Cab Pickup',
#        'Vehicle_Style_Extended Cab Pickup', 'Vehicle_Style_Passenger Minivan',
#        'Vehicle_Style_Passenger Van', 'Vehicle_Style_Regular Cab Pickup',
#        'Vehicle_Style_Sedan', 'Vehicle_Style_Wagon'
#     ]]
# Y = df_cardata['Price']
# # Y=df3["Sales_in_thousands"]

# X = sm.add_constant(X) # adding a constant

# model = sm.OLS(Y, X).fit()
# #predictions = model.predict(X) 


# print(model.summary())


# In[337]:


df1.columns


# In[ ]:





# In[ ]:




