import  numpy as np 
import pandas as pd 
from pandas_datareader import data
from pandas import Series, DataFrame
import matplotlib.pyplot as plt 
import seaborn  as sns 
sns.set_style('whitegrid')
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.DESCR)
#Linear regression


#data visualization
plt.hist(boston.target, bins=50)
plt.xlabel('Prices in $1000')
plt.ylabel('Number of houses')

plt.scatter(boston.data[:,5],boston.target)
plt.xlabel('Number of rooms')
plt.ylabel('Price')

plt.scatter(boston.data[:,5],boston.target)
plt.xlabel('Number of rooms')
plt.ylabel('Price')

bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['Price'] = boston.target
print(bos.head())
##basic regression
sns.lmplot('RM','Price',data = bos)

# Import for Linear Regression
import sklearn
from sklearn.linear_model import LinearRegression

# Create a LinearRegression Object
lreg = LinearRegression()

# Data Columns
X_multi = boston_df.drop('Price',1)

# Targets
Y_target = boston_df.Price

# Implement Linear Regression
lreg.fit(X_multi,Y_target)

print(' The estimated intercept coefficient is %.2f ' %lreg.intercept_)
print(' The number of coefficients used was %d ' % len(lreg.coef_))

# Set a DataFrame from the Features
coeff_df = DataFrame(boston_df.columns)
coeff_df.columns = ['Features']

# Set a new column lining up the coefficients from the linear regression
coeff_df["Coefficient Estimate"] = pd.Series(lreg.coef_)

# Show
coeff_df

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_multi, Y_target)
# Print shapes of the training and testing data sets
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# Create our regression object
lreg = LinearRegression()

# Once again do a linear regression, except only on the training sets this time
lreg.fit(X_train,Y_train)

# Predictions on training and testing sets
pred_train = lreg.predict(X_train)
pred_test = lreg.predict(X_test)

print("Fit a model X_train, and calculate MSE with Y_train: %.2f"  % np.mean((Y_train - pred_train) ** 2))
    
print("Fit a model X_train, and calculate MSE with X_test and Y_test: %.2f"  %np.mean((Y_test - pred_test) ** 2))

# Scatter plot the training data
train = plt.scatter(pred_train,(Y_train-pred_train),c='b',alpha=0.5)

# Scatter plot the testing data
test = plt.scatter(pred_test,(Y_test-pred_test),c='r',alpha=0.5)

# Plot a horizontal axis line at 0
plt.hlines(y=0,xmin=-10,xmax=50)

#Labels
plt.legend((train,test),('Training','Test'),loc='lower left')
plt.title('Residual Plots')


#### onther way
# Residual plot of all the dataset using seaborn
sns.residplot('RM', 'Price', data = boston_df)  ####מספיק לשים רק עמודה אחת מתוך קובץ הנתונים  וזה יחשב שארית על הכל 



##logistic regression
# Data Imports
import numpy as np
import pandas as pd
from pandas import Series,DataFrame

# Math
import math

# Plot imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# Machine Learning Imports
from sklearn.linear_model import LogisticRegression


# For evaluating our ML results
from sklearn import metrics

# Dataset Import
import statsmodels.api as sm


# Logistic Function
def logistic(t):
    return 1.0 / (1 + math.exp((-1.0)*t) )

# Set t from -6 to 6 ( 500 elements, linearly spaced)
t = np.linspace(-6,6,500)

# Set up y values (using list comprehension)
y = np.array([logistic(ele) for ele in t])

# Plot
plt.plot(t,y)
plt.title(' Logistic Function ')

df = sm.datasets.fair.load_pandas().data
# Preview
df.head()

# Create check function
def affair_check(x):
    if x!=0:
        return 1
    else:
        return 0
    
df['Had_Affair'] = df['affairs'].apply(affair_check)

# Groupby Had Affair column
df.groupby('Had_Affair').mean()
# Factorplot for age with Had Affair hue
sns.factorplot('age',data=df,hue='Had_Affair',palette='coolwarm',kind = 'count')
# Factorplot for years married with Had Affair hue
sns.factorplot('yrs_married',data=df,hue='Had_Affair',palette='coolwarm',kind = 'count')
# Factorplot for number of children with Had Affair hue
sns.factorplot('children',data=df,hue='Had_Affair',kind = 'count',palette='coolwarm')
# Factorplot for number of children with Had Affair hue
sns.factorplot('educ',data=df,hue='Had_Affair',palette='coolwarm',kind = 'count')
# Create new DataFrames for the Categorical Variables
occ_dummies = pd.get_dummies(df['occupation'])
hus_occ_dummies = pd.get_dummies(df['occupation_husb'])

# Let's take a quick look at the results
occ_dummies.head()

# Create column names for the new DataFrames
occ_dummies.columns = ['occ1','occ2','occ3','occ4','occ5','occ6']
hus_occ_dummies.columns = ['hocc1','hocc2','hocc3','hocc4','hocc5','hocc6']

# Set X as new DataFrame without the occupation columns or the Y target
X = df.drop(['occupation','occupation_husb','Had_Affair'],axis=1)

# Concat the dummy DataFrames Together
dummies = pd.concat([occ_dummies,hus_occ_dummies],axis=1)
# Now Concat the X DataFrame with the dummy variables
X = pd.concat([X,dummies],axis=1)

# Preview of Result
X.head()

# Set Y as Target class, Had Affair
Y = df.Had_Affair

# Preview
Y.head()


# Dropping one column of each dummy variable set to avoid multicollinearity
X = X.drop('occ1',axis=1)
X = X.drop('hocc1',axis=1)

# Drop affairs column so Y target makes sense
X = X.drop('affairs',axis=1)

# PReview
X.head()

# Flatten array
Y = np.ravel(Y)

# Check result
Y


# Create LogisticRegression model
log_model = LogisticRegression()

# Fit our data
log_model.fit(X,Y)

# Check our accuracy
log_model.score(X,Y)
# Check percentage of women that had affairs
Y.mean()

# Use zip to bring the column names and the np.transpose function to bring together the coefficients from the model
coeff_df = DataFrame(zip(X.columns, np.transpose(log_model.coef_)))
coeff_df


# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# Make a new log_model
log_model2 = LogisticRegression()

# Now fit the new model
log_model2.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score


# Predict the classes of the testing data set
class_predict = log_model2.predict(X_test)

# Compare the predicted classes to the actual test classes
print (accuracy_score(Y_test,class_predict))


