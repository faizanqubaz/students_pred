import pandas as pd
from distribution import check_distribution
from encode import encode_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# READ THE DATASET

data = pd.read_csv(r"C:/Users/CL/Desktop/students.csv")


# CHECK THE SHAPE
print('shape',data.shape)


# CHECK THE DATA
print(data)


# CHECK THE DATATYPES
print(data.dtypes)


# CHECK THE MATHEMATICALL
print(data.describe())

# CHECK THE DUBLICATED
print(data.duplicated())


# CHECK THE CORELATION
print(data.corr()['Performance Index'])



# check the distribution

# check_distribution(data)

# get the categoricacl_columns
categorical_columns = [col for col in data if data[col].dtypes == 'object']

# encode the categorical column first after split
encode=encode_data(data[categorical_columns])


data.drop(columns=categorical_columns,inplace=True)

print(data.shape)
print(encode.shape)

combined = pd.concat([encode.reset_index(drop=True),data.reset_index(drop=True)],axis=1)

X_train,X_test,Y_train,Y_test = train_test_split(combined.iloc[:,0:6],combined.iloc[:,-1],test_size=0.2,random_state=7)



# IMPLEMENT THE LINEAR REGRESSION
lr = LinearRegression()
lr.fit(X_train,Y_train)
y_pred = lr.predict(X_test)

score = r2_score(Y_test,y_pred)

print('r2_score',score)

print('slope',lr.coef_)
print('intercept',lr.intercept_)