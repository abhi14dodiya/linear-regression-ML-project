import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data_frame = pd.read_csv("student_scores.csv") #open file
print(data_frame) #print file

x = data_frame["Hours"].values.reshape(-1,1)
y = data_frame["Scores"].values.reshape(-1,1)

x_train,y_train = x[0:20],y[0:20] #for train perpouse
x_test,y_test = x[20:],y[20:] #for testing points

#print(x_train)#will print 2d array because of reshape we done
model = LinearRegression().fit(x_train,y_train)

#note:for prediction use fresh data lines 
y_prediction = model.predict(x_test)# we are predicting y for new x
print(y_prediction)# pridicted value of y

plt.plot(x_test,y_prediction,'o')#will print blue is predicted data
plt.plot(x_test,y_test,'ro')#will print in red dot is orignal data
plt.show()


#any linear equation(mx+c) contain m and c m= slope c=interepter
print(model.coef_)#will print slope of line (co-efficient)
print(model.intercept_)#will print intercepter of equation

z = mean_squared_error(y_test,y_prediction)#(orignal answer,predicted answer) this will give you how much error have in your model
# how much relible that model is less ans means more reliable
print("this is accuracy error for given 5 test points",z)
