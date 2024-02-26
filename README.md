# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The term neural network refers to a group of interconnected units called neurons that send signals to each other. While individual neurons are simple, many of them together in a network can perform complex tasks. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression is a method for understanding the relationship between independent variables or features and a dependent variable or outcome. Outcomes can then be predicted once the relationship between independent and dependent variables has been estimated.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 1 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.


Explain the problem statement

## Neural Network Model
![Screenshot 2024-02-26 144908](https://github.com/23008112/basic-nn-model/assets/138972470/f9c69a79-b536-4e4e-a75b-8bb845875083)

## DESIGN STEPS

### STEP 1:

Loading the dataset
### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

### Name: R.SANJANA
### Register Number: 212223240148

```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd


auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('linear').sheet1
rows = worksheet.get_all_values()


df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'int'})
df = df.astype({'output':'float'})
df.head(10)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X=df[['input']].values
Y=df[['output']].values
X

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.33,random_state =33)
Scaler  = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

ai = Sequential([
   Dense(units=1,activation='relu',input_shape=[1]),

   Dense(1)
])
ai.compile(optimizer='rmsprop',loss='mse')
ai.fit(X_train1,y_train,epochs=4000)
loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai.evaluate(X_test1,y_test)
X_n1 = [[20]]
X_n1_1 = Scaler.transform(X_n1)
ai.predict(X_n1_1)
```
## Dataset Information
![Screenshot 2024-02-26 142208](https://github.com/23008112/basic-nn-model/assets/138972470/48863db3-8467-4a7c-8d47-a61b215dd171)

![Screenshot 2024-02-26 142505](https://github.com/23008112/basic-nn-model/assets/138972470/cf32255c-4486-4f71-80fd-18a60ec403c5)

## OUTPUT

### Training Loss Vs Iteration Plot
![Screenshot 2024-02-26 143742](https://github.com/23008112/basic-nn-model/assets/138972470/332a3d85-1b73-4005-88cc-557563c0949f)

### Test Data Root Mean Squared Error
![Screenshot 2024-02-26 143958](https://github.com/23008112/basic-nn-model/assets/138972470/fc44e583-22f7-4447-acde-ccdf876d75d6)

![Screenshot 2024-02-26 144032](https://github.com/23008112/basic-nn-model/assets/138972470/b8f26f85-2841-4b33-a188-fb4d5c92054d)

### New Sample Data Prediction
![Screenshot 2024-02-26 144059](https://github.com/23008112/basic-nn-model/assets/138972470/050160ec-c4b1-46ac-8462-1a97b10dd91b)

## RESULT
The model is successfully created and the predicted value is close to the actual value
