import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense,SimpleRNN, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping


# in this code tensorflow doesn't support TimeseriesGenerator object with (model.fit_generator) input so this cant work so I convert it to the numpy array and then train the model

warnings.filterwarnings('ignore')

x = np.linspace(0, 50, 501)
# print(x)
# print(len(x))

y = np.sin(x)
# print(y)

# plt.plot(x, y)
# plt.show()

df = pd.DataFrame(data=y, index=x, columns=['Sine'])
# print(df)
# df.plot()
# plt.show()

# print(len(df))

test_percent = 0.1
# print(len(df)*test_percent)
test_point = np.round(len(df)*test_percent)
# print(test_point)
test_index = int(len(df) - test_point)
# print(test_index)
train = df.iloc[:test_index]
test = df.iloc[test_index:]
# print(train)
# print('---------------------')
# print(test)

scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# generate batches for sequence data
# print(help(TimeseriesGenerator))

# length = 2
# batch_size = 1

# generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)
# print(len(scaled_train))
# print(len(generator))

# len(generator) is equal to len(scaled_train) - length because its generate batches for us 
# x,y = generator[0] # very first batch 
# print(x) # give 2 no
# print(y)
# print(scaled_train)

# length = 4
# batch_size = 1
# generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)
# print(len(scaled_train)) 
# print(len(generator))

# x,y = generator[0] # very first batch 
# print(x) # give 4 no
# print(y)
# print(scaled_train)

# so basically generator generate batches of size of length for our train data so we have len(scaled_data) - length batches

# now we have sine wave so we put length that enough long to so we cover up a wave remember larger the length longer the trainning time
length = 50 # so now we feed 50 points as x and asking model what is 51 no of point
batch_size = 1

generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)
# print(generator[0])
# print(len(generator))
x, y = generator[0]
# print(x)
# print(x.shape)
# print(y)
# print(y.shape)
x_temp=[]
y_temp=[]
for x,y in generator:
    x_temp.append(x)
    y_temp.append(y)

x_temp = np.array(x_temp)
y_temp = np.array(y_temp)

# print(x_temp.shape)
# print(y_temp.shape)
x_temp = x_temp.reshape(401,50,1)
y_temp = y_temp.reshape(401,1,1)
# print(x_temp)
# print(x_temp.shape)
# print(y_temp.shape)
# print(type(x_temp))
# print(type(y_temp))
# x_temp = pd.DataFrame(x_temp)
# y_temp = pd.DataFrame(y_temp)
# print(data)
# print(x)
# print(y)
n_feature = 1 # because we have 1 feature input x and we try to predict y

model = Sequential()
model.add(SimpleRNN(50, input_shape=(length, n_feature))) # neuron no is equal to the length of batch

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
# print(model.summary())

# hist = model.fit(x=x_temp, y=y_temp, epochs=5)
# losses = pd.DataFrame(hist.history)
# losses.to_csv('RNNonSinewave.csv', index=False)

# model.save('RNNonSinewave.h5')

# losses = pd.read_csv('RNNonSinewave\RNNonSinewave.csv')
# losses.plot()
# plt.show()

later_model = load_model('RNNonSinewave\RNNonSinewave.h5')
first_eval_batch = scaled_train[-length:]
# print(first_eval_batch)
first_eval_batch = first_eval_batch.reshape((1, length, n_feature))
next_point = later_model.predict(first_eval_batch)
# print(next_point)
# print(scaled_test[0])


# example for only one prediction 
# first_eval_batch = scaled_train[-length:]
# current_batch = first_eval_batch.reshape(1, length, n_feature)
# predicted_value = [[[99]]]
# print(np.append(current_batch[:,1:,:], [[[99]]], axis=1))



test_prediction = []

# example for only one prediction 
first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape(1, length, n_feature)

for i in range(len(test)):
    current_pred = later_model.predict(current_batch)
    test_prediction.append(current_pred)  
    current_batch = np.append(current_batch[:,1:,:],[current_pred] , axis=1) 


test_prediction = np.array(test_prediction).reshape(50, 1)
# print(train.shape)
# print(scaled_train.shape)

true_predictions = scaler.inverse_transform(test_prediction)
test['Predictions'] = true_predictions 
# print(test)

# test.plot()
# plt.show()

# part 2 with early stopping

length = 49
batch_size = 1
validation_generatror = TimeseriesGenerator(scaled_test, scaled_test, length=length, batch_size=batch_size)
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)
# print(len(generator))
# print(len(validation_generatror))

x_train = []
y_train = []
for x, y in generator:
    x_train.append(x)
    y_train.append(y)

x_test = []
y_test = []
for x, y in validation_generatror:
    x_test.append(x)
    y_test.append(y)

x_train = np.array(x_train).reshape(402, 49, 1)
y_train = np.array(y_train).reshape(402, 1, 1)

x_test = np.array(x_test).reshape(1, 49, 1)
y_test = np.array(y_test).reshape(1, 1, 1)


# x, y = generator[0]
# print(x.shape)
# print(y.shape)

# x, y = validation_generatror[0]
# print(x.shape)
# print(y.shape)

n_feature = 1 # because we have 1 feature input x and we try to predict y

model = Sequential()
model.add(LSTM(50, input_shape=(length, n_feature)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=2)

# hist = model.fit(x=x_temp, y=y_temp,validation_data=(x_test, y_test),  epochs=20, callbacks=[early_stop])


# losses = pd.DataFrame(hist.history)
# losses.to_csv('RNNonSinewaveEarlyStopping.csv', index=False)

# model.save('RNNonSinewaveEarlyStopping.h5')

losses = pd.read_csv('RNNonSinewave\RNNonSinewaveEarlyStopping.csv')
# losses.plot()
# plt.show()

later_model = load_model('RNNonSinewave\RNNonSinewaveEarlyStopping.h5')

test_prediction = []

# example for only one prediction 
first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape(1, length, n_feature)

for i in range(len(test)):
    current_pred = later_model.predict(current_batch)
    test_prediction.append(current_pred)  
    current_batch = np.append(current_batch[:,1:,:],[current_pred] , axis=1) 


test_prediction = np.array(test_prediction).reshape(50, 1)
# print(train.shape)
# print(scaled_train.shape)

true_predictions = scaler.inverse_transform(test_prediction)
test['Predictions'] = true_predictions 
# print(test)

# test.plot()
# plt.show()

# application

forecast = []
first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_feature))

point_no = 25
for i in range(point_no):
    current_pred = later_model.predict(current_batch)
    forecast.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:], [current_pred], axis=1)

forecast = np.array(forecast).reshape(25, 1)

forecast = scaler.inverse_transform(forecast)
# df_forecast =  pd.DataFrame(forecast)
# print(df_forecast)
# df_forecast.plot()
# plt.show()

# here original df index end at 50 so our forecast index start with 50 

forecast_index = np.arange(50.1, 50.1 + point_no*0.1, step=0.1)
print(len(forecast))
print(len(forecast_index))

plt.plot(df.index, df['Sine'])
plt.plot(forecast_index, forecast)
plt.show()
