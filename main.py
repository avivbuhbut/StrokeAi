from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pybrain

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.structure.modules import SigmoidLayer
from pybrain.tools.shortcuts import buildNetwork


from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader
import pickle


dataset = pd.read_csv('C:/Users/avivb/PycharmProjects/StrokeAi1/healthcare-dataset-stroke-data.csv')
df = pd.read_csv('C:/Users/avivb/PycharmProjects/StrokeAi1/healthcare-dataset-stroke-data.csv')

dataset.isna().sum()

dataset = dataset.dropna()

input = dataset.iloc[:,1:11].values

outputs =  dataset.iloc[:,11:12].values
outputs = outputs.squeeze()


#Male Or Female Encoding Pocedure

#shaping the data of male and female
MaleFemale = dataset.iloc[:,1:2].values
print(MaleFemale)
MaleFemale = np.array(MaleFemale)
MaleFemale = MaleFemale.flatten()
print(MaleFemale.shape)

#encoding the data
values = array(MaleFemale)
print(values)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
MaleFemale = onehot_encoded
print(MaleFemale) #printing the encoded data

#inversing the encoded data to see the result
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[965, :])])
print(inverted)


"""
#Age Encoding Pocedure

#shaping the data of Age
Age = dataset.iloc[:,2:3].values
print(Age)
Age = np.array(Age)
Age = Age.flatten()
print(Age.shape)

#encoding the data
values = array(Age)
print(values)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded) #printing the encoded data

#inversing the encoded data to see the result
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[967, :])])
print(inverted)

"""
#EverMarried Encoding Pocedure
#shaping the data of EverMarried
EverMarried = dataset.iloc[:,5:6].values
print(EverMarried)
EverMarried = np.array(EverMarried)
EverMarried = EverMarried.flatten()
print(EverMarried.shape)

#encoding the data
values = array(EverMarried)
print(values)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
EverMarried = onehot_encoded
print(EverMarried) #printing the encoded data

#inversing the encoded data to see the result
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[968, :])])
print(inverted)




#WorkType Encoding Pocedure
#shaping the data of WorkType
WorkType = dataset.iloc[:,6:7].values
print(WorkType)
WorkType = np.array(WorkType)
WorkType = WorkType.flatten()
print(WorkType.shape)

#encoding the data
values = array(WorkType)
print(values)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
WorkType =onehot_encoded
print(WorkType) #printing the encoded data

#inversing the encoded data to see the result
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[968, :])])
print(inverted)


#Residence_type Encoding Pocedure
#shaping the data of Residence_type
Residence_type = dataset.iloc[:,7:8].values
print(Residence_type)
Residence_type = np.array(Residence_type)
Residence_type = Residence_type.flatten()
print(Residence_type.shape)

#encoding the data
values = array(Residence_type)
print(values)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
Residence_type =onehot_encoded
print(Residence_type) #printing the encoded data

#inversing the encoded data to see the result
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[968, :])])
print(inverted)


#smoking_status Encoding Pocedure
#shaping the data of smoking_status
smoking_status = dataset.iloc[:,10:11].values
print(smoking_status)
smoking_status = np.array(smoking_status)
smoking_status = smoking_status.flatten()
print(smoking_status.shape)

#encoding the data
values = array(smoking_status)
print(values)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
smoking_status =onehot_encoded
print(smoking_status) #printing the encoded data

#inversing the encoded data to see the result
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[968, :])])
print(inverted)


"""
j=0


for i in range(len(input)):
  if input[i][j] ==  'Male':
    input[i][j] = 1
  if input[i][j] == 'Female':
      input[i][j] = 0



j=4


for i in range(len(input)):
  if input[i][j] ==  'Yes':
    input[i][j] = 1
  if input[i][j] == 'No':
      input[i][j] = 0

j = 5

for i in range(len(input)):
    if input[i][j] == 'Private':
        input[i][j] = 1
    if input[i][j] == 'Self-employed':
        input[i][j] = 2
    if input[i][j] == 'Govt_job':
        input[i][j] = 3
    if input[i][j] == 'children':
        input[i][j] = 4
    if input[i][j] == 'Never_worked':
        input[i][j] = 5
    if input[i][j] == 'Other':
        input[i][j] = 6


j=6


for i in range(len(input)):
  if input[i][j] ==  'Urban':
    input[i][j] = 1
  if input[i][j] == 'Rural':
    input[i][j] = 2





j=9


for i in range(len(input)):
  if input[i][j] ==  'never smoked':
    input[i][j] =0
  if input[i][j] == 'formerly smoked':
    input[i][j] = 1
  if input[i][j] == 'smokes':
     input[i][j] = 2
  if input[i][j] == 'Unknown':
     input[i][j] = 0

"""

j=8


for i in range(len(input)):
  if input[i][j] ==  np.nan:
    input[i][j] = 21


#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#inputs = scaler.fit_transform(input)
print('MaleFemaleShape: ', MaleFemale.shape)
print('EverMarriedS: ', EverMarried.shape)
print('WorkType: ', WorkType.shape)
print('Residence_type: ', Residence_type.shape)
print('smoking_status: ', smoking_status.shape)

print('smoking_status[0][3]',smoking_status[0][0])
inverted = label_encoder.inverse_transform([argmax(smoking_status[3, :])])
print('inverse' , inverted)

print(input)
# MaleFemale = np.reshape(MaleFemale,(-1,2))

#EverMarried = np.reshape(EverMarried,(-1,2))

#WorkType= np.reshape(WorkType,(-1,2))

#Residence_type = np.reshape(Residence_type,(-1,2))

#smoking_status= np.reshape(smoking_status,(-1,2))


print(input[0][1])

dataset = SupervisedDataSet(20, 1)
for i in range(len(outputs)):
    if outputs[i] == 0:
        output = 0
    else:
        output = 1
    dataset.addSample((MaleFemale[i][0],MaleFemale[i][1],
                       input[i][1], input[i][2], input[i][3],
                       EverMarried[i][0], EverMarried[i][1],
                       WorkType[i][0], WorkType[i][1],WorkType[i][2], WorkType[i][3],WorkType[i][4],
                       Residence_type[i][0], Residence_type[i][1],
                       input[i][7], #Gloucose
                       input[i][8],#bmi
                       smoking_status[i][0],smoking_status[i][1],smoking_status[i][2],smoking_status[i][3],
                       ), output)








"""


#optimizer = BackpropTrainer(module=network, dataset= dataset, learningrate=0.001)


#buildNetwork(10,8,8, 1, outclass = SigmoidLayer, hiddenclass = SigmoidLayer, bias = False)
network = buildNetwork(20,300,100, 1, outclass = SigmoidLayer, hiddenclass = SigmoidLayer, bias = False)



learnRate = 0.01

epoch = 0
error = []


epochLimit = 100000

optimizer = BackpropTrainer(module=network, dataset=dataset, learningrate=0.0001)
error_average = optimizer.train()

while epoch < epochLimit:
    if epoch > 98000 and error_average > 0.001:
        epochLimit = epochLimit + 1000
    error_average = optimizer.train()
    epoch= epoch+1
    if epoch % 1 == 0:
        print('Epoch: ' + str(epoch + 1) + ' Error: ' + str(error_average))
        error.append(error_average)
        if epoch %10 ==0:
            print(' Net Saved' )
            pkl_filename = "StrokeModelBigData.pkl"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(network, file)
        if epoch %5000 ==0:
            for i in range(len(outputs)):
                print('Desired result: ',outputs[i])
                test = np.array([MaleFemale[i][0],MaleFemale[i][1],
                       input[i][1], input[i][2], input[i][3],
                       EverMarried[i][0], EverMarried[i][1],
                       WorkType[i][0], WorkType[i][1],WorkType[i][2], WorkType[i][3],WorkType[i][4],
                       Residence_type[i][0], Residence_type[i][1],
                       input[i][7], #Gloucose
                       input[i][8],#bmi
                       smoking_status[i][0],smoking_status[i][1],smoking_status[i][2],smoking_status[i][3],
                       ])
                result = network.activate(test)
                percentage = "{:.0%}".format(float(result))
                print('Net Result: ', percentage)


pkl_filename = "StrokeModelBigData.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(network, file)



plt.xlabel('Epoch')
plt.ylabel('Error')
plt.plot(error)
plt.show()

pkl_filename = "StrokeModelBigData.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(network, file)




"""



pkl_filename = "StrokeModelBigData.pkl"
with open(pkl_filename, 'rb') as file: #load the network (the network in now called model)
  network = pickle.load(file)

optimizer = BackpropTrainer(module=network, dataset= dataset, learningrate=0.008)
error_average = optimizer.train()
print(error_average)

testCostumeInput = dataset.getSample(969)


result = network.activate(np.array([MaleFemale[969][0], MaleFemale[969][1],
                     input[969][1], input[969][2], input[969][3],
                     EverMarried[969][0], EverMarried[969][1],
                     WorkType[969][0], WorkType[969][1], WorkType[969][2], WorkType[969][3], WorkType[969][4],
                     Residence_type[969][0], Residence_type[969][1],
                     input[969][7],  # Gloucose
                     input[969][8],  # bmi
                     smoking_status[969][0], smoking_status[969][1], smoking_status[969][2], smoking_status[969][3],
                     ]))
percentage = "{:.0%}".format(float(result))
print('Net Result: ', percentage)



"""
result = network.activate([testCostumeInput])
percentage = "{:.0%}".format(float(result))
print('Net Result: ', percentage)


for i in range(len(outputs)):
    print('Desired result: ', outputs[i])
    test = np.array([MaleFemale[i][0], MaleFemale[i][1],
                     input[i][1], input[i][2], input[i][3],
                     EverMarried[i][0], EverMarried[i][1],
                     WorkType[i][0], WorkType[i][1], WorkType[i][2], WorkType[i][3], WorkType[i][4],
                     Residence_type[i][0], Residence_type[i][1],
                     input[i][7],  # Gloucose
                     input[i][8],  # bmi
                     smoking_status[i][0], smoking_status[i][1], smoking_status[i][2], smoking_status[i][3],
                     ])
    result = network.activate(test)
    percentage = "{:.0%}".format(float(result))
    print('Net Result: ', percentage)

"""

df.loc[1027, 'hypertension'] = 0
print(df.loc[1027, 'hypertension'])
df.to_csv("C:/Users/avivb/PycharmProjects/StrokeAi1/healthcare-dataset-stroke-data.csv", index=False)


print(input[969,:])
