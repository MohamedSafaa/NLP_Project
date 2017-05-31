from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
import numpy
from keras.models import model_from_json
import h5py

dataset = numpy.loadtxt("Files/featureVector.csv", delimiter=",")

x1 = dataset[0:1015,0:10]
x2 = dataset[1128:2123,0:10]
x3 = dataset[2234:3134,0:10]
x = numpy.append( x1 , x2 , axis = 0)
x = numpy.append( x , x3 , axis=0 )

y1 = dataset[0:1015,10:13]
y2 = dataset[1128:2123,10:13]
y3 = dataset[2234:3134,10:13]
y = numpy.append( y1 , y2 , axis = 0)
y = numpy.append( y , y3 , axis=0 )

#Y = to_categorical(y_int)#

xtest1 = dataset[1015:1128,0:10]
xtest2 = dataset[2123:2234,0:10]
xtest3 = dataset[3134:3235,0:10]
xtest = numpy.append( xtest1 , xtest2 , axis = 0)
xtest = numpy.append( xtest , xtest3 , axis=0 )

ytest1 = dataset[1015:1128,10:13]
ytest2 = dataset[2123:2234,10:13]
ytest3 = dataset[3134:3235,10:13]
ytest = numpy.append( ytest1 , ytest2 , axis = 0)
ytest = numpy.append( ytest , ytest3 , axis=0 )
#yTest = to_categorical(ytest_int)

model = Sequential()
model.add(Dense(20, input_dim=10, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(3, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(x, y, nb_epoch=1000, batch_size=15)

# evaluate the model
scores = model.evaluate(xtest, ytest)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

##Only code needed to save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
###########################


predictions = model.predict(x)
print(predictions)
rounded= numpy.around(predictions, decimals=0)
f_out = open("OrginalPredictVector.csv" , "w")
for pred in rounded:
	f_out.write(str(pred)+'\n')
f_out.close()
print(rounded)

