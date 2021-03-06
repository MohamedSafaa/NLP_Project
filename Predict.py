from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
import numpy
from keras.models import model_from_json
import h5py

dataset = numpy.loadtxt("TextfeatureVector.csv", delimiter=",")
x = dataset[0:,0:10]
print(x)

#Only code needed to  Load Code
json_file = open("model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam')

predictions = loaded_model.predict(x)
print(predictions)
rounded= numpy.around(predictions, decimals=0)
f_out = open("PredictVector.csv" , "w")
for pred in rounded:
	f_out.write(str(pred)+'\n')
f_out.close()
print(rounded)
