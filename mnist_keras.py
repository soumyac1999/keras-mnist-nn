import import_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop

train_images, train_labels, test_images, test_labels = import_mnist.load()

x_train = train_images.astype('float32')
x_train /= 255
x_test = test_images.astype('float32')
x_test /= 255

y_train = to_categorical(train_labels, num_classes=10)
y_test = to_categorical(test_labels, num_classes=10)

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(392, activation='relu'))
model.add(Dense(196, activation = 'relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0007, rho=0.9, decay=1e-4), metrics = ['accuracy'])

model.summary()

model.fit(x_train, y_train,
		 epochs = 40, 
		 batch_size=128,
		 validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test)
print('Test loss = '+score[0]+'\nTest accuracy = '+score[1])
