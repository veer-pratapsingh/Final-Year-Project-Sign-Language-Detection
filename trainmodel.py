from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import os

train_datagen = ImageDataGenerator(
    rescale=1./255,
)

val_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 128

train_generator = train_datagen.flow_from_directory(
   r"D:\FinalProject\splitdataset48x48\train",
    target_size=(48, 48),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

validation_generator = val_datagen.flow_from_directory(
    r'D:\FinalProject\splitdataset48x48\val',
    target_size=(48, 48),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)
class_names = list(train_generator.class_indices.keys())
print(class_names)

model = Sequential()
# convolutional layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
# fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(27, activation='softmax'))

model.summary()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy' )

import os
from keras.callbacks import TensorBoard

# Define the directory for logs
logs_dir = "Logs"

# Remove the "Logs" directory if it exists
if os.path.exists(logs_dir):
    import shutil
    shutil.rmtree(logs_dir)

# Create a new "Logs" directory
os.makedirs(logs_dir)

# Create TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=logs_dir)

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=23,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[tensorboard_callback]
)



import os
from keras.models import model_from_json

# Convert model to JSON
model_json = model.to_json()

# Define the directory for saving the model
directory = r"D:\FinalProject"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Save model architecture as JSON file
json_file_path = os.path.join(directory, "signlanguagedetectionmodel48x48.json")
with open(json_file_path, 'w') as json_file:
    json_file.write(model_json)

# Save model weights and configuration to a single HDF5 file
h5_file_path = os.path.join(directory, "signlanguagedetectionmodel48x48.h5")
model.save(h5_file_path)

# Example of loading the model from the saved files
loaded_model = model_from_json(open(json_file_path, "r").read())
loaded_model.load_weights(h5_file_path)


evaluation = loaded_model.evaluate(validation_generator)

# Print the accuracy
print("Accuracy:", evaluation[1])