from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Load the saved model
saved_model_path = "D:\FinalProject\signlanguagedetectionmodel48x48.h5"
loaded_model = load_model(saved_model_path)

# Define the validation data directory
validation_data_directory = r"D:\FinalProject\splitdataset48x48\val"

# Define the validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 128
validation_generator = val_datagen.flow_from_directory(
    validation_data_directory,
    target_size=(48, 48),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

# Compile the loaded model if necessary
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Evaluate the loaded model on the validation data
evaluation = loaded_model.evaluate(validation_generator)

# Print the accuracy
print("Accuracy:", evaluation[1])
