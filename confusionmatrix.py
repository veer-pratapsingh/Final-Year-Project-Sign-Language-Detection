import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator

# Define the directory where your model and its architecture are saved
directory = r"D:\FinalProject"

# Load the saved model architecture from JSON file
json_file_path = os.path.join(directory, "signlanguagedetectionmodel48x48.json")
loaded_model = model_from_json(open(json_file_path, "r").read())

# Load the saved model weights
h5_file_path = os.path.join(directory, "signlanguagedetectionmodel48x48.h5")
loaded_model.load_weights(h5_file_path)

# Compile the loaded model
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the validation data and create a generator
val_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = val_datagen.flow_from_directory(
    os.path.join(directory, "splitdataset48x48", "val"),
    target_size=(48, 48),
    batch_size=128,
    class_mode='categorical',
    color_mode='grayscale'
)

# Make predictions on the validation data
predictions = loaded_model.predict(validation_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = validation_generator.classes

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
