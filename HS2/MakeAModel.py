
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load your dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "./Data/Dtrain/",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(224, 224),
  batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "./Data/Dtest/",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(224, 224),
  batch_size=32
)

# Preprocess your data
data_augmentation = keras.Sequential(
  [    layers.RandomFlip("horizontal"),    layers.RandomRotation(0.1),    layers.RandomZoom(0.1),  ]
)

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

# Define your model
model = keras.Sequential(
    [
        keras.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

# Compile your model
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

# Train your model
model.fit(
    train_ds,  # Training data
    epochs=145,  # Number of epochs to train for
    validation_data=val_ds  # Validation data
)

model.save('./Downloads/')

"""

from pathlib import Path
import imghdr
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

data_dir = "Data/Dtrain/c0 ear/"
image_extensions = [".png", ".jpeg"]  # add there all your images file extensions

img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
for filepath in Path(data_dir).rglob("*"):
    if filepath.suffix.lower() in image_extensions:
        img_type = imghdr.what(filepath)
        if img_type is None:
            print(f"{filepath} is not an image")
        elif img_type not in img_type_accepted_by_tf:
            print(f"{filepath} is a {img_type}, not accepted by TensorFlow")


import os

# Specify the directory path containing the image files
directory = "./Data/Dtrain/c5 eye/"

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Get the file extension
    extension = os.path.splitext(filename)[1]

    # Check if the file is an image (JPEG, PNG, BMP, GIF)
    if extension.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
        print("Image file detected:", filename, "with extension", extension)
    else:
        print("Image file detected:", filename, "with extension", "Error")
        

"""
"""
import os
from PIL import Image

# Specify the directory path containing the image files
directory = "./Data/Dtest/c5 eye/"

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is an image (JPEG, PNG, BMP, GIF)
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        # Open the image file
        try:
            img = Image.open(os.path.join(directory, filename))
            img.load()
            print("Image file", filename, "is not corrupted.")
        except (IOError, SyntaxError) as e:
            print("Image file", filename, "is ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo corrupted.")
    else:
        print("Problem")"""
"""
import os

# specify the directory containing the files
directory = "../Video/"

# specify the file extension of the files to be renamed
file_extension = ".jpeg"

# initialize a counter
counter = 500

# iterate over all files in the directory
for filename in os.listdir(directory):
    print("asfd")
    # check if the file matches the extension we want to rename
    
        # construct the new file name
    new_filename = f"{counter}.PNG"
    print("Donmeeeeeeee")
        
        # rename the file
    os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
        
        # increment the counter
    counter += 1
print("Done")

"""
