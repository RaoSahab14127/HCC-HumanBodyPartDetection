
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.preprocessing import image
import shutil
import os

loaded_model = keras.models.load_model('./Downloads/')



## Loop
os.chdir("./Data/")
os.makedirs('Res')
path = "./train/test/"
images = os.listdir(path)
print(images)
for img_path in images:
    ro = img_path
    
    img_path = path+img_path
    
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # Make a prediction using the loaded model
    preds = loaded_model.predict(x)
    predicted_class = np.argmax(preds)
    print('Predicted class:', predicted_class)
    if (predicted_class == 0):
        os.chdir("../Data/Res/")
        lis = os.listdir()
        if ('c0') not in lis:
            print(lis)
            os.makedirs('c0')
        print("ok")
        shutil.copy2("../train/test/"+ro,"./c0/")
        os.chdir("../../Data/")
    
        
        
        print("ok")
    if (predicted_class == 1):
        os.chdir("../Data/Res/")
        lis = os.listdir()
        if ('c1') not in lis:
            print(lis)
            os.makedirs('c1')
        print("ok")
        shutil.copy2("../train/test/"+ro,"./c1/")
        os.chdir("../../Data/")
        
    if (predicted_class == 2):
        os.chdir("../Data/Res/")
        lis = os.listdir()
        if ('c2') not in lis:
            print(lis)
            os.makedirs('c2')
        print("ok")
        shutil.copy2("../train/test/"+ro,"./c2/")
        os.chdir("../../Data/")
    if (predicted_class == 3):
        os.chdir("../Data/Res/")
        lis = os.listdir()
        if ('c3') not in lis:
            print(lis)
            os.makedirs('c3')
        print("ok")
        shutil.copy2("../train/test/"+ro,"./c3/")
        os.chdir("../../Data/")
    if (predicted_class == 4):
        os.chdir("../Data/Res/")
        lis = os.listdir()
        if ('c4') not in lis:
            print(lis)
            os.makedirs('c4')
        shutil.copy2("../train/test/"+ro,"./c4/")
        os.chdir("../../Data/")
    if (predicted_class == 5):
        os.chdir("../Data/Res/")
        lis = os.listdir()
        if ('c5') not in lis:
            print(lis)
            os.makedirs('c5')
        print("ok")
        shutil.copy2("../train/test/"+ro,"./c5/")
        os.chdir("../../Data/")
    
    
