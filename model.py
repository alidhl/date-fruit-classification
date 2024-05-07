import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
class Model:
    def __init__(self):
        # Load the models
        # load_model() doesn't work for my model for some reason, so we have to manually build the model and load the weights 
        base_model = MobileNet(weights=None, include_top=False, input_shape=(200, 200, 3))
        base_model.trainable = False
        self.model = Sequential([
            base_model,
            Flatten(),
            Dense(512, activation='relu'),
            Dense(9, activation='softmax')
        ])
        # Manually build the model by passing a dummy inpuy
        dummy_input = np.zeros((1, 200, 200, 3))  
        self.model(dummy_input) 
        # Load the weights
        self.model.load_weights('model.keras')
        self.class_names = ['Ajwa', 'Galaxy', 'Mejdool', 'Meneifi', 'NabtatAli', 'Rutab', 'Shaishe', 'Sokari','Sugaey']
    
    def predict(self, image):
        img = image.resize((200, 200))  # Resize image to match model's expected input
        img_array = img_to_array(img)  # Convert to array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image array
        probabilities = self.model.predict(img_array)
        class_index = np.argmax(probabilities)  # Get the index of the highest probability
        class_label = self.class_names[class_index]  # Get the class label
        return class_label
    
# Test the model
if __name__ == '__main__':
    model = Model()
    image = Image.open('test/NabtatAli/NabtatAli001.jpg')# Expected output: 'NabtatAli'
    label = model.predict(image)
    print(label)