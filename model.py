import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
class Model:
    def __init__(self):
        # Load the models
        self.model = tf.keras.models.load_model('model.keras')
        self.class_names = ['Ajwa', 'Galaxy', 'Mejdool', 'Meneifi', 'NabtatAli', 'Rutab', 'Shaishe', 'Sokari','Sugaey']
    
    def predict(self, image):
        img = image.resize((150, 150))  # Resize image to match model's expected input
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