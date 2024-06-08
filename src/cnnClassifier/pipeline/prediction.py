import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from cnnClassifier import logger

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename


    def predict(self):
        model = load_model(os.path.join("model", "model.h5"))

        image_name = self.filename
        test_image = image.load_img(image_name, target_size=(224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)
        logger.info(f"RAW PREDICTION RESULT: {result}")
        class_result = np.argmax(result, axis=1)
        

        if class_result[0] == 1:
            prediction = "Normal"
        else:
            prediction = "Adenocarcinoma Cancer"

        return [{"image": prediction}]
        # return [{"result": result}]