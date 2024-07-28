import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from cnnClassifier import logger

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename


    def predict(self):
        """
        Predicts the class of an image using a pre-trained model.

        This method loads a pre-trained model, loads an image from the specified filename,
        resizes it to 224x224 pixels, converts it to a numpy array, and passes it through the
        model to obtain a prediction. The predicted class is then determined based on the
        maximum value in the prediction array.

        Parameters:
            self (PredictionPipeline): The instance of the PredictionPipeline class.

        Returns:
            list: A list containing a dictionary with a single key-value pair. The key is
            "image" and the value is the predicted class, either "Normal" or "Adenocarcinoma
            Cancer".

        Example:
            prediction_pipeline = PredictionPipeline("image.jpg")
            prediction = prediction_pipeline.predict()
            print(prediction)  # Output: [{"image": "Normal"}]
        """
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