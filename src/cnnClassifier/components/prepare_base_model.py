from pathlib import Path
import tensorflow as tf


from cnnClassifier import logger
from cnnClassifier.config.configuration import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        """
        Initializes the base model for the classifier.

        This function creates a VGG16 model with the specified input shape, weights, and whether to include the top layer.
        The initialized model is then saved to the specified base model path.

        Parameters:
            None

        Returns:
            None
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        self.save_model(path=self.config.base_model_path, 
                        model= self.model
        )


    @staticmethod
    def _prepare_full_model(model: tf.keras.Model, 
                            classes: int, 
                            freeze_all: bool, 
                            freeze_till: int):
        """
        Prepare a full model by freezing certain layers of the given base model.

        Args:
            model (tf.keras.Model): The base model to prepare.
            classes (int): The number of classes for the output layer.
            freeze_all (bool): Whether to freeze all layers of the base model.
            freeze_till (int): The number of layers to freeze from the end of the base model.

        Returns:
            tf.keras.Model: The prepared full model.

        """
        if freeze_all: 
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_all > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # The model output (output of the convolutional layers) is flattened to 1D
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(classes, activation='softmax')(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction,
        )

        full_model.summary()
        return full_model
        
    
    def update_base_model(self):
        """
        Updates the base model by preparing a full model with frozen layers and saving it to the specified path.

        This function first prepares a full model by freezing all layers of the base model using the `_prepare_full_model` method.
        The `model` parameter is the base model to prepare, `classes` is the number of classes for the output layer,
        `freeze_all` is a boolean indicating whether to freeze all layers of the base model, and `freeze_till` is the number of layers
        to freeze from the end of the base model.

        After preparing the full model, it saves the model to the specified path using the `save_model` method.
        The `path` parameter is the path to save the model to, and `model` is the full model to save.

        If an exception is raised during the process, it logs the exception using the `logger` and re-raises it.

        Parameters:
            None

        Returns:
            None
        """
        try:
            self.full_model = self._prepare_full_model(
                model = self.model,
                classes = self.config.params_classes,
                freeze_all=True,
                freeze_till=None,
            )

            self.save_model(
                path=self.config.updated_base_model_path,
                model=self.full_model
            )

        except Exception as e:
            logger.exception(f"Exception raised while updating base model: {e}")
            raise e
        

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save a Keras model to a specified path.

        Args:
            path (Path): The path where the model will be saved.
            model (tf.keras.Model): The Keras model to be saved.

        Returns:
            None
        """
        model.save(path)

    
