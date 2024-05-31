from pathlib import Path
import tensorflow as tf


from cnnClassifier import logger
from cnnClassifier.config.configuration import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
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
                            freeze_till: int,
                            learnining_rate: float):
            if freeze_all: 
                for layer in model.layers:
                    layer.trainable = False
            elif (freeze_till is not None) and (freeze_all > 0):
                for layer in model.layers[:-freeze_till]:
                    layer.trainable = False

            # The model output (output of the convolutional layers) is flattened to 1D
            flatten_in = tf.keras.layers.Flatten()(model.output)
            # Dense layer is applied on the output of flatten_in
            prediction = tf.keras.layers.Dense(
                units=classes,
                activation="softmax"
            )(flatten_in)  

            full_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=prediction
            )

            full_model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=learnining_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["accuracy"]
            )

            full_model.summary()
            return full_model
        
    
    def update_base_model(self):
        try:
            self.full_model = self._prepare_full_model(
                model = self.model,
                classes = self.config.params_classes,
                freeze_all=True,
                freeze_till=None,
                learnining_rate=self.config.params_learning_rate
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
        model.save(path)

    
