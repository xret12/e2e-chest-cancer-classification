import mlflow
import mlflow.keras
from pathlib import Path
import tensorflow as tf
from urllib.parse import urlparse


from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json



class Evaluation:

    def __init__(self, config: EvaluationConfig):
        self.config = config


    def _valid_generator(self):
        """
        Generates a data generator for the validation set.

        This method creates a data generator for the validation set using the `ImageDataGenerator` class from the `tensorflow.keras.preprocessing.image` module. 
        It sets the `rescale` parameter to `1./255` and the `validation_split` parameter to `0.30`.
        It also sets the `target_size` parameter to `self.config.params_image_size[:-1]`, the `batch_size` parameter to `self.config.params_batch_size`,
            and the `interpolation` parameter to `"bilinear"`.

        The generated data generator is then used to flow data from the specified directory, using the `validation` subset and without shuffling. 
        The `dataflow_kwargs` dictionary is unpacked to pass additional keyword arguments to the `flow_from_directory` method.

        This method does not return anything. Instead, it sets the `self.valid_generator` attribute to the generated data generator.
        """
         
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split = 0.30
        )

        dataflow_kwargs = dict(
            target_size = self.config.params_image_size[:-1],
            batch_size = self.config.params_batch_size,
            interpolation = "bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Load a Keras model from the specified path.

        Args:
            path (Path): The path to the model file.

        Returns:
            tf.keras.Model: The loaded Keras model.
        """
        return tf.keras.models.load_model(path)
    

    def evaluate(self):
        """
        Evaluate the model using the validation data.

        This function loads the model from the specified path using the `load_model` method. 
        It then generates a data generator for the validation set using the `_valid_generator` method. 
        The model is evaluated using the generated data generator and the evaluation score is stored in the `score` attribute. 
        Finally, the evaluation score is saved using the `save_score` method.

        Parameters:
            None

        Returns:
            None
        """
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()


    def save_score(self):
        """
        Save the evaluation score of the model to a JSON file.

        This function saves the evaluation score of the model to a JSON file named "scores.json". The score is stored in a dictionary with keys "loss" and "accuracy", corresponding to the first and second elements of the `self.score` attribute, respectively. The `save_json` function from the `utils` module is used to save the dictionary to the file.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        scores = {
            "loss": self.score[0],
            "accuracy": self.score[1]
        }
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        """
        Logs the evaluation metrics and the model into MLFlow.

        This function sets the MLFlow registry URI and retrieves the tracking URI to determine the type of store being used.
        It then starts a new run in MLFlow and logs the evaluation parameters and metrics.
        If the store is not a file store, the model is registered in the MLFlow Model Registry.
        The model is logged into MLFlow with the name "model" and the registered model name "VGG16Model".

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(tracking_url_type_store)

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1]
            })
            
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")