from pathlib import Path
import tensorflow as tf

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:

    def __init__(self, config: TrainingConfig):
        self.config = config


    def get_base_model(self):
        """
        Load the base model from the specified path and compile it with the specified optimizer, loss function, and metrics.

        Parameters:
            None

        Returns:
            None
        """
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path, compile=False)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )


    def train_valid_generator(self):
        """
        Initializes the training and validation generators for the model.

        This function sets up the training and validation generators for the model by loading the data from the specified directory.
        The data is preprocessed using the ImageDataGenerator class from TensorFlow. 
        The generators are configured with the specified parameters such as rescaling, validation split, target size, batch size, and interpolation.
        """
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split = 0.20
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

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    
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


    def train(self):
        """
        Trains the model using the train and validation generators, and saves the trained model to two different paths.

        This function calculates the number of steps per epoch and the number of validation steps based on the batch size of the train and validation generators. 
        It then logs the class indices of the train and validation generators.

        The model is trained using the train generator with the specified number of epochs and steps per epoch. 
        The validation data is provided by the validation generator with the specified number of validation steps.

        After training, the model is saved to two different paths: the configured trained model path and the configured trained model path for tracking.

        Parameters:
            self (ModelTrainer): The instance of the ModelTrainer class.
        
        Returns:
            None
        """
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        logger.info(f"Class indices (train): {self.train_generator.class_indices}")
        logger.info(f"Class indices (valid): {self.valid_generator.class_indices}")

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
        # self to another path for tracking in github
        self.save_model(
            path=self.config.trained_model_path_for_tracking,
            model=self.model
        )