from pathlib import Path
from cnnClassifier.constants import *
from cnnClassifier.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, \
                                                TrainingConfig, EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        """
        Initializes a new instance of the ConfigurationManager class.

        Args:
            config_filepath (str, optional): The path to the configuration file. Defaults to CONFIG_FILE_PATH.
            params_filepath (str, optional): The path to the parameters file. Defaults to PARAMS_FILE_PATH.

        Returns:
            None

        Raises:
            FileNotFoundError: If the configuration file or parameters file cannot be found.

        Description:
            This constructor initializes the ConfigurationManager object by reading the configuration and parameters
            files specified by the provided filepaths. It then creates the necessary directories specified in the
            configuration file.

        Note:
            - The configuration file and parameters file should be in YAML format.
            - The artifacts root directory specified in the configuration file will be created if it does not exist.

        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves the configuration for data ingestion.

        Returns:
            DataIngestionConfig: The configuration for data ingestion, including the root directory, source URL,
            local data file, and unzip directory.

        Description:
            This function retrieves the data ingestion configuration from the `data_ingestion` section of the
            configuration file. It creates the necessary directories specified in the configuration file and returns
            a `DataIngestionConfig` object with the retrieved values.

        Note:
            - The `data_ingestion` section of the configuration file should contain the following keys:
                - `root_dir`: The root directory for data ingestion.
                - `source_url`: The URL of the data source.
                - `local_data_file`: The local file path where the data will be saved.
                - `unzip_dir`: The directory where the downloaded data will be extracted.

        """
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    
    
    def prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """
        Prepare the configuration for the base model.

        Returns:
            PrepareBaseModelConfig: The configuration for the base model, including the root directory, base model path,
            updated base model path, image size, include top, weights, and classes.

        Description:
            This function retrieves the configuration for the base model from the `prepare_base_model` section of the
            configuration file. It creates the necessary directories specified in the configuration file and returns
            a `PrepareBaseModelConfig` object with the retrieved values.

        Note:
            - The `prepare_base_model` section of the configuration file should contain the following keys:
                - `root_dir`: The root directory for the base model.
                - `base_model_path`: The path to the base model.
                - `updated_base_model_path`: The path to the updated base model.
                - `params_image_size`: The image size for the base model.
                - `params_include_top`: Whether to include the top layer of the base model.
                - `params_weights`: The weights for the base model.
                - `params_classes`: The number of classes for the base model.
        """
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=config.root_dir,
            base_model_path=config.base_model_path,
            updated_base_model_path=config.updated_base_model_path,
            params_image_size=self.params.IMAGE_SIZE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    

    def get_training_config(self) -> TrainingConfig:
        """
        Retrieves the training configuration from the configuration file and creates the necessary directories.

        Returns:
            TrainingConfig: The training configuration object containing the root directory, trained model path,
            trained model path for tracking, updated base model path, training data, epochs, batch size,
            augmentation flag, image size, and learning rate.

        Description:
            This function retrieves the training configuration from the `training` section of the configuration file.
            It also retrieves the `updated_base_model_path` from the `prepare_base_model` section of the configuration file.
            The function creates the necessary directories specified in the configuration file using the `create_directories` function.
            It then creates a `TrainingConfig` object with the retrieved values and returns it.

        Note:
            - The `training` section of the configuration file should contain the following keys:
                - `root_dir`: The root directory for the training.
                - `trained_model_path`: The path to the trained model.
                - `trained_model_path_for_tracking`: The path to the trained model for tracking.
                - `training_data`: The path to the training data.
            - The `prepare_base_model` section of the configuration file should contain the following key:
                - `updated_base_model_path`: The path to the updated base model.
            - The `params` section of the configuration file should contain the following keys:
                - `EPOCHS`: The number of epochs for training.
                - `BATCH_SIZE`: The batch size for training.
                - `AUGMENTATION`: Whether to apply augmentation to the training data.
                - `IMAGE_SIZE`: The image size for training.
                - `LEARNING_RATE`: The learning rate for training.
        """
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        create_directories([self.config.training.root_dir])

        training_config = TrainingConfig(
            root_dir=training.root_dir,
            trained_model_path=training.trained_model_path,
            trained_model_path_for_tracking=training.trained_model_path_for_tracking,
            updated_base_model_path = prepare_base_model.updated_base_model_path,
            training_data=training.training_data,
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_is_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
        )

        return training_config
    

    def get_evaluation_config(self) -> EvaluationConfig:
        """
        Retrieves the evaluation configuration from the configuration object.

        Returns:
            EvaluationConfig: The evaluation configuration object.

        Description:
            This function retrieves the evaluation configuration from the `training` and `evaluation` sections of the configuration object.
            It creates an `EvaluationConfig` object with the following parameters:
                - `path_of_model`: The path to the trained model.
                - `training_data`: The path to the training data.
                - `all_params`: The parameters dictionary from the configuration object.
                - `mlflow_uri`: The MLflow URI from the evaluation section of the configuration object.
                - `params_image_size`: The image size for training.
                - `params_batch_size`: The batch size for training.

            The function then returns the created `EvaluationConfig` object.

        Note:
            - The `training` section of the configuration object should contain the following keys:
                - `trained_model_path`: The path to the trained model.
                - `training_data`: The path to the training data.
            - The `evaluation` section of the configuration object should contain the following key:
                - `mlflow_uri`: The MLflow URI.
            - The `params` section of the configuration object should contain the following keys:
                - `IMAGE_SIZE`: The image size for training.
                - `BATCH_SIZE`: The batch size for training.
        """
        training = self.config.training
        evaluation = self.config.evaluation
        
        evaluation_config = EvaluationConfig(
            path_of_model=training.trained_model_path,
            training_data=training.training_data,
            all_params=self.params,
            mlflow_uri=evaluation.mlflow_uri,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )

        return evaluation_config