from pathlib import Path
from cnnClassifier.constants import *
from cnnClassifier.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, \
                                                TrainingConfig, EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
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
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=config.root_dir,
            base_model_path=config.base_model_path,
            updated_base_model_path=config.updated_base_model_path,
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        create_directories([self.config.training.root_dir])

        training_config = TrainingConfig(
            root_dir=training.root_dir,
            trained_model_path=training.trained_model_path,
            updated_base_model_path = prepare_base_model.updated_base_model_path,
            training_data=training.training_data,
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_is_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE,
        )

        return training_config
    

    def get_evaluation_config(self) -> EvaluationConfig:
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