from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import Training

STAGE_NAME = "Stage: Train Model"

class ModelTrainingPipeline:

    def __init__(self):
        pass

    
    def main(self):
        """
        The main function that trains a model using the Training class. It performs the following steps:
        
        1. Logs the start of the training stage.
        2. Retrieves the training configuration from the ConfigurationManager.
        3. Creates an instance of the Training class with the retrieved configuration.
        4. Retrieves the base model using the Training class.
        5. Generates the training and validation generators using the Training class.
        6. Trains the model using the Training class.
        7. Logs the completion of the training stage.
        
        Raises:
            Exception: If an exception occurs during the training process.
        """
        try:
            logger.info(f">>>>>>>>>>>>>> {STAGE_NAME} STARTED <<<<<<<<<<<<<<<")
            config = ConfigurationManager()
            training_config = config.get_training_config()
            training = Training(config=training_config)
            training.get_base_model()
            training.train_valid_generator()
            training.train()
            logger.info(f">>>>>>>>>>>>>> {STAGE_NAME} COMPLETED <<<<<<<<<<<<<<< \n")
            
        except Exception as e:
            logger.exception(f"Exception raised while running {STAGE_NAME}: {e}")
            raise e


# for dvc pipeline tracking
if __name__ == "__main__":
    model_training_pipeline = ModelTrainingPipeline()
    model_training_pipeline.main()