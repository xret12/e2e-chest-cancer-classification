from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import Training

STAGE_NAME = "Stage: Train Model"

class ModelTrainingPipeline:

    def __init__(self):
        pass

    
    def main(self):
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