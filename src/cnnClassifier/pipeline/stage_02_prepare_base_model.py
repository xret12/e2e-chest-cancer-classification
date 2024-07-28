from cnnClassifier import logger
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.config.configuration import ConfigurationManager



STAGE_NAME = "Stage: Prepare Base Model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        """
        Performs the Base Model Preparation pipeline.

        This function is the entry point of the `main` method. It performs the following steps:
        1. Logs a message indicating the start of the function.
        2. Creates an instance of the `ConfigurationManager` class.
        3. Retrieves the configuration for preparing the base model.
        4. Creates an instance of the `PrepareBaseModel` class.
        5. Calls the `get_base_model` method of the `PrepareBaseModel` instance.
        6. Calls the `update_base_model` method of the `PrepareBaseModel` instance.
        7. Logs a message indicating the completion of the function.

        Raises:
            Exception: If an error occurs during the execution of the function.

        """
        try:
            logger.info(f">>>>>>>>>>>>>> {STAGE_NAME} STARTED <<<<<<<<<<<<<<<")
            config = ConfigurationManager()
            prepare_base_model_config = config.prepare_base_model_config()
            prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
            prepare_base_model.get_base_model()
            prepare_base_model.update_base_model()
            logger.info(f">>>>>>>>>>>>>> {STAGE_NAME} COMPLETED <<<<<<<<<<<<<<< \n")
            
        except Exception as e:
            logger.exception(f"Exception raised while running {STAGE_NAME}: {e}")
            raise e
        

# for dvc pipeline tracking
if __name__ == "__main__":
    prepare_base_model_pipeline = PrepareBaseModelTrainingPipeline()
    prepare_base_model_pipeline.main()
    