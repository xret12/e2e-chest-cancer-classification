from cnnClassifier import logger
from cnnClassifier.components.model_evaluation_mlflow import Evaluation
from cnnClassifier.config.configuration import ConfigurationManager

STAGE_NAME = "Stage: Model Evaluation"


class EvaluationPipeline:

    def __init__(self):
        pass


    def main(self):
        """
        This function performs the following steps:
        1. Logs the start of the stage.
        2. Retrieves the evaluation configuration from the ConfigurationManager.
        3. Creates an instance of the Evaluation class with the evaluation configuration.
        4. Calls the evaluate method of the Evaluation instance.
        5. Activates MLFlow tracking.
        6. Logs the completion of the stage.

        Raises:
            Exception: If any exception occurs during the execution of the function.

        Returns:
            None
        """
        try:
            logger.info(f">>>>>>>>>>>>>> {STAGE_NAME} STARTED <<<<<<<<<<<<<<<")
            config = ConfigurationManager()
            evaluation_config = config.get_evaluation_config()
            evaluation = Evaluation(config=evaluation_config)
            evaluation.evaluate()
            evaluation.log_into_mlflow()
            logger.info(f">>>>>>>>>>>>>> {STAGE_NAME} COMPLETED <<<<<<<<<<<<<<<\n")
            
        except Exception as e:
            logger.exception(f"Exception raised while running {STAGE_NAME}: {e}")
            raise e
        

# for dvc pipeline tracking
if __name__ == "__main__":
    model_evaluation_pipeline = EvaluationPipeline()
    model_evaluation_pipeline.main()