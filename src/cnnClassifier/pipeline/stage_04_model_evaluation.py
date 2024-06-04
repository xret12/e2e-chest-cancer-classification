from cnnClassifier import logger
from cnnClassifier.components.model_evaluation_mlflow import Evaluation
from cnnClassifier.config.configuration import ConfigurationManager

STAGE_NAME = "Stage: Model Evaluation"


class EvaluationPipeline:

    def __init__(self):
        pass


    def main(self):
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