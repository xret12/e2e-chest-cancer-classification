from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger

STAGE_NAME = "STAGE: Data Ingestion"


class DataIngestionTrainingPipeline:

    def __init__(self):
        pass

    
    def main(self):
        try:
            logger.info(f">>>>>>>>>>>>>> {STAGE_NAME} STARTED <<<<<<<<<<<<<<<")
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
            logger.info(f">>>>>>>>>>>>>> {STAGE_NAME} COMPLETED <<<<<<<<<<<<<<< \n")


        except Exception as e:
            logger.exception(f"Exception raised while running {STAGE_NAME}: {e}")
            raise e


    
    

