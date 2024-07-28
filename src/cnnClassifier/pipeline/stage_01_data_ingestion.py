from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger

STAGE_NAME = "STAGE: Data Ingestion"


class DataIngestionTrainingPipeline:

    def __init__(self):
        pass

    
    def main(self):
        """
        Executes the data ingestion pipeline.

        This method performs the following steps:
        1. Logs the start of the data ingestion stage.
        2. Retrieves the configuration for data ingestion using the ConfigurationManager.
        3. Creates a DataIngestion object with the retrieved configuration.
        4. Downloads the file using the DataIngestion object.
        5. Extracts the zip file using the DataIngestion object.
        6. Logs the completion of the data ingestion stage.

        Raises:
            Exception: If any exception occurs during the execution of the data ingestion stage.

        Returns:
            None
        """
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


# for dvc pipeline tracking
if __name__ == "__main__":
    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.main()
    
    

