import gdown
import zipfile

from cnnClassifier import logger
from cnnClassifier.utils.common import create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self):
        """
        Downloads a file from a specified URL and saves it to a local directory.

        This method attempts to download a file from a given URL and save it to a specified local directory.
        The URL is obtained from the `source_url` attribute of the `config` object, while the local directory
        is obtained from the `local_data_file` attribute of the `config` object.

        Parameters:
            self (DataIngestion): The instance of the DataIngestion class.

        Returns:
            None

        Raises:
            Exception: If an error occurs during the download process, the exception is logged and re-raised.

        """
        try:
            dataset_url = self.config.source_url
            zip_download_dir = self.config.local_data_file
            create_directories([self.config.root_dir])
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix+file_id, zip_download_dir)
            logger.info(f"Finished downloading data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            logger.exception(f"Exception raised while downloading data from {dataset_url} into file {zip_download_dir}: {e}")
            raise e
        

    def extract_zip_file(self):
        """
        Extracts the contents of a zip file to a specified directory.

        This method takes the downloaded zip file from the `local_data_file` attribute of the `config` object
        and extracts its contents to the directory specified by the `unzip_dir` attribute of the `config` object.
        The `unzip_dir` directory is created if it does not exist.

        Parameters:
            self (DataIngestion): The instance of the DataIngestion class.

        Returns:
            None

        Raises:
            Exception: If an error occurs during the extraction process, the exception is logged and re-raised.
        """
        try:
            unzip_path = self.config.unzip_dir 
            zip_download_dir = self.config.local_data_file
            create_directories([unzip_path])
            with zipfile.ZipFile(zip_download_dir, 'r') as z:
                z.extractall(unzip_path)
            logger.info(f"Unzipped downloaded file in {zip_download_dir} to {unzip_path}")
        
        except Exception as e:
            logger.exception(f"Exception raised while unzipping downloaded file in {zip_download_dir} to {unzip_path}: {e}")
            raise e