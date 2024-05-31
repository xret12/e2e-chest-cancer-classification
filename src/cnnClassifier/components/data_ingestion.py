import gdown
import zipfile

from cnnClassifier import logger
from cnnClassifier.utils.common import get_size, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self):
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