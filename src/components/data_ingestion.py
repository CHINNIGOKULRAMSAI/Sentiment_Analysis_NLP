import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_path: str = os.path.join("artifacts", "train.csv")
    test_path: str = os.path.join("artifacts", "test.csv")
    raw_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        try:
            logging.info("Entered into data ingestion")

            data = pd.read_csv("notebook/data/all_kindle_review.csv")
            df = data[['reviewText', 'rating']].copy()

            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df = df.dropna(subset=['rating'])
            df['rating'] = df['rating'].astype(int)
            df['label'] = df['rating'].apply(lambda x: 0 if x < 3 else 1)

            train_set, test_set = train_test_split(
                df,
                test_size=0.25,
                random_state=42,
                stratify=df['label']
            )

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_path, index=False, header=True)

            train_set.to_csv(self.data_ingestion_config.train_path, index=False, header= True)

            test_set.to_csv(self.data_ingestion_config.test_path, index=False, header= True)

            logging.info("Data ingestion is completed")

            return (
                self.data_ingestion_config.train_path,
                self.data_ingestion_config.test_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == '__main__':
    DataIngestion = DataIngestion()
    train_path, test_path = DataIngestion.initiate_data_ingestion()

    DataTransformation = DataTransformation()
    DataTransformation.initiate_data_transformation(train_path, test_path)