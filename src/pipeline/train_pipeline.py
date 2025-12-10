from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_training():
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    train_arr, test_arr = transformer.initiate_data_transformation(train_path, test_path)

    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_arr, test_arr)

if __name__ == "__main__":
    run_training()
