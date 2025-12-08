import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from dataclasses import dataclass
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Initiating model trainer")
            
            if issparse(train_arr):
                train_arr = train_arr.tocsr()
                X_train = train_arr[:, :-1]
                y_train = train_arr[:, -1].toarray().ravel()
            else:
                X_train = train_arr[:, :-1]
                y_train = train_arr[:, -1]

            if issparse(test_arr):
                test_arr = test_arr.tocsr()
                X_test = test_arr[:, :-1]
                y_test = test_arr[:, -1].toarray().ravel()
            else:
                X_test = test_arr[:, :-1]
                y_test = test_arr[:, -1]

            # Focus on the fastest performant models
            models = {
                LogisticRegression: LogisticRegression(max_iter=400, solver="saga"),
                LinearSVC: LinearSVC(dual=False),
            }

            params = {
                LogisticRegression: {
                    "penalty": ["l1", "l2"],
                    "C": [0.25, 0.5, 1, 2, 4],
                    "class_weight": [None, "balanced"],
                },
                LinearSVC: {
                    "C": [0.25, 0.5, 1, 2, 4],
                },
                # Removed RandomForest to speed up training
            }


            model_report: dict = evaluate_models(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models = models,params = params)

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            save_object(
                file_path=self.model_trainer_config.model_file_path,
                obj=best_model
                )

            logging.info(f"best model name is {best_model_name} and its score is {best_model_score}")

            best_model.fit(X_train,y_train)
            y_pred = best_model.predict(X_test)
            acc = accuracy_score(y_test,y_pred)

            print(best_model)
            print("Accuracy score {:.4f}".format(acc))
            print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
            print("Classification report:\n", classification_report(y_test, y_pred))
            print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))

            return acc
        
        except Exception as e:
            raise CustomException(e,sys)