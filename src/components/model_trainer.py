import os
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from scipy.sparse import issparse
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC



@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
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

            models = {
                "LogisticRegression": LogisticRegression(
                    max_iter=1000,
                    solver="saga",
                ),
                "LinearSVC": LinearSVC(dual=False),
                "SGDClassifier": SGDClassifier(
                    max_iter=2000,
                    tol=1e-3,
                )
            }

            params = {
                "LogisticRegression": {
                    "penalty": ["l1", "l2"],
                    "C": [0.01, 0.03, 0.1, 0.25, 0.5, 1, 2, 4, 8, 10],
                    "class_weight": [None, "balanced"],
                },
                "LinearSVC": {
                    "C": [0.01, 0.03, 0.1, 0.25, 0.5, 1, 2, 4, 8, 10],
                    "class_weight": [None, "balanced"],
                },
                "SGDClassifier": {
                    "loss": ["hinge", "log_loss", "modified_huber"],
                    "alpha": [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                    "penalty": ["l2", "l1", "elasticnet"],
                }
            }


            model_report: dict = evaluate_models(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                models=models,
                params=params,
            )

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(
                f"Best model on train/test: {best_model_name} "
                f"with score {best_model_score:.4f}"
            )

           
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            save_object(
                file_path=self.model_trainer_config.model_file_path,
                obj=best_model,
            )

            logging.info(f"Trained model: {best_model}")
            logging.info(f"Accuracy score: {acc:.4f}")
            logging.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
            logging.info(
                f"Classification report:\n{classification_report(y_test, y_pred)}"
            )
            logging.info(
                f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}"
            )

            return acc

        except Exception as e:
            raise CustomException(e, sys)
