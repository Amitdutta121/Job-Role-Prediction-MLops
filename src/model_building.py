import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mlflow
import mlflow.sklearn
import logging
from sklearn.linear_model import LogisticRegression
import random

from src import setup_logger, constants

logger = setup_logger(__name__, log_file='model_building.log', level=logging.DEBUG)
logger.debug('src package initialized successfully.')

from src.utils import loading_data, save_sklearn_model


from sklearn.ensemble import RandomForestClassifier

def create_model(x_train, y_train):
    """
    Create and train a Random Forest Classifier.

    :param x_train: Training features
    :param y_train: Training labels
    :return: Trained Random Forest model
    """
    rf_model = RandomForestClassifier(
        n_estimators=random.randint(50, 200),
        max_depth=None,
        random_state=42,
        n_jobs=-1  # Use all cores
    )
    rf_model.fit(x_train, y_train)
    return rf_model


if __name__ == '__main__':
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(constants.MLflow.experiment_name)
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name=constants.MLflow.run_training):
        train_df, test_df = loading_data('interim')
        logger.debug('Data loaded successfully for model building.')
        x_train = train_df.drop(columns=['Category'])
        y_train = train_df['Category']
        x_test = test_df.drop(columns=['Category'])
        y_test = test_df['Category']

        model = create_model(x_train, y_train)
        logger.debug('Model trained successfully.')
        # log accuracy precision f`1
        accuracy = model.score(x_test, y_test)
        logger.info(f'Model accuracy: {accuracy:.2f}')
        logger.debug('Model accuracy logged successfully.')

        save_sklearn_model(model, 'LG/logistic_regression_model.pkl')
        print("-------------------==========================================================")

