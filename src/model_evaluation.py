import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import logging
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src import setup_logger

logger = setup_logger(__name__, log_file='model_evaluation.log', level=logging.DEBUG)
logger.debug('src package initialized successfully.')

from src.utils import loading_data, from_root, save_metrics, load_sklearn_model
import constants


def evaluate_model(clf, X_test, y_test):
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation completed with metrics: %s', metrics_dict)
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise



if __name__ == '__main__':
    mlflow.set_tracking_uri(constants.MLflow.tracking_uri)
    mlflow.set_experiment(constants.MLflow.experiment_name)

    with mlflow.start_run(run_name=constants.MLflow.run_evaluation):
        try:
            clf = load_sklearn_model('LG/logistic_regression_model.pkl')
            logger.debug('Model loaded for evaluation.')

            train_df, test_df = loading_data('interim')
            X_test = test_df.drop(columns=['Category'])
            y_test = test_df['Category']

            metrics = evaluate_model(clf, X_test, y_test)

            # Log metrics to MLflow
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Save metrics JSON locally and log as artifact
            metrics_path = from_root('reports', 'metrics.json')
            save_metrics(metrics, metrics_path)
            mlflow.log_artifact(metrics_path)

            logger.info('Model evaluation complete and logged to MLflow.')
        except Exception as e:
            logger.error('Error during model evaluation: %s', e)
            raise e