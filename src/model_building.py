# model_building.py
import logging

from sklearn.linear_model import LogisticRegression

from src import setup_logger

logger = setup_logger(__name__, log_file='model_building.log', level=logging.DEBUG)
logger.debug('src package initialized successfully.')

from src.utils import loading_data, save_sklearn_model


def create_model(x_train, y_train):
    """
    Create and train a Logistic Regression model.

    :param x_train: Training features
    :param y_train: Training labels
    :return: Trained Logistic Regression model
    """
    logreg_model = LogisticRegression(max_iter=1000, random_state=42)
    logreg_model.fit(x_train, y_train)
    return logreg_model


if __name__ == '__main__':
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

