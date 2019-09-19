from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def train_rf_grid_search(x_train, y_train, hyperparameters, required_metrics):
    """ Random Forest classifier

    Train a Random Forest classification model picking the best values from a parameter grid
    using sklearn.model_selection.GridSearchCV.

    :param np.array x_train: data features
    :param np.array y_train: data target variable
    :param dict hyperparameters: contains the following information:\n
            - `dict rf_grid`: grid (of not yet determined parameters) used from `GridSearchCV` instance\n
            - `dict rf_fixed_params`: fixed parameters of the model\n
            - `dict StratifiedKFold_params`:\n
    :param required_metrics: information about splitting
    :return: estimator
    """

    model = RandomForestClassifier(**hyperparameters.get('rf_fixed_params', {}))

    metrics = list(map(lambda x: x.replace('_score', ''), required_metrics))

    my_cv = StratifiedKFold(**hyperparameters.get('StratifiedKFold_params', {}))

    gs_rf = GridSearchCV(model,
                         hyperparameters['rf_grid'],
                         scoring=metrics,
                         cv=my_cv,
                         n_jobs=-1,
                         refit=metrics[0])

    gs_rf.fit(x_train, y_train)

    return gs_rf.best_estimator_
