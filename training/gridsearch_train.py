import sklearn
from sklearn import *
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold


def train_sklearn_grid_search(x_train, y_train, model_type, hyperparameters):
    """ Sklearn Model with GridSearch

    Train a sklearn classification/regression model by picking the best values from a parameter grid
    using sklearn.model_selection.GridSearchCV.

    :param np.array x_train: data features
    :param np.array y_train: data target variable
    :param str model_type: full model name from sklearn library. Examples:
            'sklearn.ensemble.AdaBoostClassifier', 'sklearn.linear_model.LogisticRegression'
    :param dict hyperparameters: contains the following information:\n
            - `dict params_grid`: grid (of not yet determined parameters) used from `GridSearchCV` instance\n
            - `dict params_fixed`: fixed parameters of the model\n
            - `dict params_cv`: parameters for cross validation applied to x_train, y_train\n
            - `objective`: `regression` or  `classification`\n
            - `grid_search_scoring`: scoring functions for grid search
    :return: estimator
    """

    names = model_type.split('.')
    model = sklearn
    for name in names[1:]:
        model = getattr(model, name)
    model = model(**hyperparameters.get('params_fixed', {}))

    if hyperparameters["objective"] == "classification":
        cv = StratifiedKFold(**hyperparameters.get('params_cv', {}))
    else:
        cv = KFold(**hyperparameters.get('params_cv', {}))

    gs_rf = GridSearchCV(model,
                         hyperparameters['params_grid'],
                         scoring=hyperparameters["grid_search_scoring"],
                         cv=cv,
                         n_jobs=-1,
                         refit=hyperparameters["grid_search_scoring"][0])

    gs_rf.fit(x_train, y_train)

    return gs_rf.best_estimator_
