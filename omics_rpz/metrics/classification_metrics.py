"""File to regroup classification metrics."""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_binary_auc(y_true, y_score) -> np.float64:
    """Compute binary ROC AUC.

    Default computation setting:
        'macro':
            Calculate metrics for each label, and find their unweighted mean.
            This does not take label imbalance into account.
    Models that do not return a probablity should do a sigmoid function.

    Args:
        y_true (array-like of shape (n_samples,) or (n_samples, n_classes)):
            True labels or binary label indicators. The binary and multiclass cases
            expect labels with shape (n_samples,) while the multilabel case expects
            binary label indicators with shape (n_samples, n_classes).
        y_score (array-like of shape (n_samples,) or (n_samples, n_classes)):
            Target scores, probability output of the model.
            In the binary case, it corresponds to an array of shape (n_samples,).
                Both probability estimates and non-thresholded decision values can be
                provided. The probability estimates correspond to the probability of the
                class with the greater label, i.e. estimator.classes_[1] and thus
                estimator.predict_proba(X, y)[:, 1]. The decision values corresponds to
                the output of estimator.decision_function(X, y).
                See more information in the User guide;
            In the multiclass case, it corresponds to an array of shape
                (n_samples,n_classes) of probability estimates provided by the
                predict_proba method. The probability estimates must sum to 1 across the
                possible classes. In addition, the order of the class scores must
                correspond to the order of labels, if provided, or else to the numerical
                or lexicographical order of the labels in y_true.
                See more information in the User guide;
            In the multilabel case, it corresponds to an array of shape
                (n_samples, n_classes). Probability estimates are provided by the
                predict_proba method and the non-thresholded decision values by the
                decision_function method. The probability estimates correspond to the
                probability of the class with the greater label for each output of the
                classifier. See more information in the User guide.
    Returns:
        np.float64: Area Under the Curve score.
    """
    # Models that do not return a probablity should do a sigmoid function.
    try:
        return roc_auc_score(y_true, y_score)
    except AssertionError:
        # If only 0 in the given fold - rocauc cannot be computed so we return 0.5.
        return 0.5
    except ValueError:
        return np.nan


def compute_accuracy(y_true: np.array, y_score: np.array) -> np.float64:
    """Accuracy classification score.

    Args:
        y_true (np.array): Ground truth (correct) labels.
        y_score (np.array): Predicted labels, as returned by a classifier.
            This is the sklearn output of predict_proba - with two columns if binary
            classification problem.

    Returns:
        np.float64: return the fraction of correctly classified samples (float).
    """
    # If multiclassification select max columns.
    pred = np.expand_dims(np.argmax(y_score, axis=1), axis=1)
    return accuracy_score(y_true, pred)
