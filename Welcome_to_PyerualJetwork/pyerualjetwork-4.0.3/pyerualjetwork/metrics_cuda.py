import cupy as cp

def metrics(y_ts, test_preds, average='weighted'):
    """
    Calculates precision, recall and F1 score for a classification task.
    
    Args:
        y_ts (list or numpy.ndarray): True labels.
        test_preds (list or numpy.ndarray): Predicted labels.
        average (str): Type of averaging ('micro', 'macro', 'weighted').

    Returns:
        tuple: Precision, recall, F1 score.
    """
    
    from .data_operations_cuda import decode_one_hot
    
    y_test_d = decode_one_hot(y_ts)
    y_test_d = cp.array(y_test_d)
    y_pred = cp.array(test_preds)

    if y_test_d.ndim > 1:
        y_test_d = y_test_d.reshape(-1)
    if y_pred.ndim > 1:
        y_pred = y_pred.reshape(-1)

    tp = {}
    fp = {}
    fn = {}

    classes = cp.unique(cp.concatenate((y_test_d, y_pred)))

    for c in classes:
        tp[c] = 0
        fp[c] = 0
        fn[c] = 0

    for c in classes:
        for true, pred in zip(y_test_d, y_pred):
            if true == c and pred == c:
                tp[c] += 1
            elif true != c and pred == c:
                fp[c] += 1
            elif true == c and pred != c:
                fn[c] += 1

    precision = {}
    recall = {}
    f1 = {}

    for c in classes:
        precision[c] = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0
        recall[c] = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0
        f1[c] = 2 * (precision[c] * recall[c]) / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0

    if average == 'micro':
        precision_val = cp.sum(list(tp.values())) / (cp.sum(list(tp.values())) + cp.sum(list(fp.values()))) if (cp.sum(list(tp.values())) + cp.sum(list(fp.values()))) > 0 else 0
        recall_val = cp.sum(list(tp.values())) / (cp.sum(list(tp.values())) + cp.sum(list(fn.values()))) if (cp.sum(list(tp.values())) + cp.sum(list(fn.values()))) > 0 else 0
        f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0

    elif average == 'macro':
        precision_val = cp.mean(list(precision.values()))
        recall_val = cp.mean(list(recall.values()))
        f1_val = cp.mean(list(f1.values()))

    elif average == 'weighted':
        weights = cp.array([cp.sum(y_test_d == c) for c in classes])
        weights = weights / cp.sum(weights)
        precision_val = cp.sum([weights[i] * precision[classes[i]] for i in range(len(classes))])
        recall_val = cp.sum([weights[i] * recall[classes[i]] for i in range(len(classes))])
        f1_val = cp.sum([weights[i] * f1[classes[i]] for i in range(len(classes))])

    else:
        raise ValueError("Invalid value for 'average'. Choose from 'micro', 'macro', 'weighted'.")

    return precision_val, recall_val, f1_val


def roc_curve(y_true, y_score):
    """
    Compute Receiver Operating Characteristic (ROC) curve.

    Parameters:
    y_true : array, shape = [n_samples]
        True binary labels in range {0, 1} or {-1, 1}.
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned
        by decision_function on some classifiers).

    Returns:
    fpr : array, shape = [n]
        Increasing false positive rates such that element i is the false positive rate
        of predictions with score >= thresholds[i].
    tpr : array, shape = [n]
        Increasing true positive rates such that element i is the true positive rate
        of predictions with score >= thresholds[i].
    thresholds : array, shape = [n]
        Decreasing thresholds on the decision function used to compute fpr and tpr.
    """
    
    y_true = cp.asarray(y_true)
    y_score = cp.asarray(y_score)

    if len(cp.unique(y_true)) != 2:
        raise ValueError("Only binary classification is supported.")

    
    desc_score_indices = cp.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]


    fpr = []
    tpr = []
    thresholds = []
    n_pos = cp.sum(y_true)
    n_neg = len(y_true) - n_pos

    tp = 0
    fp = 0
    prev_score = None

    
    for i, score in enumerate(y_score):
        if score != prev_score:
            fpr.append(fp / n_neg)
            tpr.append(tp / n_pos)
            thresholds.append(score)
            prev_score = score

        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1

    fpr.append(fp / n_neg)
    tpr.append(tp / n_pos)
    thresholds.append(score)

    return cp.array(fpr), cp.array(tpr), cp.array(thresholds)


def confusion_matrix(y_true, y_pred, class_count):
    """
    Computes confusion matrix.

    Args:
        y_true (numpy.ndarray): True class labels (1D array).
        y_pred (numpy.ndarray): Predicted class labels (1D array).
        num_classes (int): Number of classes.

    Returns:
        numpy.ndarray: Confusion matrix of shape (num_classes, num_classes).
    """
    confusion = cp.zeros((class_count, class_count), dtype=int)

    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred[i]
        confusion[true_label, pred_label] += 1

    return confusion


def pca(X, n_components):
    """
    
    Parameters:
    X (numpy array): (n_samples, n_features)
    n_components (int):
    
    Returns:
    X_reduced (numpy array): (n_samples, n_components)
    """
    
    X_meaned = X - cp.mean(X, axis=0)
    
    covariance_matrix = cp.cov(X_meaned, rowvar=False)
    
    eigenvalues, eigenvectors = cp.linalg.eigh(covariance_matrix)
    
    sorted_index = cp.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    
    eigenvectors_subset = sorted_eigenvectors[:, :n_components]
    
    X_reduced = cp.dot(X_meaned, eigenvectors_subset)
    
    return X_reduced