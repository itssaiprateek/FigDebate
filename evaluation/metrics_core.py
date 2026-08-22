"""Small dependency-free classification metrics used as a safe fallback."""


def _values(items):
    return list(items)


def accuracy_score(y_true, y_pred):
    truth, predicted = _values(y_true), _values(y_pred)
    return sum(a == b for a, b in zip(truth, predicted)) / len(truth)


def confusion_matrix(y_true, y_pred, labels):
    labels = list(labels)
    positions = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, predicted in zip(_values(y_true), _values(y_pred)):
        if truth in positions and predicted in positions:
            matrix[positions[truth]][positions[predicted]] += 1
    try:
        import numpy as np
        return np.asarray(matrix, dtype=int)
    except ImportError:
        return _Matrix(matrix)


class _Matrix(list):
    def tolist(self):
        return list(self)


def precision_recall_fscore_support(
    y_true, y_pred, labels, zero_division=0
):
    truth, predicted = _values(y_true), _values(y_pred)
    precisions, recalls, scores, supports = [], [], [], []
    for label in labels:
        true_positive = sum(
            actual == label and guess == label
            for actual, guess in zip(truth, predicted)
        )
        false_positive = sum(
            actual != label and guess == label
            for actual, guess in zip(truth, predicted)
        )
        false_negative = sum(
            actual == label and guess != label
            for actual, guess in zip(truth, predicted)
        )
        support = sum(actual == label for actual in truth)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else float(zero_division)
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else float(zero_division)
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else float(zero_division)
        )
        precisions.append(precision)
        recalls.append(recall)
        scores.append(f1)
        supports.append(support)
    return precisions, recalls, scores, supports


def f1_score(y_true, y_pred, labels, average="macro", zero_division=0):
    if average != "macro":
        raise ValueError("The internal fallback supports macro F1 only.")
    _, _, scores, _ = precision_recall_fscore_support(
        y_true, y_pred, labels, zero_division
    )
    return sum(scores) / len(scores)


def balanced_accuracy_score(y_true, y_pred):
    truth, predicted = _values(y_true), _values(y_pred)
    labels = sorted(set(truth))
    recalls = []
    for label in labels:
        positions = [index for index, item in enumerate(truth) if item == label]
        recalls.append(
            sum(predicted[index] == label for index in positions) / len(positions)
        )
    return sum(recalls) / len(recalls)
