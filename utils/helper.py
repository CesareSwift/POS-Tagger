import json

from sklearn import metrics


def read_jsonl(data_path):
    with open(data_path) as fh:
        for line in fh:
            yield json.loads(line)


def get_classification_report(labels, predictions, top_k=None, criterion='support', reverse=True, **kwargs):
    """Get sorted classification report, optionally filtering the top-k classes.

    Args:
        labels (List): target labels.
        predictions (List): predicted labels.
        top_k (int, optional): number of classes to include in the report. Defaults to None (all classes).
        criterion (bool, optional): report key to be used as criterion to filter classes.
        (precision | recall | f1-score | support). Defaults to 'support'.
        reverse: (bool, optional): if True return items sorted in reverse order.

    Returns:
        [type]: [description]
    """
    report = metrics.classification_report(labels, predictions, **kwargs)
    report_dict = metrics.classification_report(labels, predictions, output_dict=True, **kwargs)
    # report_dict['accuracy'] = {'accuracy': report_dict['accuracy'], 'support': report_dict['macro avg']['support']}
    report_dict.pop('accuracy')

    if criterion:
        report_dict = {k: v for k, v in sorted(report_dict.items(), key=lambda item: item[1][criterion], reverse=reverse)}

    report_keys = [k for ii, (k, v) in enumerate(report_dict.items()) if top_k is None or ii < top_k]
    report_keys += ['accuracy', 'support']
    report_lines = report.split('\n')
    report = '\n'.join([line for line in report_lines if any([key in line for key in report_keys])])
    return report
