import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def accuracy(y, decision_fn, threshold):
    '''Returns accuracy in a [0, 1] range.'''
    return sum(y == (decision_fn >= threshold)) * 1.0 / len(y)
    
def evaluate(y, decision_fn, title, plot=True, calc_accuracy=False):
    '''Plots a ROC curve and returns the AUC score and accuracy.
    
    Args:
        y: True labels
        decision_fn: Decision function results, before thresholding.
        title: Title to put in a plot
        plot: Whether to draw a plot
        calc_accuracy: Whether to calculate accuracy
        
    Returns:
        auc: AUC score
        thresholds: thresholds returned by roc_curve
        accs: Accuracy given thresholds'''
    fpr, tpr, thresholds = roc_curve(y, decision_fn)
    auc = roc_auc_score(y, decision_fn)
    accs = None
    if plot:
        plt.plot(fpr, tpr, label=title + ' AUC={0:.4f}'.format(auc))
    if calc_accuracy:
        accs = [accuracy(y, decision_fn, t) for t in thresholds]
    return auc, thresholds, accs
    
def evaluate_svm(
        decisions_array, y_valid, title='', ns=[1, 2, 5, 10, 20, 50, 100]):
    def get_decisions(n):
        return np.mean(decisions_array[:,:n], axis=1)
    evaluate_results(get_decisions, y_valid, title, ns)

def evaluate_results(decision_fn, y_valid, title, params):
    plt.figure(figsize=(4,4))
    
    plt.title('ROC curve: ' + title)
    aucs = []
    for n in params:
        # n = how many SVMs to average over
        aucs.append(evaluate(y_valid, decision_fn(n), '{0}'.format(n))[0])

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.show()

    plt.figure()
    plt.plot(params, aucs)
    plt.title('AUC after averaging')
    plt.ylabel('AUC score')
    plt.show()

