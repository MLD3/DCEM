import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from utils import cross_auc, fast_auc

class MetricRecord(object):
    def __init__(self):
        pass

    def metric_as_record(self, name, val0, val1, overall):
        return {
            "metric": name,
            "Group 0 value": val0,
            "Group 1 value": val1,
            "diff": val0 - val1,
            "overall": overall,
        }

    def compute_metric_0(self, *args):
        pass

    def compute_metric_1(self, *args):
        pass

    def compute_metric_all(self, *args):
        pass

    def __call__(self, *args):
        return self.metric_as_record(
                self.__class__.__name__,
                self.compute_metric_0(*args),
                self.compute_metric_1(*args),
                self.compute_metric_all(*args)
                )

class SimpleMetric(MetricRecord):

    def compute_metric(self, pred, Y, group, group_id, probs=False):
        if probs:
            Yhat = pred[:, 1]
        else:
            Yhat = pred
        return self.metric_function(Y[group == group_id], Yhat[group == group_id])

    def compute_metric_0(self, pred, Y, group):
        return self.compute_metric(pred, Y, group, 0, probs=pred.ndim > 1)

    def compute_metric_1(self, pred, Y, group):
        return self.compute_metric(pred, Y, group, 1, probs=pred.ndim > 1)

    def compute_metric_all(self, pred, Y, group):
        return self.compute_metric(pred, Y, group, group, probs=pred.ndim > 1)


class AUC(SimpleMetric):
    def __init__(self, use_sklearn_auc=False):
        super().__init__()
        self.metric_function = roc_auc_score if use_sklearn_auc else fast_auc 

class xAUC(MetricRecord):

    def compute_metric_0(self, pred_probs, Y, group):
        if pred_probs.ndim > 1:
            pred_probs = pred_probs[:, 1]
        pos_0 = pred_probs[(group == 0) & (Y == 1)]
        neg_1 = pred_probs[(group == 1) & (Y == 0)]
        return cross_auc(pos_0, neg_1)

    def compute_metric_1(self, pred_probs, Y, group):
        if pred_probs.ndim > 1:
            pred_probs = pred_probs[:, 1]
        neg_0 = pred_probs[(group == 0) & (Y == 0)]
        pos_1 = pred_probs[(group == 1) & (Y == 1)]
        return cross_auc(pos_1, neg_0)

class ROCGap(MetricRecord):
    def __init__(self, h=1e-4):
        super().__init__()
        self.diff = None
        self.h = h

    def get_roc_curve_diff(self, preds, Y, group):
        if len(np.unique(Y[group == 0])) < 2 or len(np.unique(Y[group == 1])) < 2:
            print("Degenerate TPR/FPR, so ROC curve is undefined - setting diff to empty ndarray")
            self.diff = np.array([])
            return self.diff
        fpr0, tpr0, th0 = roc_curve(Y[group == 0], preds[group == 0])
        fpr1, tpr1, th1 = roc_curve(Y[group == 1], preds[group == 1])

        roc_0 = interp1d(fpr0, tpr0)
        roc_1 = interp1d(fpr1, tpr1)

        xx_ = np.arange(0, 1, self.h)
        y0_ = roc_0(xx_)
        y1_ = roc_1(xx_)
        self.diff = y0_ - y1_
        return self.diff

    def compute_metric_0(self, pred_probs, Y, group):
        if pred_probs.ndim > 1:
            pred_probs = pred_probs[:, 1]
        if self.diff is None:
            self.diff = self.get_roc_curve_diff(pred_probs, Y, group)
        if len(self.diff) == 0:
            print("Degenerate TPR/FPR - setting ROCGap contribution for Group 0 to -1.")
            return -1
        diff0 = np.maximum(self.diff, 0) # y0_ > y1_ — group 0 has higher TPR at same FPR
        return np.trapz(diff0, dx=self.h)
        
    def compute_metric_1(self, pred_probs, Y, group):
        if pred_probs.ndim > 1:
            pred_probs = pred_probs[:, 1]
        if self.diff is None:
            self.diff = self.get_roc_curve_diff(pred_probs, Y, group)
        if len(self.diff) == 0:
            print("Degenerate TPR/FPR - setting ROCGap contribution for Group 1 to -1.")
            return -1
        diff1 = np.minimum(self.diff, 0)
        return np.trapz(diff1, dx=self.h) # y1_ > y0_ - group 1 has higher TPR at same FPR
        

class Accuracy(SimpleMetric):
    def __init__(self):
        super().__init__()
        self.metric_function = accuracy_score
