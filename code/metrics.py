import numpy as np
import scipy
from numba import jit, int64, boolean

import utils
import sklearn.metrics as sm

from functools import partial

from collections import OrderedDict
class ComputeMetrics:

    metric_types = ["accuracy", "edit_score", "overlap_f1"]
    # metric_types = ["macro_accuracy", "acc_per_class"]
    # metric_types += ["classification_accuracy"]
    # metric_types += ["precision", "recall"]
    # metric_types += ["mAP1", "mAP5", "midpoint"]
    trials = []


    def __init__(self, metric_types=None, overlap=.1, bg_class=None, n_classes=None):
        if metric_types is not None:
            self.metric_types = metric_types

        self.scores = OrderedDict()
        self.attrs = {"overlap":overlap, "bg_class":bg_class, "n_classes":n_classes}
        self.trials = []

        for m in self.metric_types:
            self.scores[m] = OrderedDict()

    @property
    def n_classes(self):
        return self.attrs['n_classes']

    def set_classes(self, n_classes):
        self.attrs['n_classes'] = n_classes

    def add_predictions(self, trial, P, Y):
        if trial not in self.trials:
            self.trials += [trial]

        for m in self.metric_types:
            self.scores[m][trial] = globals()[m](P, Y, **self.attrs)

    def print_trials(self, metric_types=None):
        if metric_types is None:
            metric_types = self.metric_types

        for trial in self.trials:
            scores = [self.scores[m][trial] for m in metric_types]
            scores_txt = []
            for m,s in zip(metric_types, scores):
                if type(s) is np.float64:
                    scores_txt += ["{}:{:.04}".format(m, s)]
                else:
                    scores_txt += [("{}:[".format(m)+"{:.04},"*len(s)).format(*s)+"]"]
            # txt = "Trial {}: ".format(trial) + " ".join(["{}:{:.04}".format(metric_types[i], scores[i]) for i in range(len(metric_types))])
            txt = "Trial {}: ".format(trial) + ", ".join(scores_txt)
            print(txt)


    def print_scores(self, metric_types=None):
        if metric_types is None:
            metric_types = self.metric_types

        scores = [np.mean([self.scores[m][trial] for trial in self.trials]) for m in metric_types]
        txt = "All: " + " ".join(["{}:{:.04}".format(metric_types[i], scores[i]) for i in range(len(metric_types))])
        print(txt)



def accuracy(P, Y, **kwargs):
    def acc_(p,y):
        return np.mean(p==y)*100
    if type(P) == list:
        return np.mean([np.mean(P[i]==Y[i]) for i in range(len(P))])*100
    else:
        return acc_(P,Y)

def macro_accuracy(P, Y, n_classes, bg_class=None, return_all=False, **kwargs):
    def macro_(P, Y, n_classes=None, bg_class=None, return_all=False):
        conf_matrix = sm.confusion_matrix(Y, P, labels=np.arange(n_classes))
        conf_matrix = conf_matrix/(conf_matrix.sum(0)[:,None]+1e-5)
        conf_matrix = np.nan_to_num(conf_matrix)
        diag = conf_matrix.diagonal()*100.

        # Remove background score
        if bg_class is not None:
            diag = np.array([diag[i] for i in range(n_classes) if i!=bg_class])

        macro = diag.mean()
        if return_all:
            return macro, diag
        else:
            return macro

    if type(P) == list:
        out = [macro_(P[i], Y[i], n_classes=n_classes, bg_class=bg_class, return_all=return_all) for i in range(len(P))]
        if return_all:
            return (np.mean([o[0] for o in out]), np.mean([o[1] for o in out],0))
        else:
            return np.mean(out)
    else:
        return macro_(P,Y, n_classes=n_classes, bg_class=bg_class, return_all=return_all)


def acc_per_class(P, Y, bg_class=None,n_classes=None, **kwargs):
    return macro_accuracy(P, Y, bg_class=bg_class, return_all=True, n_classes=n_classes, **kwargs)[1]


# ---------------------------------------------------
def classification_accuracy(P, Y, bg_class=None, **kwargs):
    # Assumes known temporal segmentation
    # P can either be a set of predictions (1d) or scores (2d)
    def clf_(p, y, bg_class):
        sums = 0.
        n_segs = 0.

        S_true = utils.segment_labels(y)
        I_true = np.array(utils.segment_intervals(y))

        for i in range(len(S_true)):
            if S_true[i] == bg_class:
                continue

            # If p is 1d, compute the most likely label, otherwise take the max over the score
            if p.ndim==1:
                pred_label = scipy.stats.mode(p[I_true[i][0]:I_true[i][1]])[0][0]
            else:
                pred_label = p[I_true[i][0]:I_true[i][1]].mean(1).argmax()
            sums += pred_label==S_true[i]
            n_segs += 1

        return sums / n_segs * 100

    if type(P) == list:
        return np.mean([clf_(P[i], Y[i], bg_class) for i in range(len(P))])
    else:
        return clf_(P, Y, bg_class)


@jit("float64(int64[:], int64[:], boolean)")
def levenstein_(p,y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i,0] = i
    for i in range(n_col+1):
        D[0,i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1]==p[i-1]:
                D[i,j] = D[i-1,j-1] 
            else:
                D[i,j] = min(D[i-1,j]+1,
                             D[i,j-1]+1,
                             D[i-1,j-1]+1)
    
    if norm:
        score = (1 - D[-1,-1]/max(m_row, n_col) ) * 100
    else:
        score = D[-1,-1]

    return score

def edit_score(P, Y, norm=True, bg_class=None, **kwargs):
    if type(P) == list:
        tmp = [edit_score(P[i], Y[i], norm, bg_class) for i in range(len(P))]
        return np.mean(tmp)
    else:
        P_ = utils.segment_labels(P)
        Y_ = utils.segment_labels(Y)
        if bg_class is not None:
            P_ = [c for c in P_ if c!=bg_class]
            Y_ = [c for c in Y_ if c!=bg_class]
        return levenstein_(P_, Y_, norm)


def overlap_f1(P, Y, n_classes=0, bg_class=None, overlap=.1, **kwargs):
    def overlap_(p,y, n_classes, bg_class, overlap):

        true_intervals = np.array(utils.segment_intervals(y))
        true_labels = utils.segment_labels(y)
        pred_intervals = np.array(utils.segment_intervals(p))
        pred_labels = utils.segment_labels(p)

        # Remove background labels
        if bg_class is not None:
            true_intervals = true_intervals[true_labels!=bg_class]
            true_labels = true_labels[true_labels!=bg_class]
            pred_intervals = pred_intervals[pred_labels!=bg_class]
            pred_labels = pred_labels[pred_labels!=bg_class]

        n_true = true_labels.shape[0]
        n_pred = pred_labels.shape[0]

        # We keep track of the per-class TPs, and FPs.
        # In the end we just sum over them though.
        TP = np.zeros(n_classes, np.float)
        FP = np.zeros(n_classes, np.float)
        true_used = np.zeros(n_true, np.float)

        for j in range(n_pred):
            # Compute IoU against all others
            intersection = np.minimum(pred_intervals[j,1], true_intervals[:,1]) - np.maximum(pred_intervals[j,0], true_intervals[:,0])
            union = np.maximum(pred_intervals[j,1], true_intervals[:,1]) - np.minimum(pred_intervals[j,0], true_intervals[:,0])
            IoU = (intersection / union)*(pred_labels[j]==true_labels)

            # Get the best scoring segment
            idx = IoU.argmax()

            # If the IoU is high enough and the true segment isn't already used
            # Then it is a true positive. Otherwise is it a false positive.
            if IoU[idx] >= overlap and not true_used[idx]:
                TP[pred_labels[j]] += 1
                true_used[idx] = 1
            else:
                FP[pred_labels[j]] += 1


        TP = TP.sum()
        FP = FP.sum()
        # False negatives are any unused true segment (i.e. "miss")
        FN = n_true - true_used.sum()
        
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        F1 = 2 * (precision*recall) / (precision+recall)

        # If the prec+recall=0, it is a NaN. Set these to 0.
        F1 = np.nan_to_num(F1)

        return F1*100

    if type(P) == list:
        return np.mean([overlap_(P[i],Y[i], n_classes, bg_class, overlap) for i in range(len(P))])
    else:
        return overlap_(P, Y, n_classes, bg_class, overlap)


def overlap_score(P, Y, bg_class=None, **kwargs):
    # From ICRA paper:
    # Learning Convolutional Action Primitives for Fine-grained Action Recognition
    # Colin Lea, Rene Vidal, Greg Hager 
    # ICRA 2016

    def overlap_(p,y, bg_class):
        true_intervals = np.array(utils.segment_intervals(y))
        true_labels = utils.segment_labels(y)
        pred_intervals = np.array(utils.segment_intervals(p))
        pred_labels = utils.segment_labels(p)

        if bg_class is not None:
            true_intervals = np.array([t for t,l in zip(true_intervals, true_labels) if l!=bg_class])
            true_labels = np.array([l for l in true_labels if l!=bg_class])
            pred_intervals = np.array([t for t,l in zip(pred_intervals, pred_labels) if l!=bg_class])
            pred_labels = np.array([l for l in pred_labels if l!=bg_class])            

        n_true_segs = true_labels.shape[0]
        n_pred_segs = pred_labels.shape[0]
        seg_scores = np.zeros(n_true_segs, np.float)

        for i in range(n_true_segs):
            for j in range(n_pred_segs):
                if true_labels[i]==pred_labels[j]:
                    intersection = min(pred_intervals[j][1], true_intervals[i][1]) - max(pred_intervals[j][0], true_intervals[i][0])
                    union        = max(pred_intervals[j][1], true_intervals[i][1]) - min(pred_intervals[j][0], true_intervals[i][0])
                    score_ = float(intersection)/union
                    seg_scores[i] = max(seg_scores[i], score_)

        return seg_scores.mean()*100

    if type(P) == list:
        return np.mean([overlap_(P[i],Y[i], bg_class) for i in range(len(P))])
    else:
        return overlap_(P, Y, bg_class)


# ----------- mAP metrics -------------

import numpy as np

def argsort(seq):
    """ The numpy sort is inconistent with the matlab version for values that are the same """
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def prap(rec, prec, tmp, npos):
    return prec[tmp==1].sum() / npos

def interval_overlap(gt_inter,det_inter):
    ov = np.zeros([gt_inter.shape[0], det_inter.shape[0]], np.float)
    for i in range(gt_inter.shape[0]):
        union = np.maximum(gt_inter[i,1], det_inter[:,1]) - \
                np.minimum(gt_inter[i,0], det_inter[:,0])
        intersection = np.minimum(gt_inter[i,1], det_inter[:,1]) - \
                        np.maximum(gt_inter[i,0], det_inter[:,0])
        intersection = intersection.clip(0, np.inf)    
        ov[i] = intersection/union

    return ov

def midpoint_criterion(gt_inter,det_inter):
    # gt_inter : truth
    # det_inter : detection
    # '1' if detection midpoint is within truth, otherwise 0
    ov = np.zeros([gt_inter.shape[0], det_inter.shape[0]], np.float)
    for i in range(gt_inter.shape[0]):
        midpoints = det_inter.mean(1)
        ov[i] = (midpoints>=gt_inter[i][0])*(midpoints<gt_inter[i][1])

    return ov    

# Recreated from the THUMOS2014 code for computing mAP@k
def TH14eventdetpr(gt_files, det_files, 
                   gt_labels, det_labels, 
                   gt_intervals,det_intervals, label, det_conf, 
                    overlap, overlap_fcn=interval_overlap):
    # returns recall, precision, ap

    videonames = np.unique(np.hstack([gt_files, det_files]))
    n_videos = len(videonames)

    unsortConf=[]
    unsortFlag=[]
    npos = float((gt_labels==label).sum())

    # Ensure there are predicted labels in detection list
    if not any(det_labels==label):
        return 0., 0., 0.

    for i in range(n_videos):
        gt = np.nonzero((gt_files==videonames[i])*(gt_labels==label))[0]
        det = np.nonzero((det_files==videonames[i])*(det_labels==label))[0]

        # If there are detections of this class:
        if det.shape[0]>0:
            # Sort based on confidences. Most conf first
            ind_s = argsort(-det_conf[det])
            det = det[ind_s]
            conf = det_conf[det]
            ind_free = np.ones(len(det))

            # If there are true segments of this class:
            if gt.shape[0]>0:
                ov = overlap_fcn(gt_intervals[gt], det_intervals[det])
                for k in range(ov.shape[0]):
                    ind = np.nonzero(ind_free)[0]
                    if len(ind)>0:
                        ind_m = np.argmax(ov[k][ind])
                        val_m = ov[k][ind][ind_m]
                        if val_m > overlap:
                            ind_free[ind[ind_m]] = 0

            ind1 = np.nonzero(ind_free==0)[0]
            ind2 = np.nonzero(ind_free==1)[0]

            # Mark '1' if true positive, '2' for false positive
            flag = np.hstack([np.ones(len(ind1)), 2*np.ones(len(ind2))])
            ttIdx = argsort(np.hstack([ind1, ind2]))
            idx_all = np.hstack([ind1, ind2])[ttIdx]
            flagall = flag[ttIdx]

            unsortConf = np.hstack([unsortConf, conf[idx_all]])
            unsortFlag = np.hstack([unsortFlag, flagall])

    conf=np.vstack([np.hstack(unsortConf), np.hstack(unsortFlag)])

    idx_s = argsort(-conf[0])
    tp = (conf[1][idx_s]==1).cumsum()*1.
    fp = (conf[1][idx_s]==2).cumsum()*1.
    is_correct = conf[1][idx_s]==1;
    rec = tp/float(npos)
    prec = tp/(fp+tp)
    ap = prap(rec, prec, is_correct, npos)

    return rec, prec, ap


def midpoint_mAP(gt_files, det_files, gt_labels, det_labels, gt_intervals, det_intervals, 
            det_conf, bg_class=None):
    return IoU_mAP(gt_files, det_files, gt_labels, det_labels, gt_intervals, det_intervals, \
            det_conf, bg_class=bg_class, use_midpoint=True)

def IoU_mAP(gt_files, det_files, gt_labels, det_labels, gt_intervals, det_intervals, 
            det_conf, threshold=.5, bg_class=None, use_midpoint=False):
# %    Output:         pr_all: precision-recall curves
# %                    ap_all: AP for each class
# %                       map: MAP 

    # Get labels. Remove background class if necessary
    labels = np.unique(gt_labels).tolist()
    if bg_class is not None and bg_class in labels:
        labels.remove(bg_class)
    n_labels = len(labels)

    if use_midpoint:
        overlap_fcn = midpoint_criterion
        threshold = 0.5
    else:
        overlap_fcn = interval_overlap

    pr_labels = []
    pr_all, rec_all, ap_all = [],[],[]
    for i in range(n_labels):
        label = labels[i]
        if any(gt_labels==label):
            rec, prec, ap = TH14eventdetpr(gt_files, det_files, gt_labels, det_labels, 
                                            gt_intervals, det_intervals, label, det_conf, 
                                            threshold, overlap_fcn)
            pr_labels += [label]
            pr_all += [prec]
            rec_all += [rec]
            ap_all += [ap]

    mAP = np.mean(ap_all)

    return pr_all, ap_all, mAP    


