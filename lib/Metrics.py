
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, roc_curve, precision_recall_curve
from sklearn.metrics import matthews_corrcoef
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import sys


def evaluate_model(y_true_label,y_prediction_prob,cutoff=0.5,plot=False, out_dir = "./", prefix="test"):
    y_true_class = y_true_label
    if len(y_true_label.shape) >= 2:
        if y_true_label.shape[1] >= 2:
            y_true_class = np.argmax(y_true_label, axis=1)
    else:
        y_true_class = y_true_label

    roc = roc_auc_score(y_true_class, y_prediction_prob)
    fpr, tpr, thresholds = roc_curve(y_true_class, y_prediction_prob)

    # classification report: precision recall f1-score support
    y_class_final = np.where(y_prediction_prob > cutoff, 1, 0)
    class_report = classification_report(y_true_class, y_class_final)

    mcc = matthews_corrcoef(y_true_class, y_class_final)
    acc = accuracy_score(y_true_class, y_class_final)
    f1 = f1_score(y_true_class,y_class_final)
    best_cutoff = thresholds[np.argmax(tpr - fpr)]
    print("\nMCC: %f, Accuracy: %f, F1: %f, AUROC: %f, prob cutoff: %f\n" % (mcc, acc, f1, roc, best_cutoff))
    print(class_report)

    ##
    cm = confusion_matrix(y_true_class, y_class_final, labels=[1,0])
    print("Confusion matrix:\n")
    print(cm)


    rr = calc_metrics(y_true_class,y_prediction_prob,cutoff=cutoff)
    print("Cutoff=%f, FPR=%f, TPR:%f (FP=%d, TP=%d, FN=%d, TN=%d)\n" % (cutoff,rr['fpr'],rr['tpr'],rr['fp'],rr['tp'],rr['fn'],rr['tn']))
    rr = calc_metrics(y_true_class,y_prediction_prob,cutoff=best_cutoff)
    print("Cutoff=%f, FPR=%f, TPR:%f (FP=%d, TP=%d, FN=%d, TN=%d)\n" % (best_cutoff,rr['fpr'],rr['tpr'],rr['fp'],rr['tp'],rr['fn'],rr['tn']))


    ## plot ROC curve
    #if plot
    if plot is True:
        roc_fig = out_dir + "/" + prefix + "_roc.png"
        plot_roc(y_true_class, y_prediction_prob, roc_fig)
        recall_fig = out_dir + "/" + prefix + "_recall.png"
        plot_recall(y_true_class, y_prediction_prob, recall_fig)

def plot_roc(y_true_label,y_prediction_prob,fig=None):

    fpr, tpr, thresholds = roc_curve(y_true_label, y_prediction_prob)
    roc = roc_auc_score(y_true_label, y_prediction_prob)

    plt.figure(figsize=(5, 5))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    if fig is not None:
        plt.savefig(fig,format="png",dpi=140)
        plt.close()
    else:
        plt.show()



def plot_recall(y_true_label,y_prediction_prob,fig=None):
    precision, recall, thresholds = precision_recall_curve(y_true_label,y_prediction_prob)
    plt.figure(figsize=(5, 5))
    lw = 2
    plt.plot(recall, precision, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)')
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('Receiver operating characteristic example')
    #plt.legend(loc="lower right")
    if fig is not None:
        plt.savefig(fig,format="png",dpi=140)
        plt.close()
    else:
        plt.show()


def plot_train(file:str, fig=None):
    a = pd.read_table(file, header=0, sep=",")
    a['order'] = range(0, a.shape[0])
    plt.figure(figsize=(7, 5))
    plt.plot(a['order'], a['acc'], label="acc")
    plt.plot(a['order'], a['val_acc'], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    max_ind = a['val_acc'].idxmax()
    plt.plot(a['order'][max_ind], a['val_acc'][max_ind], "h")
    plt.legend(loc="lower right")

    if fig is not None:
        print("Save figure: %s" % (fig))
        plt.savefig(fig,format="png",dpi=140)
        plt.close()
    else:
        plt.show()


def calc_metrics(y_true,y_prob,cutoff=0.5):

    if len(y_prob.shape) > 1:
        y_prob = y_prob.reshape(y_prob.shape[0])

    y_pred = np.where(y_prob >= cutoff, 1, 0)

    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))

    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)

    res = dict()
    res['fpr'] = fpr
    res['tpr'] = tpr
    res['fp'] = fp
    res['tp'] = tp
    res['fn'] = fn
    res['tn'] = tn
    
    return res

def calc_FPR(y_true,y_prob,cutoff=0.5):
    res = calc_metrics(y_true,y_prob,cutoff)
    return res['fpr']

def calc_TPR(y_true,y_prob,cutoff=0.5):
    res = calc_metrics(y_true,y_prob,cutoff)
    return res['tpr']

def add_confidence_metrics(x,model_dir:str,metric="fpr"):
    test_file = model_dir + "/site_prediction.tsv"
    if os.path.isfile(test_file):
        a = pd.read_csv(test_file,sep="\t",low_memory=False)
        if len(x) >= 1:
            if metric == "fpr":
                res = [calc_FPR(a['y'],a['y_pred'],i) for i in x]
            elif metric == "tpr":
                res = [calc_TPR(a['y'], a['y_pred'], i) for i in x]
            else:
                return None
            return res
        else:
            print("Input is empty!")
            sys.exit(1)
    else:
        print("The file for computing FPR doesn't exist:%s" % (test_file))
        return None



