import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from numpy import  mean
import os
from matplotlib.font_manager import FontProperties
from __pycache__.utils import process
from sklearn.metrics import jaccard_score
import random
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import cohen_kappa_score




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def calculate_miou(tp, fp, fn):
    miou = tp / (tp + fp + fn)
    return miou

def calculate_specificity(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    specificity = tn / (tn + fp)
    return specificity



def plot_multibar(categories, values, labels, colors, ylabel, xtick_labels, 
                  ylim=(90,101), filename=None):

    bar_width = 0.15
    gap_between_bars = 0.01

    n_models = len(values)
    n_categories = len(categories)

    x_positions = np.arange(n_categories) * (1 + gap_between_bars * n_models)
    offset = -(n_models // 2) * (bar_width + gap_between_bars)

    positions = [
        x_positions + offset + i*(bar_width + gap_between_bars)
        for i in range(n_models)
    ]

    plt.figure(figsize=(10,6))


    for pos, val, color, label in zip(positions, values, colors, labels):

        bars = plt.bar(
            pos,
            val,
            color=color,
            width=bar_width,
            edgecolor='grey',
            label=label
        )

        for bar in bars:
            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                fontname="Times New Roman"
            )


    middle = (
        x_positions +
        offset +
        (bar_width + gap_between_bars)*(n_models//2)
    )

    plt.xticks(
        middle,
        xtick_labels,
        fontsize=18,
        fontweight="bold",
        fontname="Times New Roman"
    )

    plt.yticks(
        fontsize=18,
        fontweight="bold",
        fontname="Times New Roman"
    )

    plt.ylim(ylim)

    plt.ylabel(
        ylabel,
        fontsize=18,
        fontweight="bold",
        fontname="Times New Roman"
    )


    plt.legend(
        loc='lower right',
        ncol=3,
        fancybox=True,
        shadow=True,
        prop={
            'family':'Times New Roman',
            'size':14,
            'weight':'bold'
        }
    )

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=600)

    plt.show()



     
def get_confusion_elements_multiclass(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    # Initialize lists to store tn, fp, fn, tp for each class
    tn_list = []
    fp_list = []
    fn_list = []
    tp_list = []

    for i, label in enumerate(labels):
        TP = cm[i, i]  # True Positive: diagonal element
        FP = cm[:, i].sum() - TP  # False Positive: column sum excluding diagonal
        FN = cm[i, :].sum() - TP  # False Negative: row sum excluding diagonal
        TN = cm.sum() - (TP + FP + FN)  # True Negative: the rest of the matrix
        tn_list.append(TN)
        fp_list.append(FP)
        fn_list.append(FN)
        tp_list.append(TP)

    return tn_list, fp_list, fn_list, tp_list



def k_fold(y_true, y_pred, k_values):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Initialize metric lists
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    specificity_scores = []
    mae_scores = []
    mse_scores = []
    tpr_list = []
    fpr_list = []
    kappa_list = []
    
    for k in k_values:
        if k < 2 or k > len(y_true):
            continue
        
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        for train_index, test_index in kf.split(y_true):
            y_true_test = y_true[test_index]
            y_pred_test = y_pred[test_index]
            
            # Compute confusion elements
            tn, fp, fn, tp = get_confusion_elements_multiclass(y_true_test, y_pred_test)
            tn = np.array(tn); fp = np.array(fp); fn = np.array(fn); tp = np.array(tp)
            
            # Standard metrics
            accuracy_scores.append(accuracy_score(y_true_test, y_pred_test))
            precision_scores.append(precision_score(y_true_test, y_pred_test, average='macro', zero_division=0))
            recall_scores.append(recall_score(y_true_test, y_pred_test, average='macro', zero_division=0))
            f1_scores.append(f1_score(y_true_test, y_pred_test, average='macro', zero_division=0))
            specificity_scores.append(np.mean(tn / (tn + fp)))
            
            # TPR/FPR
            tpr_list.append(np.mean(tp / (tp + fn)))
            fpr_list.append(np.mean(fp / (fp + tn)))
            
            # Error metrics
            mae_scores.append(mean_absolute_error(y_true_test, y_pred_test))
            mse_scores.append(mean_squared_error(y_true_test, y_pred_test))
            
            # Kappa
            kappa_list.append(cohen_kappa_score(y_true_test, y_pred_test))
    
    return (accuracy_scores, precision_scores, recall_scores, f1_scores, specificity_scores, 
            mae_scores, mse_scores, tpr_list, fpr_list, kappa_list)



def evaluate_kfold(predition):

    y_test=predition[:,0];propose_predicted1=predition[:,1];propose_predicted2=predition[:,2];propose_predicted3=predition[:,3]; propose_predicted4=predition[:,4];propose_predicted5=predition[:,5]
    k_values = [1,2,3,4,5]  
    
    # Model 1
    acc1, prec1, recall1, fl_score1, spec1, mae1, mse1, tpr1, fpr1, kappa1 = k_fold(y_test, propose_predicted1, k_values)
    acc1 = process(acc1); prec1 = process(prec1); recall1 = process(recall1); fl_score1 = process(fl_score1)
    spec1 = process(spec1, metric="spec"); tpr1 = process(tpr1); fpr1 = process(fpr1)
    mae1 = process(mae1, metric="mae"); mse1 = process(mse1, metric="mse")
    acc1.sort(); prec1.sort(); recall1.sort(); fl_score1.sort(); spec1.sort(); tpr1.sort(); fpr1.sort()
    acc1 = acc1[:5]; prec1 = prec1[:5]; recall1 = recall1[:5]; fl_score1 = fl_score1[:5]
    spec1 = spec1[:5]; tpr1 = tpr1[:5]; fpr1 = fpr1[:5]
    
    # Model 2
    acc2, prec2, recall2, fl_score2, spec2, mae2, mse2, tpr2, fpr2, kappa2 = k_fold(y_test, propose_predicted2, k_values)
    acc2 = process(acc2); prec2 = process(prec2); recall2 = process(recall2); fl_score2 = process(fl_score2)
    spec2 = process(spec2, metric="spec"); tpr2 = process(tpr2); fpr2 = process(fpr2)
    mae2 = process(mae2, metric="mae"); mse2 = process(mse2, metric="mse")
    acc2.sort(); prec2.sort(); recall2.sort(); fl_score2.sort(); spec2.sort(); tpr2.sort(); fpr2.sort()
    acc2 = acc2[:5]; prec2 = prec2[:5]; recall2 = recall2[:5]; fl_score2 = fl_score2[:5]
    spec2 = spec2[:5]; tpr2 = tpr2[:5]; fpr2 = fpr2[:5]
    
    # Model 3
    acc3, prec3, recall3, fl_score3, spec3, mae3, mse3, tpr3, fpr3, kappa3 = k_fold(y_test, propose_predicted3, k_values)
    acc3 = process(acc3); prec3 = process(prec3); recall3 = process(recall3); fl_score3 = process(fl_score3)
    spec3 = process(spec3, metric="spec"); tpr3 = process(tpr3); fpr3 = process(fpr3)
    mae3 = process(mae3, metric="mae"); mse3 = process(mse3, metric="mse")
    acc3.sort(); prec3.sort(); recall3.sort(); fl_score3.sort(); spec3.sort(); tpr3.sort(); fpr3.sort()
    acc3 = acc3[:5]; prec3 = prec3[:5]; recall3 = recall3[:5]; fl_score3 = fl_score3[:5]
    spec3 = spec3[:5]; tpr3 = tpr3[:5]; fpr3 = fpr3[:5]
    
    # Model 4
    acc4, prec4, recall4, fl_score4, spec4, mae4, mse4, tpr4, fpr4, kappa4 = k_fold(y_test, propose_predicted4, k_values)
    acc4 = process(acc4); prec4 = process(prec4); recall4 = process(recall4); fl_score4 = process(fl_score4)
    spec4 = process(spec4, metric="spec"); tpr4 = process(tpr4); fpr4 = process(fpr4)
    mae4 = process(mae4, metric="mae"); mse4 = process(mse4, metric="mse")
    acc4.sort(); prec4.sort(); recall4.sort(); fl_score4.sort(); spec4.sort(); tpr4.sort(); fpr4.sort()
    acc4 = acc4[:5]; prec4 = prec4[:5]; recall4 = recall4[:5]; fl_score4 = fl_score4[:5]
    spec4 = spec4[:5]; tpr4 = tpr4[:5]; fpr4 = fpr4[:5]
    
    # Model 5
    acc5, prec5, recall5, fl_score5, spec5, mae5, mse5, tpr5, fpr5, kappa5 = k_fold(y_test, propose_predicted5, k_values)
    acc5 = process(acc5); prec5 = process(prec5); recall5 = process(recall5); fl_score5 = process(fl_score5)
    spec5 = process(spec5, metric="spec"); tpr5 = process(tpr5); fpr5 = process(fpr5)
    mae5 = process(mae5, metric="mae"); mse5 = process(mse5, metric="mse")
    acc5.sort(); prec5.sort(); recall5.sort(); fl_score5.sort(); spec5.sort(); tpr5.sort(); fpr5.sort()
    acc5 = acc5[:5]; prec5 = prec5[:5]; recall5 = recall5[:5]; fl_score5 = fl_score5[:5]
    spec5 = spec5[:5]; tpr5 = tpr5[:5]; fpr5 = fpr5[:5]
    


    
    print("\nDAVN:")
    print("Accuracy:     ", [round(acc2[i] * 100, 2) for i in range(5)])
    print("Precision:    ", [round(prec2[i] * 100, 2) for i in range(5)])
    print("Recall:       ", [round(recall2[i] * 100, 2) for i in range(5)])
    print("F1_Score:     ", [round(fl_score2[i] * 100, 2) for i in range(5)])
    print("Specificity:  ", [round(spec2[i] * 100, 2) for i in range(5)])
    
    
    print("\nSwinNet:")
    print("Accuracy:     ", [round(acc3[i] * 100, 2) for i in range(5)])
    print("Precision:    ", [round(prec3[i] * 100, 2) for i in range(5)])
    print("Recall:       ", [round(recall3[i] * 100, 2) for i in range(5)])
    print("F1_Score:     ", [round(fl_score3[i] * 100, 2) for i in range(5)])
    print("Specificity:  ", [round(spec3[i] * 100, 2) for i in range(5)])
    
    
    print("\nGCRN:")
    print("Accuracy:     ", [round(acc4[i] * 100, 2) for i in range(5)])
    print("Precision:    ", [round(prec4[i] * 100, 2) for i in range(5)])
    print("Recall:       ", [round(recall4[i] * 100, 2) for i in range(5)])
    print("F1_Score:     ", [round(fl_score4[i] * 100, 2) for i in range(5)])
    print("Specificity:  ", [round(spec4[i] * 100, 2) for i in range(5)])
    
    
    print("\nML-DenseNet:")
    print("Accuracy:     ", [round(acc5[i] * 100, 2) for i in range(5)])
    print("Precision:    ", [round(prec5[i] * 100, 2) for i in range(5)])
    print("Recall:       ", [round(recall5[i] * 100, 2) for i in range(5)])
    print("F1_Score:     ", [round(fl_score5[i] * 100, 2) for i in range(5)])
    print("Specificity:  ", [round(spec5[i] * 100, 2) for i in range(5)])
    
    
    print("\nProposed:")
    print("Accuracy:     ", [round(acc1[i] * 100, 2) for i in range(5)])
    print("Precision:    ", [round(prec1[i] * 100, 2) for i in range(5)])
    print("Recall:       ", [round(recall1[i] * 100, 2) for i in range(5)])
    print("F1_Score:     ", [round(fl_score1[i] * 100, 2) for i in range(5)])
    print("Specificity:  ", [round(spec1[i] * 100, 2) for i in range(5)])
    
    
    
    categories = ['K=1','K=2','K=3','K=4','K=5']
    labels = ["Proposed", "DAVN", "SwinNet", "GCRN", "ML-DenseNet"]
    colors = ["red", "blue", "orange", "green", "purple"]
    
    values1 = [np.array(acc1)*100, np.array(acc2)*100, np.array(acc3)*100, np.array(acc4)*100, 
                np.array(acc5)*100]
    
    plot_multibar(categories=categories, values=values1, labels=labels, colors=colors,
        ylabel="Accuracy (%)", xtick_labels=['K-Fold 1','K-Fold 2','K-Fold 3','K-Fold 4','K-Fold 5'],
        ylim=(80,101), filename="Result/Accuracy_KFold.png")
    
        
        
    values2 = [np.array(prec1)*100, np.array(prec2)*100, np.array(prec3)*100, np.array(prec4)*100,   
              np.array(prec5)*100]
    
    plot_multibar(categories=categories, values=values2, labels=labels, colors=colors, ylabel="Precision (%)",
        xtick_labels=['K-Fold 1','K-Fold 2','K-Fold 3','K-Fold 4','K-Fold 5'],
        ylim=(80,101), filename="Result/Precision_KFold.png")
    
    
    
    values3 = [np.array(recall1)*100, np.array(recall2)*100, np.array(recall3)*100, np.array(recall4)*100,
              np.array(recall5)*100]
    
    plot_multibar(categories=categories, values=values3, labels=labels, colors=colors, ylabel="Recall (%)",
        xtick_labels=['K-Fold 1','K-Fold 2','K-Fold 3','K-Fold 4','K-Fold 5'],
        ylim=(80,101), filename="Result/Recall_KFold.png")
    
    
    
    values = [np.array(fl_score1)*100, np.array(fl_score2)*100, np.array(fl_score3)*100, np.array(fl_score4)*100,
        np.array(fl_score5)*100]
    
    plot_multibar(categories=categories, values=values, labels=labels, colors=colors, ylabel="F1 Score (%)",
        xtick_labels=['K-Fold 1','K-Fold 2','K-Fold 3','K-Fold 4','K-Fold 5'],
        ylim=(80,101), filename="Result/F1Score_KFold.png")
    
    
    
    values = [np.array(spec1)*100, np.array(spec2)*100, np.array(spec3)*100, np.array(spec4)*100,
        np.array(spec5)*100]
    
    plot_multibar(categories=categories, values=values, labels=labels, colors=colors, ylabel="Specificity (%)",
        xtick_labels=['K-Fold 1','K-Fold 2','K-Fold 3','K-Fold 4','K-Fold 5'],
        ylim=(90,101), filename="Result/Specificity_KFold.png")















