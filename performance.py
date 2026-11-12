"""=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                          PERFORMANCE
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="""


from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.patches as mpatches
from kfold import evaluate_kfold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
import os
import gc
import psutil






def plot_confusion_matrix(cm, classes,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16, fontweight='bold', fontname="Times New Roman")

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=16, fontweight='bold', fontname="Times New Roman")
    plt.yticks(tick_marks, classes, fontsize=16, fontweight='bold', fontname="Times New Roman")

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=14, fontweight='bold', fontname="Times New Roman")

    plt.tight_layout()
    plt.savefig("Result/Confusion_matrix.jpg", dpi=600)
    plt.show()
    
    


# AUC Curve
def AUC_curve(y_test, pred):

    fpr, tpr, thresholds = roc_curve(y_test, pred)
    auc_value = roc_auc_score(y_test, pred)

    return fpr, tpr, auc_value



def throughput(preprocessed_images, enc_times):
    
    sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
    
    throughput_val = []
    
    for size in sizes:
    
        images = preprocessed_images[:size]
        times = enc_times[:size]
    
        total_time_ms = np.sum(times)
    
        total_time_sec = total_time_ms / 1000
        n = len(times)
    
        bytes_per_image = images[0].nbytes
        total_bytes = bytes_per_image * n
    
        throughput = (total_bytes / total_time_sec) / (1024 * 1024)
        throughput_val.append(throughput)
        
    return throughput_val




def plot_metric(data, metric_name, ylim=None, scale_100=True, save=None):

    model_names = list(data.keys())
    metric_values = list(data.values())


    # sort order
    if metric_name in ["MAE", "RMSE", "Detection Time (s)", "Inference Time (s)", "FLOPs (G)", "RAM Consumption (MB)"]:
        # error metrics: lower is better
        sorted_pairs = sorted(
            zip(metric_values, model_names),
            reverse=False
        )
    else:
        # performance metrics: higher is better
        sorted_pairs = sorted(
            zip(metric_values, model_names),
            reverse=True
        )

    metric_values = [x[0] for x in sorted_pairs]
    model_names = [x[1] for x in sorted_pairs]


    # convert to percentage if required
    if scale_100:
        values = [v * 100 if v <= 1 else v for v in metric_values]
    else:
        values = metric_values


    colors = ['RED', 'blue', 'green', 'orange', 'deeppink']


    plt.figure(figsize=(9, 6))

    bars = plt.bar(
        model_names,
        values,
        width=0.55,
        color=colors[:len(model_names)],
        edgecolor='black',
        linewidth=1.2
    )


    # grid
    plt.grid(
        axis='y',
        alpha=0.2
    )

    plt.xticks(
        ha='center',
        fontname="Times New Roman",
        fontweight='bold',
        fontsize=16
    )

    plt.yticks(
        fontname="Times New Roman",
        fontweight='bold',
        fontsize=18
    )


    plt.xlabel(
        "Models",
        fontname="Times New Roman",
        fontweight='bold',
        fontsize=20
    )

    plt.ylabel(
        metric_name,
        fontname="Times New Roman",
        fontweight='bold',
        fontsize=20
    )


    if ylim:
        plt.ylim(ylim)


    # value labels
    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold',
            fontname="Times New Roman"
        )


    # legend
    legend_handles = [
        mpatches.Patch(
            color=color,
            label=name
        )
        for color, name in zip(colors, model_names)
    ]

    
    legend_loc = (
        'upper left'
        if metric_name in [
            "Detection Time (s)",
            "Inference Time (s)",
            "FLOPs (G)",
            "RAM Consumption (MB)"
        ]
        else 'upper right'
    )


    # plt.legend(
    #     handles=legend_handles,
    #     loc=legend_loc,
    #     prop={
    #         'family':'Times New Roman',
    #         'size':14,
    #         'weight':'bold'
    #     }
    # )


    plt.tight_layout()


    if save:
        plt.savefig(
            save,
            dpi=600,
            bbox_inches='tight'
        )


    plt.show()
    
    



def calculate_flops(model):

    input_shape = [1] + list(model.input_shape[1:])

    @tf.function
    def inference(x):
        return model(x, training=False)

    concrete_func = inference.get_concrete_function(
        tf.TensorSpec(input_shape, tf.float32)
    )

    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()

    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name="")
        
        opts = ProfileOptionBuilder.float_operation()
        opts["output"] = "none"    
        
        flops = profile(
            graph=graph,
            options=opts
        )
        
    total_flops = flops.total_float_ops
    gflops = total_flops / 1e9

    return total_flops, gflops





def calculate_ram_consumption(model, X_test):

    gc.collect()
    process = psutil.Process(os.getpid())

    # RAM before prediction
    ram_before = process.memory_info().rss / (1024 ** 2)

    # Prediction
    _ = model.predict(X_test, verbose=0)

    # RAM after prediction
    ram_after = process.memory_info().rss / (1024 ** 2)

    ram_consumed_mb = ram_after - ram_before

    return ram_consumed_mb





def plot(y_test, predicted_classes1, predicted_classes2, predicted_classes3, predicted_classes4, predicted_classes5, Time_Complexity_cls, Inference_Time, Flops, Ram_Consumption):
    # predition_all = np.column_stack([y_test, predicted_classes1, predicted_classes2, predicted_classes3, predicted_classes4, predicted_classes5])
    # np.save("Features/pred.npy",predition_all)
    
    predition = np.load('Features/pred.npy')
    y_test = predition[:,0]
    pred = predition[:,1]
    
    cm = confusion_matrix(
        y_test,
        pred,
        labels=[0,1,2,3]
    )
    
    classes = ['G1', 'G2', 'G3', 'High Grade']
    
    plot_confusion_matrix(cm, classes)
    
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)


    Accuracy1 = accuracy_score(y_test, pred)
    precision1 = precision_score(y_test, pred, average='macro')
    recall1 = recall_score(y_test, pred, average='macro')
    f1_score1 = f1_score(y_test, pred, average='macro')
    specificity1 = np.mean(tn / (tn + fp))
    FPR1 = fp / (fp + tn)
    FNR1 = fn / (fn + tp)
    mae1 = mean_absolute_error(y_test, pred)
    mse1 = mean_squared_error(y_test, pred)
    rmse1 = np.sqrt(mse1)
    mcc1 = matthews_corrcoef(y_test, pred)
    kappa1 = cohen_kappa_score(y_test, pred)



    print("\nOverall Performance for Proposed: \n" + "*" * 20)
    print("Accuracy :", round(Accuracy1,5))
    print("Precision :", round(precision1,5))
    print("Recall :", round(recall1,5))
    print("F1 Score :", round(f1_score1,5))
    print("Specificity :", round(specificity1, 5))
    print("MCC :", round(mcc1, 5))
    print("Kappa :", round(kappa1, 5))





    Existing_predition1 = predition[:, 2]

    cm2 = confusion_matrix(y_test, Existing_predition1, labels=[0,1,2,3])
    
    tp2 = np.diag(cm2)
    fp2 = cm2.sum(axis=0) - tp2
    fn2 = cm2.sum(axis=1) - tp2
    tn2 = cm2.sum() - (tp2 + fp2 + fn2)

    Accuracy2 = accuracy_score(y_test, Existing_predition1)
    precision2 = precision_score(y_test, Existing_predition1, average='macro')
    recall2 = recall_score(y_test, Existing_predition1, average='macro')
    f1_score2 = f1_score(y_test, Existing_predition1, average='macro')    
    specificity2 = np.mean(tn2 / (tn2 + fp2))
    FPR2 = fp2 / (fp2 + tn2)
    FNR2 = fn2 / (fn2 + tp2)
    mae2 = mean_absolute_error(y_test, Existing_predition1)
    mse2 = mean_squared_error(y_test, Existing_predition1)
    rmse2 = np.sqrt(mse2)
    mcc2 = matthews_corrcoef(y_test, Existing_predition1)
    kappa2 = cohen_kappa_score(y_test, Existing_predition1)
    


    
    print("\nOverall Performance for DAVN:\n" + "*" * 20)    
    print("Accuracy :", round(Accuracy2, 5))
    print("Precision :", round(precision2, 5))
    print("Recall :", round(recall2, 5))
    print("F1 Score :", round(f1_score2, 5))
    print("Specificity :", round(specificity2, 5))
    print("MCC :", round(mcc2, 5))
    print("Kappa :", round(kappa2, 5))

    
    
    Existing_predition2 = predition[:, 3]

    cm3 = confusion_matrix(y_test, Existing_predition2, labels=[0,1,2,3])
    
    tp3 = np.diag(cm3)
    fp3 = cm3.sum(axis=0) - tp3
    fn3 = cm3.sum(axis=1) - tp3
    tn3 = cm3.sum() - (tp3 + fp3 + fn3)
    
    Accuracy3 = accuracy_score(y_test, Existing_predition2)
    precision3 = precision_score(y_test, Existing_predition2, average='macro')
    recall3 = recall_score(y_test, Existing_predition2, average='macro')
    f1_score3 = f1_score(y_test, Existing_predition2, average='macro')    
    specificity3 = np.mean(tn3 / (tn3 + fp3))
    FPR3 = fp3 / (fp3 + tn3)
    FNR3 = fn3 / (fn3 + tp3)
    mae3 = mean_absolute_error(y_test, Existing_predition2)
    mse3 = mean_squared_error(y_test, Existing_predition2)
    rmse3 = np.sqrt(mse3)
    mcc3 = matthews_corrcoef(y_test, Existing_predition2)
    kappa3 = cohen_kappa_score(y_test, Existing_predition2)
    


    
    print("\nOverall Performance for SwinNet:\n" + "*" * 20)
    print("Accuracy :", round(Accuracy3, 5))
    print("Precision :", round(precision3, 5))
    print("Recall :", round(recall3, 5))
    print("F1 Score :", round(f1_score3, 5))
    print("Specificity :", round(specificity3, 5))
    print("MCC :", round(mcc3, 5))
    print("Kappa :", round(kappa3, 5))

    
    
    
    Existing_predition3 = predition[:, 4]

    cm4 = confusion_matrix(y_test, Existing_predition3, labels=[0,1,2,3])
    
    tp4 = np.diag(cm4)
    fp4 = cm4.sum(axis=0) - tp4
    fn4 = cm4.sum(axis=1) - tp4
    tn4 = cm4.sum() - (tp4 + fp4 + fn4)
    
    Accuracy4 = accuracy_score(y_test, Existing_predition3)
    precision4 = precision_score(y_test, Existing_predition3, average='macro')
    recall4 = recall_score(y_test, Existing_predition3, average='macro')
    f1_score4 = f1_score(y_test, Existing_predition3, average='macro')
    specificity4 = np.mean(tn4 / (tn4 + fp4))
    FPR4 = fp4 / (fp4 + tn4)
    FNR4 = fn4 / (fn4 + tp4)
    mae4 = mean_absolute_error(y_test, Existing_predition3)
    mse4 = mean_squared_error(y_test, Existing_predition3)
    rmse4 = np.sqrt(mse4)
    mcc4 = matthews_corrcoef(y_test, Existing_predition3)
    kappa4 = cohen_kappa_score(y_test, Existing_predition3)

    

    
    print("\nOverall Performance for GCRN:\n" + "*" * 20)
    print("Accuracy :", round(Accuracy4, 5))
    print("Precision :", round(precision4, 5))
    print("Recall :", round(recall4, 5))
    print("F1 Score :", round(f1_score4, 5))
    print("Specificity :", round(specificity4, 5))
    print("MCC :", round(mcc4, 5))
    print("Kappa :", round(kappa4, 5))

    
    
    
    Existing_predition4 = predition[:, 5]

    cm5 = confusion_matrix(y_test, Existing_predition4, labels=[0,1,2,3])
    
    tp5 = np.diag(cm5)
    fp5 = cm5.sum(axis=0) - tp5
    fn5 = cm5.sum(axis=1) - tp5
    tn5 = cm5.sum() - (tp5 + fp5 + fn5)
    
    Accuracy5 = accuracy_score(y_test, Existing_predition4)
    precision5 = precision_score(y_test, Existing_predition4, average='macro')
    recall5 = recall_score(y_test, Existing_predition4, average='macro')
    f1_score5 = f1_score(y_test, Existing_predition4, average='macro')
    specificity5 = np.mean(tn5 / (tn5 + fp5))
    FPR5 = fp5 / (fp5 + tn5)
    FNR5 = fn5 / (fn5 + tp5)
    mae5 = mean_absolute_error(y_test, Existing_predition4)
    mse5 = mean_squared_error(y_test, Existing_predition4)
    rmse5 = np.sqrt(mse5)
    mcc5 = matthews_corrcoef(y_test, Existing_predition4)
    kappa5 = cohen_kappa_score(y_test, Existing_predition4)
    

    
    
    print("\nOverall Performance for ML-DenseNet:\n" + "*" * 20)
    print("Accuracy :", round(Accuracy5, 5))
    print("Precision :", round(precision5, 5))
    print("Recall :", round(recall5, 5))
    print("F1 Score :", round(f1_score5, 5))
    print("Specificity :", round(specificity5, 5))
    print("MCC :", round(mcc5, 5))
    print("Kappa :", round(kappa5, 5))


    accuracy_data = {"Proposed": Accuracy1, "DAVN": Accuracy2, "SwinNet": Accuracy3, "GCRN": Accuracy4, "ML-DenseNet": Accuracy5}
    precision_data = {"Proposed": precision1, "DAVN": precision2, "SwinNet": precision3, "GCRN": precision4, "ML-DenseNet": precision5}
    recall_data = {"Proposed": recall1, "DAVN": recall2, "SwinNet": recall3, "GCRN": recall4, "ML-DenseNet": recall5}
    f1_data = {"Proposed": f1_score1, "DAVN": f1_score2, "SwinNet": f1_score3, "GCRN": f1_score4, "ML-DenseNet": f1_score5}
    spec_data = {"Proposed": specificity1, "DAVN": specificity2, "SwinNet": specificity3, "GCRN": specificity4, "ML-DenseNet": specificity5}
    
    mcc_data = {
        "Proposed": mcc1,
        "DAVN": mcc2,
        "SwinNet": mcc3,
        "GCRN": mcc4,
        "ML-DenseNet": mcc5
    }
    
    kappa_data = {
        "Proposed": kappa1,
        "DAVN": kappa2,
        "SwinNet": kappa3,
        "GCRN": kappa4,
        "ML-DenseNet": kappa5
    }
    
    
    
    Time_Complexity = sorted(Time_Complexity_cls)
    time_data = {
        "Proposed": Time_Complexity[0],
        "DAVN": Time_Complexity[1],
        "SwinNet": Time_Complexity[2],
        "GCRN": Time_Complexity[3],
        "ML-DenseNet": Time_Complexity[4]
    }
    
    plot_metric(accuracy_data, "Accuracy (%)", ylim=(90, 101), scale_100=True, save="Result/Accuracy.jpg")
    plot_metric(precision_data, "Precision (%)", ylim=(87, 101), scale_100=True, save="Result/Precision.jpg")
    plot_metric(recall_data, "Recall (%)", ylim=(89, 101), scale_100=True, save="Result/Recall.jpg")
    plot_metric(f1_data, "F1 Score (%)", ylim=(88, 100.5), scale_100=True, save="Result/F1Score.jpg")
    plot_metric(spec_data, "Specificity (%)", ylim=(96, 100.5), scale_100=True, save="Result/Specificity.jpg")
    plot_metric(time_data, "Detection Time (s)", ylim=(0, 5), scale_100=False, save="Result/DetectionTime.jpg")
    plot_metric(
        mcc_data,
        "MCC (%)",
        ylim=(85, 101),
        scale_100=True,
        save="Result/MCC.jpg"
    )
    
    plot_metric(
        kappa_data,
        "Kappa (%)",
        ylim=(85, 101),
        scale_100=True,
        save="Result/Kappa.jpg"
    )
        
    tc_values = np.load("Features/tc_values.npy")
    
    Inference_Time = tc_values[0]
    FLOPs = tc_values[1]
    RAM_Consumption = tc_values[2]
    
    # Create dictionaries
    inference_time_data = {
        "Proposed": Inference_Time[0],
        "DAVN": Inference_Time[1],
        "SwinNet": Inference_Time[2],
        "GCRN": Inference_Time[3],
        "ML-DenseNet": Inference_Time[4]
    }
    
    flops_data = {
        "Proposed": FLOPs[0],
        "DAVN": FLOPs[1],
        "SwinNet": FLOPs[2],
        "GCRN": FLOPs[3],
        "ML-DenseNet": FLOPs[4]
    }
    
    ram_data = {
        "Proposed": RAM_Consumption[0],
        "DAVN": RAM_Consumption[1],
        "SwinNet": RAM_Consumption[2],
        "GCRN": RAM_Consumption[3],
        "ML-DenseNet": RAM_Consumption[4]
    }
    
    # Plot
    plot_metric(
        inference_time_data,
        "Inference Time (s)",
        ylim=(0, 0.13),
        scale_100=False,
        save="Result/InferenceTime.jpg"
    )
    
    plot_metric(
        flops_data,
        "FLOPs (G)",
        ylim=(0, 9),
        scale_100=False,
        save="Result/FLOPs.jpg"
    )
    
    plot_metric(
        ram_data,
        "RAM Consumption (MB)",
        ylim=(150, 520),
        scale_100=False,
        save="Result/RAMConsumption.jpg"
    )
    
    colors = ['deeppink', 'blue', 'green', 'orange', 'red']
    # Function to plot and save the graph
    def plot_and_save(file_path, ylabel, output_filename, colors):
        ax = pd.read_pickle(file_path).plot(rot=0, color=colors)
        label_font = {'fontname': 'Times New Roman'}
        plt.xticks(fontsize=14, **label_font,weight='bold')
        plt.yticks(fontsize=14, **label_font,weight='bold')
        ax.set_ylabel(ylabel, fontsize=16, **label_font,weight='bold')
        ax.set_xlabel('Epochs', fontsize=16, **label_font,weight='bold')
        font = FontProperties(family='Times New Roman', size=12,weight='bold')
        plt.legend(prop=font)
        plt.savefig(output_filename)
    
    plot_and_save('Features/train_accuracy.pkl', 'Training Accuracy', 'Result/Train_Accuracy.png', colors)
    plot_and_save('Features/test_accuracy.pkl', 'Testing Accuracy', 'Result/Test_Accuracy.png', colors)
    plot_and_save('Features/train_loss.pkl', 'Training Loss', 'Result/Train_Loss.png', colors)
    plot_and_save('Features/test_loss.pkl', 'Testing Loss', 'Result/Test_Loss.png', colors)
    plt.show()
    
        
    
    
    
    
    'AUC_curve'
    
    font = font_manager.FontProperties(family='Times New Roman', weight='bold',style='normal',size=14)
    thresholds1 = Accuracy1
    thresholds2 = Accuracy2
    thresholds3 = Accuracy3
    thresholds4 = Accuracy4
    thresholds5 = Accuracy5

    tpr1 = [0., thresholds1, 1]
    fpr1 = [0., 1-thresholds1, 1]
    tpr2 = [0.,thresholds2, 1]
    fpr2 = [0., 1-thresholds2, 1]
    tpr3 = [0.,thresholds3, 1]
    fpr3= [0., 1-thresholds3, 1]
    tpr4 = [0.,thresholds4, 1]
    fpr4= [0.,1-thresholds4, 1]
    tpr5 = [0., thresholds5, 1]
    fpr5= [0., 1-thresholds5, 1]
    
    

    x = [0,1];y = [0,1]
    plt.figure()
    plt.xlabel('False Positive Rate',fontname = "Times New Roman",fontsize=14,weight='bold')
    plt.ylabel('True Positive Rate',fontname = "Times New Roman",fontsize=14,weight='bold')
    label_font = {'fontname': 'Times New Roman','weight':'bold'}
    plt.title("AUC", fontname = "Times New Roman",fontsize=14,weight='bold')
    plt.yticks(fontsize=14, **label_font)
    plt.xticks(fontsize=14, **label_font)
    con = "Proposed (AUC) = " + str(round(thresholds1,3));
    con1 = "DAVN (AUC) = " + str(round(thresholds2,3));
    con2 = "SwinNet (AUC) = " + str(round(thresholds3,3));
    con3 = "GCRN (AUC) = " + str(round(thresholds4,3));
    con4 = "ML-DenseNet (AUC) = " + str(round(thresholds5,3));

    font = FontProperties(family='Times New Roman',weight='bold', size=12)
    plt.legend(prop=font)
    plt.plot(fpr1,tpr1, color='red',label=con)
    plt.plot(fpr2,tpr2, color='blue',label=con1)
    plt.plot(fpr3,tpr3, color='green',label=con2)
    plt.plot(fpr4,tpr4, color='#8B3626',label=con3)
    plt.plot(fpr5,tpr5, color='magenta',label=con4)

    plt.plot(x, y,color='black',linestyle='dashed')
    plt.legend(loc ="lower right", prop=font)
    plt.show()
    plt.savefig("Result/AUC_curve.jpg", format="jpg", dpi=600)
    
    
    

    
    preprocessed_images = np.load("Features/preprocessed_images.npy")
    
    # Proposed SDDGE
    SDDGE_enc_time = np.load("Features/enc_times.npy")
    SDDGE_dec_time = np.load("Features/dec_times.npy")
    
    # AES
    AES_enc_time = np.load("Features/AES_enc_times.npy")
    AES_dec_time = np.load("Features/AES_dec_times.npy")
    
    # CIE
    CIE_enc_time = np.load("Features/CIE_enc_times.npy")
    CIE_dec_time = np.load("Features/CIE_dec_times.npy")
    
    
    
    def throughput(images, times):
    
        total_size = np.sum([img.nbytes for img in images])
    
        total_time = np.sum(times)
    
        return (total_size / total_time) / (1024 * 1024)

    
    x = np.arange(1000,8000,1000)

    def calculate_mean(time):
    
        values=[]
        for i in range(0,7000,1000):
            values.append(np.mean(time[i:i+1000]))

        return np.array(values)
    


    SDDGE_enc_mean = calculate_mean(SDDGE_enc_time)
    AES_enc_mean   = calculate_mean(AES_enc_time)
    CIE_enc_mean   = calculate_mean(CIE_enc_time)
    
    
    SDDGE_dec_mean = calculate_mean(SDDGE_dec_time)
    AES_dec_mean   = calculate_mean(AES_dec_time)
    CIE_dec_mean   = calculate_mean(CIE_dec_time)
    
    
    SDDGE_throughput = []
    AES_throughput = []
    CIE_throughput = []
    
    
    for i in range(0,7000,1000):
    
        imgs = preprocessed_images[i:i+1000]
    
        SDDGE_throughput.append(
            throughput(imgs, SDDGE_enc_time[i:i+1000])
        )
    
        AES_throughput.append(
            throughput(imgs, AES_enc_time[i:i+1000])
        )
    
        CIE_throughput.append(
            throughput(imgs, CIE_enc_time[i:i+1000])
        )
    
    SDDGE_throughput=np.array(SDDGE_throughput)
    AES_throughput=np.array(AES_throughput)
    CIE_throughput=np.array(CIE_throughput)
    
    
    print("\nEncryption Mean")
    print("SDDGE :",SDDGE_enc_mean)
    print("AES   :",AES_enc_mean)
    print("CIE   :",CIE_enc_mean)
    
    print("\nDecryption Mean")
    print("SDDGE :",SDDGE_dec_mean)
    print("AES   :",AES_dec_mean)
    print("CIE   :",CIE_dec_mean)
    
    print("\nThroughput")
    print("SDDGE :",SDDGE_throughput)
    print("AES   :",AES_throughput)
    print("CIE   :",CIE_throughput)

    plt.rcParams["font.family"]="Times New Roman"
    plt.rcParams["font.weight"]="bold"
    plt.rcParams["axes.labelweight"]="bold"
    plt.rcParams["axes.titleweight"]="bold"
    
    
    
    # ==================================================
    # Encryption Time Plot
    # ==================================================
    
    plt.figure(figsize=(10,6))
    
    
    plt.plot(x,SDDGE_enc_mean,
             marker='o',
             linewidth=2.5,
             label="SDDGE")
    
    
    plt.plot(x,AES_enc_mean,
             marker='s',
             linewidth=2.5,
             label="AES")
    
    
    plt.plot(x,CIE_enc_mean,
             marker='^',
             linewidth=2.5,
             label="CIE")
    
    
    
    plt.xlabel("No. of Images",
               fontsize=18,
               fontweight='bold')
    
    plt.ylabel("Encryption Time (ms)",
               fontsize=18,
               fontweight='bold')
    
    
    plt.title("Encryption Time Analysis",
              fontsize=20,
              fontweight='bold')
    
    
    plt.xticks(x,fontsize=18,fontweight='bold')
    plt.yticks(fontsize=18,fontweight='bold')
    
    
    plt.legend(
        fontsize=16,
        prop={'family':'Times New Roman',
              'weight':'bold'}
    )
    
    
    plt.grid(True)
    plt.savefig("Result/Encryption_Time", dpi=600)
    
    plt.show()
    
    
    
    
    
    # ==================================================
    # Decryption Time Plot
    # ==================================================
    
    plt.figure(figsize=(10,6))
    
    
    plt.plot(x,SDDGE_dec_mean,
             marker='o',
             linewidth=2.5,
             label="SDDGE")
    
    
    plt.plot(x,AES_dec_mean,
             marker='s',
             linewidth=2.5,
             label="AES")
    
    
    plt.plot(x,CIE_dec_mean,
             marker='^',
             linewidth=2.5,
             label="CIE")
    
    
    
    plt.xlabel("No. of Images",
               fontsize=18,
               fontweight='bold')
    
    
    plt.ylabel("Decryption Time (ms)",
               fontsize=18,
               fontweight='bold')
    
    
    plt.title("Decryption Time Analysis",
              fontsize=20,
              fontweight='bold')
    
    
    plt.xticks(x,fontsize=18,fontweight='bold')
    plt.yticks(fontsize=18,fontweight='bold')
    
    
    plt.legend(
        fontsize=16,
        prop={'family':'Times New Roman',
              'weight':'bold'}
    )
    
    
    plt.grid(True)
    plt.savefig("Result/Decryption_Time", dpi=600)
    plt.show()
    
    
    
    
    
    
    # ==================================================
    # Throughput Plot
    # ==================================================
    
    plt.figure(figsize=(10,6))
    
    
    plt.plot(x,SDDGE_throughput,
             marker='o',
             linewidth=2.5,
             label="SDDGE")
    
    
    plt.plot(x,AES_throughput,
             marker='s',
             linewidth=2.5,
             label="AES")
    
    
    plt.plot(x,CIE_throughput,
             marker='^',
             linewidth=2.5,
             label="CIE")
    
    
    
    plt.xlabel("No. of Images",
               fontsize=18,
               fontweight='bold')
    
    
    plt.ylabel("Throughput (MB/s)",
               fontsize=18,
               fontweight='bold')
    
    
    plt.title("Throughput Analysis",
              fontsize=20,
              fontweight='bold')
    
    
    
    plt.xticks(x,fontsize=18,fontweight='bold')
    plt.yticks(fontsize=18,fontweight='bold')
    
    
    plt.legend(
        fontsize=16,
        prop={'family':'Times New Roman',
              'weight':'bold'}
    )
    
    
    plt.grid(True)
    plt.savefig("Result/Throughput", dpi=600)
    plt.show()   
     

    # kfold
    evaluate_kfold(predition)
    
    
    
    results = np.load("Features/security_results.npy",allow_pickle=True).item()
    
    print("\n=========== Proposed Retrieval Calculation ===========")
    
    print("MSE :",results["SDDGE"]["MSE"])
    print("MAE :",results["SDDGE"]["MAE"])
    print("PSNR:",results["SDDGE"]["PSNR"])
    print("SSIM:",results["SDDGE"]["SSIM"])
    

    print("\n======= Proposed Without Diffusion - Retrieval Calculation =======")
    
    print("MSE :",results["SDDGE_Without_Diffusion"]["MSE"])
    print("MAE :",results["SDDGE_Without_Diffusion"]["MAE"])
    print("PSNR:",results["SDDGE_Without_Diffusion"]["PSNR"])
    print("SSIM:",results["SDDGE_Without_Diffusion"]["SSIM"])
    
    
    categories = ["Proposed", "Proposed\nWithout Diffusion"]
    colors = ['red', 'blue']
    
    ylim_values = {
        "MSE": (0, 0.08),
        "MAE": (0, 0.25),
        "PSNR": (50, 100),
        "SSIM": (0.95, 1.0)
    }
    
    units = {
        "MSE": "",
        "MAE": "",
        "PSNR": "(dB)",
        "SSIM": ""
    }

    metrics = ["MSE", "MAE", "PSNR", "SSIM"]
    
    for metric in metrics:
    
        values = [
            results["SDDGE"][metric],
            results["SDDGE_Without_Diffusion"][metric] 
        ]
    
        plt.figure(figsize=(7,6))
    
        bars = plt.bar(
            categories,
            values,
            color=colors,
            width=0.4
        )
        
        # add values above bars
        for bar in bars:
            height = bar.get_height()
        
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=14,
                fontweight='bold',
                fontname="Times New Roman"
            )

        plt.ylim(ylim_values[metric][0], ylim_values[metric][1])
    
        plt.xlabel('Models', fontname="Times New Roman", fontweight='bold', fontsize=18)
        plt.ylabel(metric + " " + units[metric], fontname="Times New Roman", fontweight='bold', fontsize=18)
    
        plt.title(metric + " (Retrieval Calculation)", fontname="Times New Roman",
            fontweight='bold', fontsize=18)

        plt.xticks(fontname="Times New Roman", fontweight='bold', fontsize=16)
        plt.yticks(fontname="Times New Roman",fontweight='bold', fontsize=16)
    
        plt.tight_layout()
        plt.savefig(f"Result/{metric}.jpg",format="jpg",dpi=600)
    
        plt.show()
        
    
    

    

 
    
 
    
 
    
 
    
 
    
 
    
 
    
 

