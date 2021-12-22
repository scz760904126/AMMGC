import numpy as np
import math
from sklearn.metrics import roc_auc_score, precision_recall_curve


def model_evaluate(yt, yp):
    precision, recall, _, = precision_recall_curve(yt, yp)
    aupr = -np.trapz(precision, recall)
    auc = roc_auc_score(yt, yp)
    # ---f1,acc,recall, specificity, precision
    real_score = np.mat(yt)
    predict_score = np.mat(yp)
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN
    tpr = TP / (TP + FN)
    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return auc, aupr, f1_score[0, 0], accuracy[0, 0], recall[0, 0], specificity[0, 0], precision[0, 0]

# def model_evaluate(real_score, predict_score):
#     AUPR = get_AUPR(real_score, predict_score)
#     AUC = get_AUC(real_score, predict_score)
#     [f1, accuracy, recall, spec, precision] = get_Metrics(real_score, predict_score)
#     return np.array([AUPR, AUC, f1, accuracy, recall, spec, precision])
#
#
# def get_AUPR(real_score, predict_score):
#     sorted_predict_score = sorted(list(set(np.array(predict_score).flatten())))
#     sorted_predict_score_num = len(sorted_predict_score)
#     thresholdlist = []
#     for i in range(999):
#         threshold = sorted_predict_score[int(math.ceil(sorted_predict_score_num * (i + 1) / 1000) - 1)]
#         thresholdlist.append(threshold)
#     thresholds = np.matrix(thresholdlist)
#     TN = np.zeros((1, len(thresholdlist)))
#     TP = np.zeros((1, len(thresholdlist)))
#     FN = np.zeros((1, len(thresholdlist)))
#     FP = np.zeros((1, len(thresholdlist)))
#     for i in range(thresholds.shape[1]):
#         p_index = np.where(predict_score >= thresholds[0, i])
#         TP[0, i] = len(np.where(real_score[p_index] == 1)[0])
#         FP[0, i] = len(np.where(real_score[p_index] == 0)[0])
#         # print(TP[0, i], FP[0, i])
#         n_index = np.where(predict_score < thresholds[0, i])
#         FN[0, i] = len(np.where(real_score[n_index] == 1)[0])
#         TN[0, i] = len(np.where(real_score[n_index] == 0)[0])
#     precision = TP / (TP + FP)
#     recall = TP / (TP + FN)
#     x = list(np.array(recall).flatten())
#     y = list(np.array(precision).flatten())
#     xy = [(x, y) for x, y in zip(x, y)]
#     xy.sort()
#     x = [x for x, y in xy]
#     y = [y for x, y in xy]
#     new_x = [x for x, y in xy]
#     new_y = [y for x, y in xy]
#     new_x[0] = 0
#     new_y[0] = 1
#     new_x.append(1)
#     new_y.append(0)
#     name1 = 'plot_curve/non_attention_AUPR_X.csv'
#     np.savetxt(name1, new_x, delimiter=',')
#     name2 = 'plot_curve/non_attention_AUPR_Y.csv'
#     np.savetxt(name2, new_y, delimiter=',')
#
#     area = 0
#     for i in range(thresholds.shape[1]):
#         area = area + (new_y[i] + new_y[i + 1]) * (new_x[i + 1] - new_x[i]) / 2
#     return area
#
#
# def get_AUC(real_score, predict_score):
#     sorted_predict_score = sorted(list(set(np.array(predict_score).flatten())))
#     sorted_predict_score_num = len(sorted_predict_score)
#     thresholdlist = []
#     for i in range(999):
#         threshold = sorted_predict_score[int(math.ceil(sorted_predict_score_num * (i + 1) / 1000) - 1)]
#         thresholdlist.append(threshold)
#     thresholds = np.matrix(thresholdlist)
#     TN = np.zeros((1, len(thresholdlist)))
#     TP = np.zeros((1, len(thresholdlist)))
#     FN = np.zeros((1, len(thresholdlist)))
#     FP = np.zeros((1, len(thresholdlist)))
#     for i in range(thresholds.shape[1]):
#         p_index = np.where(predict_score >= thresholds[0, i])
#         TP[0, i] = len(np.where(real_score[p_index] == 1)[0])
#         FP[0, i] = len(np.where(real_score[p_index] == 0)[0])
#         n_index = np.where(predict_score < thresholds[0, i])
#         FN[0, i] = len(np.where(real_score[n_index] == 1)[0])
#         TN[0, i] = len(np.where(real_score[n_index] == 0)[0])
#     sen = TP / (TP + FN)
#     spe = TN / (TN + FP)
#     x = list(np.array(1 - spe).flatten())
#     y = list(np.array(sen).flatten())
#     xy = [(x, y) for x, y in zip(x, y)]
#     xy.sort()
#     new_x = [x for x, y in xy]
#     new_y = [y for x, y in xy]
#     new_x[0] = 0
#     new_y[0] = 0
#     new_x.append(1)
#     new_y.append(1)
#     name1 = 'plot_curve/non_attention_AUC_X.csv'
#     np.savetxt(name1, new_x, delimiter=',')
#     name2 = 'plot_curve/non_attention_AUC_Y.csv'
#     np.savetxt(name2, new_y, delimiter=',')
#     area = 0
#     for i in range(thresholds.shape[1]):
#         area = area + (new_y[i] + new_y[i + 1]) * (new_x[i + 1] - new_x[i]) / 2
#     return area
#
#
# def get_Metrics(real_score, predict_score):
#     sorted_predict_score = sorted(list(set(np.array(predict_score).flatten())))
#     sorted_predict_score_num = len(sorted_predict_score)
#     thresholdlist = []
#     for i in range(999):
#         threshold = sorted_predict_score[int(math.ceil(sorted_predict_score_num * (i + 1) / 1000) - 1)]
#         thresholdlist.append(threshold)
#     thresholds = np.matrix(thresholdlist)
#     TN = np.zeros((1, len(thresholdlist)))
#     TP = np.zeros((1, len(thresholdlist)))
#     FN = np.zeros((1, len(thresholdlist)))
#     FP = np.zeros((1, len(thresholdlist)))
#     for i in range(thresholds.shape[1]):
#         p_index = np.where(predict_score >= thresholds[0, i])
#         TP[0, i] = len(np.where(real_score[p_index] == 1)[0])
#         FP[0, i] = len(np.where(real_score[p_index] == 0)[0])
#         n_index = np.where(predict_score < thresholds[0, i])
#         FN[0, i] = len(np.where(real_score[n_index] == 1)[0])
#         TN[0, i] = len(np.where(real_score[n_index] == 0)[0])
#     accuracy = (TP + TN) / (TP + TN + FP + FN)
#     sen = TP / (TP + FN)
#     recall = sen
#     spec = TN / (TN + FP)
#     precision = TP / (TP + FP)
#     f1 = 2 * recall * precision / (recall + precision)
#     max_index = np.argmax(f1)
#     max_f1 = f1[0, max_index]
#     max_accuracy = accuracy[0, max_index]
#     max_recall = recall[0, max_index]
#     max_spec = spec[0, max_index]
#     max_precision = precision[0, max_index]
#     return [max_f1, max_accuracy, max_recall, max_spec, max_precision]
