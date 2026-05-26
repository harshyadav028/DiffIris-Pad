import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from scipy.special import softmax
import pylab as pl
import csv
import os
import torch


class evaluation:
    def __init__(self):
        return None

    def plot_confusion_matrix(self,cm,method,path, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center", fontsize=30,
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label',fontsize=20)
        plt.xlabel('Predicted label',fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(path,method +'_ConfMatrix.jpg'))

    def get_threshold(self, fprs, thresholds, fpr):
    # Getting threshold for particular fpr
       threshold = 0
       for x in range(0, fprs.size):
         if fprs[x] >= fpr:
             break
         threshold = thresholds[x]
       return threshold

    def get_result(self, method,imgNames, true_label,predict_score,pai,path, minThreshold = -1):

        # Getting predicted scores
        #predict_score = torch.nn.functional.softmax(predict_score, dim=1)
        predict = np.array(predict_score)
        #print(predict.shape)
        predict = softmax(predict, axis=1)
        if(len(predict.shape)==2):
            predict =  predict[:, 1]
        elif(len(predict.shape)==3):
            predict = predict[:, :, 1]

        # Normalization of scores in [0,1]
        #print(predict.shape)
        #predictScore = (predict-min(predict))/ (max(predict) - min(predict))
        predictScore = predict
        print('Max Score:'+ str(max(predict)))
        print('Min Score:'+ str(min(predict)))

        # Saving image or video name with match score
        if imgNames != 'None':
            imgNameScore=[]
            for i in range(len(imgNames)):
                imgNameScore.append([imgNames[i], float(true_label[i]), predictScore[i]])
            with open(os.path.join(path, method + '_Match_Scores.csv'), 'w', newline='') as fout:
                writer = csv.writer(fout)
                writer.writerows(imgNameScore)
        # Histogram plot
        live = []
        [live.append(predictScore[i]) for i in range(len(true_label)) if (true_label[i] == 0)]
        spoof = []
        [spoof.append(predictScore[j]) for j in range(len(true_label)) if (true_label[j] == 1)]
        pai_type=[]
        [pai_type.append(pai[j]) for j in range(len(true_label)) if (true_label[j] == 1)]


        # Plot ROC curves in semilog scale
        (fprs, tprs, thresholds) = roc_curve(true_label, predictScore)
        plt.figure()
        plt.semilogx(fprs, tprs, label=method)
        plt.grid(True, which="major")
        plt.legend(loc='lower right', fontsize=15)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xticks([0.001, 0.01, 0.1, 1])
        plt.xlabel('False Detection Rate')
        plt.ylabel('True Detection Rate')
        plt.xlim((0.0005, 1.01))
        plt.ylim((0, 1.02))
        plt.plot([0.002, 0.002], [0, 1], color='#A0A0A0', linestyle='dashed')
        plt.plot([0.001, 0.001], [0, 1], color='#A0A0A0', linestyle='dashed')
        plt.plot([0.01, 0.01], [0, 1], color='#A0A0A0', linestyle='dashed')
        plt.savefig(os.path.join(path,method +"_ROC.jpg"))

        #Plot Raw ROC curves
        plt.figure()
        plt.plot(fprs, tprs)
        plt.grid(True, which="major")
        plt.legend(method, loc='lower right', fontsize=15)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xticks([0.01, 0.1, 1])
        plt.xlabel('False Detection Rate')
        plt.ylabel('True Detection Rate')
        plt.xlim((0.0005, 1.01))
        plt.ylim((0, 1.02))
        plt.savefig(os.path.join(path,method +"_RawROC.jpg"))

        # Calculation of TDR at 0.2% , 0.1% and  5% FDR
        with open(os.path.join(path , method +'_TDR-ACER.csv'), mode='w+') as fout:
            fprArray = [0.001,0.002, 0.01, 0.05]
            for fpr in fprArray:
                tpr = np.interp(fpr, fprs, tprs)
                threshold = self.get_threshold(fprs, thresholds, fpr)
                fout.write("TDR @ FDR, threshold: %f @ %f ,%f\n" % (tpr, fpr, threshold))
                print("TDR @ FDR, threshold: %f @ %f ,%f " % (tpr, fpr, threshold))

        # Calculation of APCER, BPCER and ACER
            # if minThreshold == -1:
            #     minACER= 1000
            #     for thresh in np.arange(0,1,0.025):
            #         APCER = np.count_nonzero(np.less(spoof,thresh))/len(spoof)
            #         BPCER = np.count_nonzero(np.greater_equal(live,thresh))/len(live)
            #         ACER = (APCER*len(spoof) + BPCER*len(live))/(len(spoof)+len(live))
            #         if ACER < minACER:
            #             minThreshold = thresh
            #             minAPCER = APCER
            #             minBPCER = BPCER
            #             minACER = ACER
            unique_pais = np.unique(pai_type)
            if minThreshold == -1:
                minACER = 1000
                minACER_PAI = 1000  # for PAI-weighted ACER
                for thresh in np.arange(0, 1, 0.025):
                    # Basic APCER and BPCER
                    APCER = np.count_nonzero(np.less(spoof, thresh)) / len(spoof)
                    BPCER = np.count_nonzero(np.greater_equal(live, thresh)) / len(live)

                    # Standard ISO 30107-3 ACER
                    ACER = (APCER + BPCER) / 2

                    # PAI-weighted APCER
                    weighted_APCER = 0.0
                    total_spoof = len(spoof)
                    for pai_class in unique_pais:
                        indices = np.where(np.array(pai_type) == pai_class)[0]
                        if len(indices) == 0:
                            continue
                        #print(indices)
                        np_spoof=np.array(spoof)
                        apcer_i = np.count_nonzero(np.less(np_spoof[indices], thresh)) / len(indices)
                        weighted_APCER +=  (1/len(unique_pais)) * apcer_i
                        #fout.write("Attack = %s , APCER = %f\n" % (pai_class,apcer_i))

                    # PAI-aware ACER
                    ACER_PAI = (weighted_APCER + BPCER) / 2

                    # Save best threshold (can be tuned to use ACER or ACER_PAI)
                    if ACER < minACER:
                        minThreshold = thresh
                        minAPCER = APCER
                        minAPCER_weighted = weighted_APCER
                        minBPCER = BPCER
                        minACER = ACER
                        minACER_PAI = ACER_PAI

                fout.write("Weighted APCER and APCER and BPCER @ ACER and ACER weighted and threshold: %f and %f and %f @ %f and %f and %f\n" % (minAPCER_weighted,minAPCER, minBPCER, minACER,minACER_PAI ,minThreshold))
                print("Weighted APCER, APCER and BPCER @ ACER and ACER weighted, threshold: %f, %f and %f @ %f and %f, %f\n" % (minAPCER_weighted,minAPCER, minBPCER, minACER,minACER_PAI ,minThreshold))
            else:
                APCER = np.count_nonzero(np.less(spoof, minThreshold)) / len(spoof)
                BPCER = np.count_nonzero(np.greater_equal(live, minThreshold)) / len(live)
                weighted_APCER = 0.0
                total_spoof = len(spoof)
                for pai_class in unique_pais:
                    indices = np.where(np.array(pai_type) == pai_class)[0]
                    if len(indices) == 0:
                        continue
                    np_spoof=np.array(spoof)
                    apcer_i = np.count_nonzero(np.less(np_spoof[indices], minThreshold)) / len(indices)
                    weighted_APCER += (1/len(unique_pais)) * apcer_i
                    fout.write("Attack = %s , APCER = %f\n" % (pai_class,apcer_i))
                ACER_PAI = (weighted_APCER + BPCER) / 2
                ACER = (APCER + BPCER) / 2
                fout.write("Equally Weighted APCER and APCER and BPCER @ ACER and Equally weighted ACER and threshold: %f and %f and %f @ %f and %f and %f\n" % (weighted_APCER,APCER, BPCER, ACER,ACER_PAI ,minThreshold))
                print("Weighted APCER, APCER and BPCER @ ACER and ACER weighted, threshold: %f, %f and %f @ %f and %f, %f\n" % (weighted_APCER,APCER, BPCER, ACER,ACER_PAI ,minThreshold))

        bins = np.linspace(np.min(np.array(spoof + live)), np.max(np.array(spoof + live)), 60)
        plt.figure()
        plt.hist(live, bins, alpha=0.5, label='Bonafide', density=True, edgecolor='black', facecolor='g')
        plt.hist(spoof, bins, alpha=0.5, label='PA', density=True, edgecolor='black',facecolor='r' )
        plt.axvline(x=minThreshold)
        plt.xlim(0.0, 1.0)
        plt.legend(loc='upper right', fontsize=15)
        plt.xlabel('Scores')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(path, method +"_Histogram.jpg"))

        # Calculation of Confusion matrix
        #threshold = self.get_threshold(fprs, thresholds, 0.002)
        predict = predictScore >= minThreshold
        predict_label =[]
        [predict_label.append(int(predict[i])) for i in range(len(predict))]
        conf_matrix = confusion_matrix(true_label, predict_label)   # 0 for live and 1 for spoof
        print(conf_matrix)

        # Plot non-normalized confusion matrix
        np.set_printoptions(precision=2)
        class_names = ['0', '1']
        self.plot_confusion_matrix(conf_matrix, method,path, classes=class_names,normalize=False)

        # Saving evaluation measures
        pickle.dump((fprs,tprs,minThreshold,tpr,fpr,conf_matrix), open(os.path.join(path,method +".pickle"), "wb"))
        errorIndex=[]
        [errorIndex.append(i) for i in range(len(true_label)) if true_label[i] != predict_label[i]]
        return errorIndex, predictScore, minThreshold


if __name__ == '__main__':
    true_label= [0,0,0,0,0,1,1,1,1,1]
    predict_score = [[0.8,0.20], [0.44, 0.56],[0.6,0.4],[0.7, 0.3], [0.9,0.1],[0.2,0.8],[0.3,0.7],[0.4,0.6],[0.01,0.99],[0.1, 0.9]]
    obvResult = evaluation()
    predict_score = np.array(predict_score)
    predict_score = predict_score.reshape((10,1, 2))
    result = obvResult.get_result('Test', 'None', true_label, predict_score, "../TempData/results")

