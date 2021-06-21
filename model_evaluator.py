# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
import scipy as sci

from eslearn.utils.timer import  timer

class ModelEvaluator():
    """Model evaluation

    """
    
    def binary_evaluator(self, true_label=None, predict_label=None, predict_score=None,
                        accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
                        verbose=True, is_showfig=True, legend1='HC', legend2='Patients', is_savefig=False, out_name=None):
        
        """
        This function is used to evaluate performance of the binary classification model.

        Parameters:
        ----------
        true_label: 1d array with N-sample items
            Ground truth labels.

        predict_label: 1d array with N-sample items
            predicted label

        predict_score: N-sample by N-class matrix 
            Output predict_score of model

        accuracy_kfold: 1d array with K items
            accuracy of k-fold cross validation

        sensitivity_kfold: 1d array with K items
            sensitivity of k-fold cross validation

        specificity_kfold: 1d array with K items
            specificity of k-fold cross validation

        AUC_kfold: 1d array with K items
             AUC of k-fold cross validation

        verbose: bool
             if print performances

        is_showfig: bool
             if show figure

        legend1, legend2: str
            scatter figure legends,

        is_savefig: bool
            if save figure to local disk

        out_name: str
            output name of the figure
            
        Returns:
        -------
        accuracy:
        sensitivity:
        specificity: 
        auc:
        confusion_matrix_values:
        """
        
        # One Hot encode
        # lcode=LabelEncoder()
        # true_label=lcode.fit_transform(true_label)
        # predict_label = lcode.transform(predict_label)
        
        # reshape to one column
        true_label = np.reshape(true_label, [np.size(true_label), ])
        predict_label = np.reshape(predict_label, [np.size(predict_label), ])
        predict_score = np.array(predict_score)
        if len(np.shape(predict_score)) > 1:
            predict_score = predict_score[:,-1]

        # accurcay, specificity and sensitivity 
        accuracy = np.float64(f"{accuracy_score(true_label, predict_label):.2f}")
        # confusion_matrix matrix
        confusion_matrix_values = confusion_matrix(true_label, predict_label)
        tn,fp,fn,tp = confusion_matrix_values.ravel()
        sensitivity = float(f"{tp/(tp+fn):.2f}")
        specificity = float(f"{tn/(tn+fp):.2f}")

        # roc and auc
        if len(np.unique(true_label)) == 2:
            fpr, tpr, thresh = roc_curve(true_label, predict_score)
            auc = roc_auc_score(true_label, predict_score)
        else:
            auc = None

        # print performances
        if verbose:
            print('\naccuracy={:.2f}\n'.format(accuracy))
            print('sensitivity={:.2f}\n'.format(sensitivity))
            print('specificity={:.2f}\n'.format(specificity))
            if auc is not None:
                print('auc={:.2f}\n'.format(auc))
            else:
                print('Multi-Classification or only one class can not calculate the AUC\n')

        #%% Plot
        try:
            matplotlib.use('Qt5Agg')
        except Exception as e:
            print(f'{e}')
                
        if not is_showfig:
            matplotlib.use('PDF')
            
        fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(9,4))

        # Plot classification 2d scatter
        decision_0 = predict_score[true_label == 0]
        decision_1 = predict_score[true_label == 1]
        # Identify the separation line located at the 0 or 0.5
        # if np.min(predict_score) >= 0:
        #     separation_point = 0.5
        # else:
        #     separation_point = 0
        # if np.ndim(predict_score) == 2:
        # predict_score = predict_score[:,-1]  # Retained the positive probability
        # ax[0].scatter(decision_0, np.arange(0, len(decision_0)), marker="o", linewidth=2, color='paleturquoise')
        # ax[0].scatter(decision_1, np.arange(len(decision_0), len(predict_score)), marker="*", linewidth=2, color='darkturquoise')
        # # TODO: Identify the separation line located at the 0 or 0.5
        # ax[0].plot(np.zeros(10) + separation_point, np.linspace(0, len(predict_score),10), '--', color='k', linewidth=1.5)
        # if separation_point == 0.5:
        #     ax[0].axis([-0.05, 1.05, 0 - len(predict_score) / 20, len(predict_score) + len(predict_score) / 20]) # x and y lim
        # else:
        #     ax[0].axis([-1.05, 1.05, 0 - len(predict_score) / 20, len(predict_score) + len(predict_score) / 20]) # x and y lim               
        # ax[0].set_xlabel('Decision values', fontsize=10)
        # ax[0].set_ylabel('Subjects', fontsize=10)

        # Plot distribution
        sns.kdeplot(decision_0, shade=True, ax = ax[0])  
        sns.kdeplot(decision_1, shade=True, ax = ax[0])
        
        # Grid and spines
        ax[0].grid(False)
        ax[0].set_title('Distribution of prediction in each group', fontsize=10, fontweight='bold')
        ax[0].spines['bottom'].set_position(('axes', 0))
        ax[0].spines['left'].set_position(('axes', 0))
        ax[0].spines['top'].set_linewidth(1)
        ax[0].spines['right'].set_linewidth(1)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['bottom'].set_linewidth(1)
        ax[0].spines['left'].set_linewidth(1)
        ax[0].set_xlabel('Decision values', fontsize=8)
        ax[0].set_ylabel('Density', fontsize=8)
        num1, num2, num3, num4 = 0, 1.2, 3, 0
        ax[0].legend([legend1, legend2], bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)

        # Plot ROC
        if auc is not None:
            auc = '{:.2f}'.format(auc)
            auc = eval(auc)
            ax[1].set_title(f'ROC Curve (AUC = {auc})', fontsize=10, fontweight='bold')
            ax[1].set_xlabel('False Positive Rate', fontsize=8)
            ax[1].set_ylabel('True Positive Rate', fontsize=8)
            ax[1].plot(fpr, tpr, markersize=2, linewidth=1, color=[0, 84/255, 95/255])
            plt.tick_params(labelsize=12)
            # Grid and spines
            ax[1].grid(False)
            ax[1].spines['top'].set_linewidth(1)
            ax[1].spines['right'].set_linewidth(1)
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['bottom'].set_position(('axes', 0))
            ax[1].spines['left'].set_position(('axes', 0))
            ax[1].spines['bottom'].set_linewidth(1)
            ax[1].spines['left'].set_linewidth(1)
            # Plot random line
            ax[1].plot(np.linspace(0, 1,10), np.linspace(0, 1,10), '--', color='k', linewidth=1)

        # Plot Bar
        if (accuracy_kfold is not None) and (sensitivity_kfold is not None) and (specificity_kfold is not None):
            performances = [np.mean(accuracy_kfold), np.mean(sensitivity_kfold), np.mean(specificity_kfold)]
            std = [np.std(accuracy_kfold), np.std(sensitivity_kfold), np.std(specificity_kfold)]
            ax[2].bar(np.arange(0,len(performances)), performances, yerr = std, capsize=5, linewidth=2, color='darkturquoise')
            
            bid = np.arange(0,len(performances))
            for (ibar, perf_, std_) in zip (bid, performances, std):
                ax[2].text(ibar, 0.05, f"{perf_:.2f}±{std_:.2f}", rotation=90) 
        else:
            performances = [accuracy, sensitivity, specificity]
            ax[2].bar(np.arange(0, len(performances)), performances, linewidth=2, color='darkturquoise')
            
            bid = np.arange(0,len(performances))
            for (ibar, perf_) in zip (bid, performances):
                ax[2].text(ibar, 0.05, f"{perf_:.2f}", rotation=90) 

        ax[2].tick_params(labelsize=12)
        ax[2].set_title('Classification performances', fontsize=10, fontweight='bold')
        ax[2].set_xticks(np.arange(0,len(performances)))
        ax[2].set_xticklabels(('Accuracy', 'Sensitivity', 'Specificity'), rotation=45, fontsize=8)
        # Setting
        ax[2].spines['top'].set_linewidth(1)
        ax[2].spines['right'].set_linewidth(1)
        ax[2].spines['top'].set_visible(False)
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['bottom'].set_linewidth(1)
        ax[2].spines['left'].set_linewidth(1)
        # ax[2].grid(axis='y', linestyle='-.')
        y_major_locator=MultipleLocator(0.1)
        ax[2].yaxis.set_major_locator(y_major_locator)
        
        # # Plot calibration curve
        # if auc is not None:
        #     # predict_score = (predict_score - predict_score.min()) / (predict_score.max() - predict_score.min())
        #     fraction_of_positives, mean_predicted_value = calibration_curve(true_label, predict_score, n_bins=10, normalize=True)
        #     ax[1][1].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        #     ax[1][1].plot(mean_predicted_value, fraction_of_positives, "-.", color="k")
        #     # Setting
        #     ax[1][1].spines['top'].set_linewidth(1)
        #     ax[1][1].spines['right'].set_linewidth(1)
        #     ax[1][1].spines['bottom'].set_linewidth(1)
        #     ax[1][1].spines['left'].set_linewidth(1)
        #     ax[1][1].set_xlabel("Predicted probability of positives", fontsize=10)
        #     ax[1][1].set_ylabel("Fraction of positives", fontsize=10)
        #     ax[1][1].set_title("Calibration curves", fontsize=12, fontweight='bold')
        #     ax[1][1].set_ylim([-0.05, 1.05])

        # Save figure to PDF file
        plt.tight_layout()
        plt.subplots_adjust(wspace = 0.3, hspace = 0.4)
        if is_savefig:
            pdf = PdfPages(out_name)
            pdf.savefig()
            pdf.close()
            
        if is_showfig:
            plt.show()
            # plt.pause(5)
            # plt.close()
            
        #%% Plot
        try:
            matplotlib.use('Qt5Agg')
        except Exception as e:
            print(f'{e}')
            
        return accuracy, sensitivity, specificity, auc, confusion_matrix_values

    def regression_evaluator(self, real_target, predict_score, reg_metrics, 
                             is_showfig=True, is_savefig=False, out_name=None):

        """Evaluation of regression

        Parameters:
        ----------
        real_target: ndarray or list et al.
            Real targets

        predict_proba: ndarray or list
            Predicted scores

        reg_metrics: list
            Regression metric scores, e.g., MAE

        is_showfig: bool
         If show figure

        is_savefig: bool
         If save figure

        out_name: str
            Output file name of saved figure (pdf)
        """
        
        mean_metrics = np.mean(reg_metrics)
        std_metrics = np.std(reg_metrics)
        coef = np.corrcoef(real_target, predict_score)[0,1]
        
        # Set matplotlib backend
        try:
            matplotlib.use('Qt5Agg')
        except Exception as e:
            print(f'{e}')
             
        if not is_showfig:
            matplotlib.use('PDF')
        
        ax = sns.jointplot(x=predict_score, 
                           y=real_target, 
                            kind='reg', 
                            size=5)
                            # scatter_kws={'s': 20})
                # Setting
        # ax=plt.gca()
        # ax.xaxis.set_major_locator(x_major_locator)
        # ax.yaxis.set_major_locator(x_major_locator)
        # ax.set_ylim(-0.05,1.05)
        # ax.set_xlim(-0.05,1.05)

        # plt.rcParams["font.family"] = "arial"
        # plt.rcParams["font.weight"] = "bold"
        # plt.rcParams["xtick.major.width"] = 2
        # plt.rcParams["ytick.major.width"] = 2
        # plt.rcParams['xtick.direction'] = 'out'
        # plt.rcParams['ytick.direction'] = 'out'
        
        ax.set_axis_labels("Predicted score", "Real score", fontsize=15)
        plt.tight_layout()
        xmargin = (np.max(predict_score)-np.min(predict_score))/50
        ymargin = (np.max(real_target)-np.min(real_target))/50
        ax.ax_joint.text(np.min(predict_score)+xmargin, np.max(real_target)-ymargin, f"MAE={mean_metrics :.2f}±{std_metrics:.2f}\nR={coef:.2f}",
                         fontweight="normal", fontsize=10)
        
        if is_savefig:
            pdf = PdfPages(out_name)
            pdf.savefig()
            pdf.close()
            
        if is_showfig:
            plt.show()
            plt.pause(5)
            plt.close()


if __name__ == "__main__":
    pass