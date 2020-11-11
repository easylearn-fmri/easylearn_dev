import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score


def eval_performance(label_real=None, label_predict=None, decision=None,
                    accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
                    verbose=True, is_showfig=True, legend1='HC', legend2='Patients', is_savefig=False, out_name=None):
    """
    This function is used to evaluate performance of the classification model.
    Parameters:
    ----------
        label_real: real label
        label_predict: predicted label
        decision: model output decision
        accuracy_kfold: accuracy of k-fold cross validation
        sensitivity_kfold: sensitivity of k-fold cross validation
        specificity_kfold: specificity of k-fold cross validation
        AUC_kfold; AUC of k-fold cross validation
        verbose: if print performances
        is_showfig: if show figure
        legend1, legend2: scatter figure legends,
        is_savefig: if save figure to local disk
        out_name: save name of the figure
    TODO: Generate to multiple classification.
    """

    # reshape to one column
    label_real = np.reshape(label_real, [np.size(label_real), -1])
    label_predict = np.reshape(label_predict, [np.size(label_predict), -1])
    decision = np.reshape(decision, [np.size(decision), -1])

    # accurcay, specificity(recall of negative) and
    # sensitivity(recall of positive)
    accuracy = accuracy_score(label_real, label_predict)
    report = classification_report(label_real, label_predict)
    report = report.split('\n')
    specificity = report[2].strip().split(' ')
    sensitivity = report[3].strip().split(' ')
    specificity = float([spe for spe in specificity if spe != ''][2]) 
    sensitivity = float([sen for sen in sensitivity if sen != ''][2])
    # confusion_matrix matrix
#    confusion_matrix = confusion_matrix(label_real, label_predict)

    # roc and auc
    if len(np.unique(label_real)) == 2:
        fpr, tpr, thresh = roc_curve(label_real, decision)
        auc = roc_auc_score(label_real, decision)
    else:
        auc = np.nan

    # print performances
    if verbose:
        print('\naccuracy={:.2f}\n'.format(accuracy))
        print('sensitivity={:.2f}\n'.format(sensitivity))
        print('specificity={:.2f}\n'.format(specificity))
        if not np.isnan(auc):
            print('auc={:.2f}\n'.format(auc))
        else:
            print('Multi-Classification can not calculate the AUC\n')

    if is_showfig and auc:
        # Import modules
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.pyplot import MultipleLocator
        
        if ((accuracy_kfold) and (sensitivity_kfold) and (specificity_kfold)):
            fig, ax = plt.subplots(1,3, figsize=(15,5))
        else:
            fig, ax = plt.subplots(1,2, figsize=(10,5))
        # Plot classification 2d figure
        decision_0 = decision[label_real == 0]
        decision_1 = decision[label_real == 1]
        ax[0].scatter(decision_0, np.arange(0, len(decision_0)), marker="o", linewidth=2, color='paleturquoise')
        ax[0].scatter(decision_1, np.arange(len(decision_0), len(decision)), marker="*", linewidth=2, color='darkturquoise')
        # Grid and spines
        ax[0].grid(False)
        ax[0].spines['bottom'].set_position(('axes', 0))
        ax[0].spines['left'].set_position(('axes', 0))
        ax[0].spines['top'].set_linewidth(1.5)
        ax[0].spines['right'].set_linewidth(1.5)
        ax[0].spines['bottom'].set_linewidth(1.5)
        ax[0].spines['left'].set_linewidth(1.5)
        ax[0].plot(np.zeros(10), np.linspace(0, len(decision),10), '--', color='k', linewidth=1.5)
        ax[0].axis([-np.max(np.abs(decision))-0.2, np.max(np.abs(decision))+0.2, 0 - len(decision) / 10, len(decision) + len(decision) / 10]) # x and y lim
        ax[0].set_xlabel('Decision values', fontsize=15)
        ax[0].set_ylabel('Subjects', fontsize=15)
        num1, num2, num3, num4 = 0, 1.01, 3, 0
        ax[0].legend(['Discriminant line', legend1, legend2],
                  bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)

        # Plot ROC
        auc = '{:.2f}'.format(auc)
        auc = eval(auc)
        ax[1].set_title(f'ROC Curve (AUC = {auc})', fontsize=15, fontweight='bold')
        ax[1].set_xlabel('False Positive Rate', fontsize=15)
        ax[1].set_ylabel('True Positive Rate', fontsize=15)
        ax[1].plot(fpr, tpr, marker=".", markersize=5, linewidth=2, color='darkturquoise')
        plt.tick_params(labelsize=12)
        # Grid and spines
        ax[1].grid(False)
        ax[1].spines['top'].set_linewidth(1.5)
        ax[1].spines['right'].set_linewidth(1.5)
        ax[1].spines['bottom'].set_position(('axes', 0))
        ax[1].spines['left'].set_position(('axes', 0))
        ax[1].spines['bottom'].set_linewidth(1.5)
        ax[1].spines['left'].set_linewidth(1.5)
        # Plot random line
        ax[1].plot(np.linspace(0, 1,10), np.linspace(0, 1,10), '--', color='k', linewidth=1)
        
        # Plot Bar
        if ((accuracy_kfold) and (sensitivity_kfold) and (specificity_kfold)):
            mean = [np.mean(accuracy_kfold), np.mean(sensitivity_kfold),np.mean(specificity_kfold),np.mean(AUC_kfold)]
            std = [np.std(accuracy_kfold), np.std(sensitivity_kfold), np.std(specificity_kfold), np.std(AUC_kfold)]
            # Plot
            ax[2].bar(np.arange(0,len(mean)), mean, yerr = std, capsize=5, linewidth=2, color='darkturquoise')
            ax[2].tick_params(labelsize=12)
            ax[2].set_title('Classification performances', fontsize=15, fontweight='bold')
            plt.xticks(np.arange(0,len(mean)), ['Accuracy', 'Sensitivity', 'Specificity', 'AUC'], fontsize=12, rotation=45)
            # Setting
            ax[2].spines['top'].set_linewidth(1.5)
            ax[2].spines['right'].set_linewidth(1.5)
            ax[2].spines['bottom'].set_linewidth(1.5)
            ax[2].spines['left'].set_linewidth(1.5)
            plt.grid(axis='y')
            y_major_locator=MultipleLocator(0.1)
            ax[2].yaxis.set_major_locator(y_major_locator)
        
        # Save figure to PDF file
        plt.tight_layout()
        plt.subplots_adjust(wspace = 0.2, hspace = 0)
        if is_savefig:
            pdf = PdfPages(out_name)
            pdf.savefig()
            pdf.close()
            plt.show()
        else:
            plt.show()

    return accuracy, sensitivity, specificity, auc

if __name__ == "__main__":
    pass