"""Plot decision curve

Author:Li Chao
Email: lichao19870617@163.com
"""


import numpy as np
import matplotlib.pyplot as plt


def plot_dca(pred_prob, true_label):
    """Plot decision curve

    Parameters:
    ----------
    pred_prob:ndarray with shape of [n_samples,]
        Predicted probabilities

    true_label::ndarray with shape of [n_samples,]
        True labels of samples
    """
    
    plt.figure(figsize=(5,5))
    
    get_netbenefit = lambda TP, FP, N, Pt: TP/N - FP/N * Pt/(1 - Pt)
    N = len(pred_prob)
    numberOfPt = 100
    step = (np.max(pred_prob) - np.min(pred_prob)) / numberOfPt
    
    # target dca
    Pt = 0
    NetBenefit = []
    PT = []
    for i in np.arange(1, numberOfPt, 1):
        predict = np.int32(pred_prob>=Pt)
        TP = np.sum((true_label==1)&(predict==1))
        FP = np.sum((true_label==0)&(predict==1))
        PT.append(i/numberOfPt)
        NetBenefit.append(get_netbenefit(TP, FP, N, PT[i-1]))
        Pt = Pt + step
        
    plt.plot(PT, NetBenefit, color='deeppink',  linewidth=2)
    
    
    # Treat None
    plt.plot(PT, np.linspace(0, 0, numberOfPt-1), color='k', linewidth=2)
    
    # Treat All
    PT = []
    NetBenefit = []
    TP = np.sum(true_label==1)
    FP = np.sum(true_label==0)
    for i in np.arange(1, numberOfPt, 1):
        PT.append(i / numberOfPt)
        NetBenefit.append(get_netbenefit(TP, FP, N, PT[i-1]))

    plt.plot(PT, NetBenefit, color="c", linewidth=2)
    
    # Figure settings
    plt.axis([0, 1, -0.05, np.max(NetBenefit)+0.1])
    plt.xlabel('Threshold Probability', fontsize=15)
    plt.ylabel('Net Benefit', fontsize=15)
    plt.legend(['Radiomics', 'None', 'All'])

    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    plt.tight_layout()


if __name__ == "__main__":
    import pandas as pd
    data = pd.read_excel("./tests/dcaDemoData.xlsx", header=None)
    pred_prob, true_label = data[0], data[1]
    plot_dca(pred_prob, true_label)
    plt.savefig("./decision_curve.pdf")