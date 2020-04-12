""" Selection features using F-score

Copy form https://www.zealseeker.com/archives/f-score-for-feature-selection-python/

TODO: Extend to multiple-class classification (Refrence: http://www.joca.cn/EN/Y2010/V30/I4/993#)
"""


def fscore_core(np,nn,xb,xbp,xbn,xkp,xkn):
    '''
    np: number of positive features
    nn: number of negative features
    xb: list of the average of each feature of the whole instances
    xbp: list of the average of each feature of the positive instances
    xbn: list of the average of each feature of the negative instances
    xkp: list of each feature which is a list of each positive instance
    xkn: list of each feature which is a list of each negatgive instance
    reference: http://link.springer.com/chapter/10.1007/978-3-540-35488-8_13
    '''

    def sigmap (i,np,xbp,xkp):
        return sum([(xkp[i][k]-xbp[i])**2 for k in range(np)])

    def sigman (i,nn,xbn,xkn):
        print sum([(xkn[i][k]-xbn[i])**2 for k in range(nn)])
        return sum([(xkn[i][k]-xbn[i])**2 for k in range(nn)])

    n_feature = len(xb)
    fscores = []
    for i in range(n_feature):
        fscore_numerator = (xbp[i]-xb[i])**2 + (xbn[i]-xb[i])**2
        fscore_denominator = (1/float(np-1))*(sigmap(i,np,xbp,xkp))+ \
                             (1/float(nn-1))*(sigman(i,nn,xbn,xkn))
        fscores.append(fscore_numerator/fscore_denominator)

    return fscores

def fscore(feature,classindex):
    '''
    feature: a matrix whose row indicates instances, col indicates features
    classindex: 1 indicates positive and 0 indicates negative
    '''
    n_instance = len(feature)
    n_feature  = len(feature[0])
    np = sum(classindex)
    nn = n_instance - np
    xkp =[];xkn =[];xbp =[];xbn =[];xb=[]
    for i in range(n_feature):
        xkp_i = [];xkn_i = []
        for k in range(n_instance):
            if classindex[k] == 1:
                xkp_i.append(feature[k][i])
            else:
                xkn_i.append(feature[k][i])
        xkp.append(xkp_i)
        xkn.append(xkn_i)
        sum_xkp_i = sum(xkp_i)
        sum_xkn_i = sum(xkn_i)
        xbp.append(sum_xkp_i/float(np))
        xbn.append(sum_xkn_i/float(nn))
        xb.append((sum_xkp_i+sum_xkn_i)/float(n_instance))
    return fscore_core(np,nn,xb,xbp,xbn,xkp,xkn)