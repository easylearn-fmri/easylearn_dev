"""
@author: Muhammet Balcilar
LITIS Lab, Rouen, France
muhammetbalcilar@gmail.com

"""


import numpy as np
from scipy.io import loadmat
import networkx as nx
import numpy.linalg as linalg
from sklearn.preprocessing import StandardScaler
def loadCobraData(fname='cobradat.mat'):

    # read mat file
    mat = loadmat(fname)
    # make Adjagency and Connectivity matrixes as list
    A=[];C=[]
    for i in range(0,mat['A'].shape[0]):
        A.append(mat['A'][i][0])
        C.append(mat['C'][i][0])
    # read global features descriptors
    F=mat['F']

    # read global descr names
    Vnames=[]
    for i in range(0,mat['Vnames'][0].shape[0]):
        Vnames.append(mat['Vnames'][0][i][0])

    # read file name and molecule names
    FILE=[];NAME=[]
    for i in range(0,mat['FILE'].shape[0]):
        FILE.append(mat['FILE'][i][0][0])
        NAME.append(mat['NAME'][i][0][0])

    # read atomic descriptor name
    Anames=[]
    for i in range(0,mat['Anames'].shape[1]):
        Anames.append(mat['Anames'][0][i][0])
    # read atomic descriptors
    TT=[];Atom=[]
    for i in range(0,mat['TT'].shape[0]):
        TT.append(mat['TT'][i][0])
        SA=[]
        for j in range(0,mat['Atom'][i][0].shape[0]):
            SA.append(mat['Atom'][i][0][j][0][0])
        Atom.append(SA)
    #TT Atom Anames 

    return A,C,F,TT,Atom,Anames,Vnames,FILE,NAME


def loadCobraGraphAsNetworkx(fname='cobradat.mat'):
    
    # read dataset
    A,C,F,TT,Atom,Anames,Vnames,FILE,NAME=loadCobraData(fname)
    G=[];N=[];F=[]
    #an=0
    for i in range(0,len(FILE)):
        name = FILE[i]
        atm=Atom[i]
        AA=A[i]

        edge=[]
        for j in range(0,len(atm)-1):
            for k in range(j,len(atm)):
                if AA[j,k]==1:
                    edge.append([str(j), str(k)])
        graph = nx.from_edgelist(edge)

        feat={}
        for j in range(0,len(atm)):            
                if atm[j][0]=='C' and atm[j][1]!='l':
                    feat[str(j)]={'Label': u'C'} #, 'label': str(j)}                    
                elif atm[j][0]=='H':
                    feat[str(j)]={'Label': u'H'} #, 'label': str(j)}
                elif atm[j][0]=='O':
                    feat[str(j)]={'Label': u'O'} #, 'label': str(j)}
                elif atm[j][0]=='N':
                    feat[str(j)]={'Label': u'N'} #, 'label': str(j)}
                elif atm[j][0]=='F':
                    feat[str(j)]={'Label': u'F'} #, 'label': str(j)}
                elif atm[j][0:2]=='Br':
                    feat[str(j)]={'Label': u'Br'} #, 'label': str(j)}
                elif atm[j][0]=='S' and atm[j][1]!='i':
                    feat[str(j)]={'Label': u'S'} #, 'label': str(j)}                
                elif atm[j][0:2]=='Cl':
                    feat[str(j)]={'Label': u'Cl'} #, 'label': str(j)}
                elif atm[j][0:2]=='Si':
                    feat[str(j)]={'Label': u'Si'} #, 'label': str(j)}
                else:
                    feat[str(j)]={'Label': u'X'} #, 'label': str(j)}

                #an+=1        
        nx.set_node_attributes(graph, feat)        
        G.append(graph)
    return G,FILE

def normalize_wrt_train(trX,tsX):
    """Normalize signal data S and global data GF respect to train set
    trX is list of [S,U,B,Nd,GF]    
    """
    n=int(np.round(1/trX[3][0][0]))
    trainX=trX[0][0][0:n]
    for i in range(1,len(trX[0])):
        n=int(np.round(1/trX[3][i][0]))
        trainX=np.vstack((trainX,trX[0][i][0:n]))

    n=int(np.round(1/tsX[3][0][0]))
    testX=tsX[0][0][0:n]
    for i in range(1,len(tsX[0])):
        n=int(np.round(1/tsX[3][i][0]))
        testX=np.vstack((testX,tsX[0][i][0:n]))

    scaler = StandardScaler()
    scaler.fit(trainX)
    trainX=scaler.transform(trainX)
    testX=scaler.transform(testX)
    trainX[:,17]=1
    testX[:,17]=1

    # mn=trainX.mean(axis=0)
    # sd=trainX.std(axis=0)
    # trainX=(trainX-mn)/sd
    # testX=(testX-mn)/sd

    mn=trainX.min(axis=0)
    mx=trainX.max(axis=0)

    for i in range(0,50):
        testX[np.where(testX[:,i]<mn[i]),i]=mn[i]
        testX[np.where(testX[:,i]>mx[i]),i]=mx[i]

    mn=trX[4].mean(axis=0)
    sd=trX[4].std(axis=0)
    trX[4]=(trX[4]-mn)/sd
    tsX[4]=(tsX[4]-mn)/sd

    
    b=0
    for i in range(0,len(trX[0])):
        n=int(np.round(1/trX[3][i][0]))
        trX[0][i][0:n]=trainX[b:b+n,:]
        b=b+n
    b=0
    for i in range(0,len(tsX[0])):
        n=int(np.round(1/tsX[3][i][0]))
        tsX[0][i][0:n]=testX[b:b+n,:]
        b=b+n
    return trX,tsX

def laplacian(W, normalized=0):
    """Return graph Laplacian"""

    # Degree matrix.
    W=1.0*W
    d = W.sum(axis=0)

    # Laplacian matrix.
    if normalized==0:
        D = np.diag(d)
        L = D - W
    elif normalized==1:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = np.diag(d)
        I = np.eye(d.size, dtype=W.dtype)
        L = I - D.dot(W).dot(D)  
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = np.diag(d)
        I = np.eye(d.size, dtype=W.dtype)
        L = D.dot(W).dot(D) 

    return L

def eigenValuesVectors(A,sorted=True):
    """Return sorted eigenvalues and corresponding vectors"""

    eigenValues, eigenVectors = linalg.eigh(A)
    if sorted:
        idx = (-eigenValues).argsort()[::-1]   
        eigenValues = np.real(eigenValues[idx])
        eigenVectors = np.real(eigenVectors[:,idx])
        eigenValues[np.where(eigenValues<0)]=0 
    return eigenValues,eigenVectors


def bspline_basis(K, v, x, degree=3):

    def cox_deboor(k, d):        
        
        if d == 0:
            ret= np.zeros((x.shape[0],),dtype=np.float)
            ret[np.where( (x - kv[k] >= 0) * (x - kv[k + 1] < 0)==True)]=1

        else:
            
            denom1 = kv[k + d] - kv[k]
            term1 = 0
            if denom1 > 0:
                term1 = ((x - kv[k]) / denom1) * cox_deboor(k, d - 1)
            
            
            denom2 = kv[k + d + 1] - kv[k + 1]
            term2 = 0
            if denom2 > 0:
                term2 = ((-(x - kv[k + d + 1]) / denom2) * cox_deboor(k + 1, d - 1))
            
            ret= term1 + term2
        return ret

    basis=np.zeros((x.shape[0],K))
    kv1 = v.min() * np.ones((degree,))
    kv2 = np.linspace(v.min(), v.max(), K-degree+1)
    kv3 = v.max() * np.ones((degree,))
    kv = np.hstack((kv1 ,kv2 ,kv3))

    for k in range(0,K):
        basis[:,k]=cox_deboor(k, degree)

    return basis
    
    #%basis(end,end)=1;



def prepare_data(A,K,mxeigv=None,degree=2):

    L=laplacian(A)
    V,U=eigenValuesVectors(L) 
    if mxeigv is None:
        mxeigv=V.max()
    nv=np.linspace(-0.0001,mxeigv,K)
    # nv=np.array([ 0.  ,        0.13761198 , 0.37610449 , 0.56015261 , 0.74230873 , 0.97686203,
    #    1.00783197 , 1.30345163 , 1.569913  ,  1.82041479 , 2.    ,      2.09325829,
    #    2.45462153,  2.70768938 , 2.97389003 , 3.14016581 , 3.45690623 , 3.76513438,
    #    4.0641928 ,  4.46580989 , 4.8878061 ,  5.28384145 , 5.83663165 , 6.40090109,
    #    7.37654178 , 8.65535957, 11.26793143 ,37.09755486, V[-2] , V[-1]])

    B=np.zeros((V.shape[0],nv.shape[0]))
    for i in range(0,V.shape[0]):
        i1=np.where(nv<=V[i])[0][-1]
        i2=np.where(nv>=V[i])[0]
        if len(i2)==0:
            continue
        i2=i2[0]
        if i1==i2:
            B[i,i1]=1
        else:
            B[i,i2]=(nv[i2]-V[i])/(nv[i2]-nv[i1])
            B[i,i1]=1-B[i,i2]

    #    tmp=np.exp(-0.1*np.abs(nv-V[i]))
    #    tmp=tmp/tmp.sum()
    #    B[i,:]=tmp 
    #    #B[i,np.argmin(np.abs(nv-V[i]))]=1
    
    #B=bspline_basis(K,nv,V,degree=degree)
    #if B[-1,:].sum()==0:
    #    B[-1,-1]=1

    # B=np.zeros((V.shape[0],K))
    # for i in range(0,V.shape[0]):
    #     B[i,np.argmin(np.abs(nv-V[i]))]=1
    return U,B,V
        



    
        
   