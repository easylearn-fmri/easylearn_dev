# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:29:59 2018
奇异值分解 Singular  Value  Decomposion， SVD 
样本数据矩阵 分解得到其奇异值矩阵
"""
# 测试
# 商品推荐
# import svdRec as svd    svd.recommend_test()
# 奇异值分解重构 图像压缩
# import svdRec as svd    svd.imgCompress()   svd.imgCompress(numSV=2) 
from numpy import *
from numpy import linalg as la
 
def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
 
#相似度 距离越近相似度越高，距离越远相似度越低
 
# 欧式距离计算的相似度            1~0之间  
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB)) # linalg.norm(计算2阶范数)
# 皮尔逊相关系数 距离 计算的相似度 1~0之间  
def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0  # corrcoef() 在 -1 ~ 1之间
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]
# 余弦相似度 距离  计算相似度
def cosSim(inA,inB):
    num = float(inA.T*inB)  # 向量 A*B
    denom = la.norm(inA)*la.norm(inB)# 向量A的模(2阶范数) * 向量B的模(2阶范数)
    return 0.5+0.5*(num/denom) # 向量A 与 向量B 的家教余弦 A*B / (||A||*||B||) 范围在-1~1
 
# 标准相似度方法 估计 商品评价
# 用户对各个物品评价矩阵间的相似性 
#          数据矩阵 用户编号  相似度计算方法  物品编号
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1] # 列维度 物品数量
    simTotal = 0.0; ratSimTotal = 0.0 # 变量初始化
    for j in range(n): # 对每一个物品
        userRating = dataMat[user,j]# 获得 该用户的 评价
        if userRating == 0: continue# 未评价的则 跳过
        # 哪些客户对两种物品都进行了评价 
        overLap = nonzero(logical_and(dataMat[:,item].A>0,\
                                      dataMat[:,j].A>0))[0]  #  选取的物品的评价 其他物品的评价
        if len(overLap) == 0: similarity = 0 # 若同时评价了的用户数量为零，则相似性为0
        else: similarity = simMeas(dataMat[overLap,item],\
                                   dataMat[overLap,j])# 计算 共同评价了的评价矩阵之间的 相似性
        print '物品 %d 和 %d 的相似性为: %f' % (item, j, similarity) # 两种物品间的相似性
        simTotal += similarity # 总的相似性
        ratSimTotal += similarity * userRating # 选取的物品和各个物品相似性乘上 用户对物品的 评价权重
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
 
# 基于 矩阵奇异值分解转换 的 商品评价估计
#       数据集矩阵  用户   相似性方法  物品标识  
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1] # 物品种类数据
    simTotal = 0.0; ratSimTotal = 0.0 # 相似性总和 变量初始化
    U,Sigma,VT = la.svd(dataMat) # 数据集矩阵 奇异值分解  返回的Sigma 仅为对角线上的值
    Sig4 = mat(eye(4)*Sigma[:4]) # 前四个已经包含了 90%的能力了，转化成对角矩阵
    # 计算能量 Sigma2 = Sigma**2  energy=sum(Sigma2)   energy90= energy*0.9   energy4 = sum(Sigma2[:4])
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  # 将数据转换到低维空间
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue # 跳过其他未评价的商品
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)# 计算 svd分解 转换过后矩阵的相似度
        print '物品 %d 和 %d 的相似性为: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
 
# 预测 用户评级较高的 前几种物品
#              数据集矩阵 用户 推荐物品数量  相似性函数  估计方法
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1]          # 为0的表示未评价的物品
    if len(unratedItems) == 0: return 'you rated everything' 
    itemScores = []
    for item in unratedItems:#在未评价的物品中
        estimatedScore = estMethod(dataMat, user, simMeas, item)#计算评价值
        itemScores.append((item, estimatedScore))#记录商品及对于评价值
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]# 逆序 排序 前N个预测评价最高的未评价的商品
 
# 测试
def recommend_test():
    mymat1 = mat(loadExData())
    mymat2 = mat(loadExData2())
    print "矩阵1:"
    print mymat1
    print '标准推荐方法：'
    print '余弦相似度，'
    print recommend(mymat1,2)
    print '欧式距离相似度，'
    print recommend(mymat1,2,simMeas=ecludSim)
    print '皮尔逊相似度，'
    print recommend(mymat1,2,simMeas=pearsSim)
    print '奇异值分解推荐方法：'
    print '余弦相似度，'
    print recommend(mymat1,2,estMethod=svdEst)
    print '欧式距离相似度，'
    print recommend(mymat1,2,simMeas=ecludSim,estMethod=svdEst)
    print '皮尔逊相似度，'
    print recommend(mymat1,2,simMeas=pearsSim,estMethod=svdEst)
    print "矩阵2:"
    print mymat2
    print '标准推荐方法：'
    print '余弦相似度，'
    print recommend(mymat2,2)
    print '欧式距离相似度，'
    print recommend(mymat2,2,simMeas=ecludSim)
    print '皮尔逊相似度，'
    print recommend(mymat2,2,simMeas=pearsSim)
    print '奇异值分解推荐方法：'
    print '余弦相似度，'
    print recommend(mymat2,2,estMethod=svdEst)
    print '欧式距离相似度，'
    print recommend(mymat2,2,simMeas=ecludSim,estMethod=svdEst)
    print '皮尔逊相似度，'
    print recommend(mymat2,2,simMeas=pearsSim,estMethod=svdEst)
 
#二值化打印数据
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print 1,
            else: print 0,
        print ''
 
# 利用SVD 对 手写字体 数据进行压缩
# 原始数据为 32*32 =1024
# 而利用奇异值分解 得到三个矩阵 利用2个奇异值就可以比较完整的 重构原来的矩阵
# 即奇异值分解压缩后 所用数据为 32*2 + 32*2 + 2 = 130
# 将近十倍的压缩率
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines(): # 每一行
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl) #原始数据集矩阵
    print "****原始矩阵******"
    printMat(myMat, thresh)#二值化打印数据
    print '大小：'
    print  shape(myMat)
    U,Sigma,VT = la.svd(myMat)# 奇异值分解
    SigRecon = mat(zeros((numSV, numSV)))# 简化的奇异值对角矩阵
    for k in range(numSV):               
        SigRecon[k,k] = Sigma[k]          # 替换对角上的元素
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:] 
    # 简化后的 奇异值对角矩阵 以及截断的 U, VT 逆变换得到重构后的矩阵
    print "****使用 %d 个奇异值重构后的矩阵******" % numSV
    printMat(reconMat, thresh)
    print '大小：'
print shape(reconMat)