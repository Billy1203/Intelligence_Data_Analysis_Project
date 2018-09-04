import numpy as numpy



def zeroMean(dataMat):
	meanVal = np.mean(dataMat, axis = 0) #按列求均值，求各个特征的均值
	newData = dataMat - meanVal
	return newData, meanVal

'''
def pca(dataMat, n):
	newData, meanVal = zeroMean(dataMat)
	covMat = np.cov(newData, rowvar = 0) #求协方差矩阵，return ndarray, if rowvar = 0, 一行代表一个样本，否则一列代表一个样本

	eigVals, eigVects = np.linalg.eig(np.mat(covMat)) #求特征值和特征向量，特征向量按列放，一列代表一个特征向量
	eigValIndice = np.argsort(eigVals) #对特征值从小到大排序
	n_eigValIndice = eigValIndice[-1:-(n+1):-1] #最大的n个特征值的下标
	n_eigVect = eigVects[:, n_eigValIndice] #最大的n个特征值对应的特征向量
	lowDDataMat = newData * n_eigVect #低维特征空间的数据
	reconMat = (lowDDataMat * n_eigVect.T) + meanVal #重构数据
	return lowDDataMat, reconMat
'''

def percentage2n(eigVals, percentage):
	sortArray = np.sort(eigVals)
	sorArray = sorArray[-1::-1]
	arraySum = sum(sortArray)
	tmpSum = 0
	num = 0
	for i in sortArray:
		tmpSum+=i
		num+=1
		if tmpSum >= arraySum * percentage:
			return num

def pca(dataMat, percentage = 0.80):
	newData, meanVal = zeroMean(dataMat)
	covMat = np.cov(newData, rowvar = 0) #求协方差矩阵，return ndarray, if rowvar = 0, 一行代表一个样本，否则一列代表一个样本

	eigVals, eigVects = np.linalg.eig(np.mat(covMat)) #求特征值和特征向量，特征向量按列放，一列代表一个特征向量
	n = percentage2n(eigVals, percentage)
	eigValIndice = np.argsort(eigVals) #对特征值从小到大排序
	n_eigValIndice = eigValIndice[-1:-(n+1):-1] #最大的n个特征值的下标
	n_eigVect = eigVects[:, n_eigValIndice] #最大的n个特征值对应的特征向量
	lowDDataMat = newData * n_eigVect #低维特征空间的数据
	reconMat = (lowDDataMat * n_eigVect.T) + meanVal #重构数据
	return lowDDataMat, reconMat