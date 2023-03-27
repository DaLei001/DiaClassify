import numpy as np
import matplotlib.pyplot as plt
 
class SFA:  # slow feature analysis class
    def __init__(self):
        self._Z = []
        self._B = []
        self._eigenVector = []
 
    def getB(self, data):
        self._B = np.matrix(data.T.dot(data)) / (data.shape[0] - 1)
 
    def getZ(self, data):
        derivativeData = self.makeDiff(data)
        self._Z = np.matrix(derivativeData.T.dot(derivativeData)) / (derivativeData.shape[0] - 1)
 
    def makeDiff(self, data):
        diffData = np.mat(np.zeros((data.shape[0], data.shape[1])))
        for i in range(data.shape[1] - 1):
            diffData[:, i] = data[:, i] - data[:, i + 1]
        diffData[:, -1] = data[:, -1] - data[:, 0]
        return np.mat(diffData)
 
    def fit_transform(self, data, threshold=1e-7, conponents=-1):
        if conponents == -1:
            conponents = data.shape[0]
        self.getB(data)
        U, s, V = np.linalg.svd(self._B)
 
        count = len(s)
        for i in range(len(s)):
            if s[i] ** (0.5) < threshold:
                count = i
                break
        s = s[0:count]
        s = s ** 0.5
        S = (np.mat(np.diag(s))).I
        U = U[:, 0:count]
        whiten = S * U.T
        Z = (whiten * data.T).T
 
        self.getZ(Z)
        PT, O, P = np.linalg.svd(self._Z)
 
        self._eigenVector = P * whiten
        self._eigenVector = self._eigenVector[-1 * conponents:, :]
 
        return data.dot(self._eigenVector.T)
 
    def transfer(self, data):
        return data.dot(self._eigenVector.T)

m = 300
n = 100
x_train = np.random.randint(low=0,high=5,size=(m,n))
x_test = np.random.randint(low=0, high=5, size=(m,n))

# 首先定义SFA()
sfa = SFA()
# 接着利用训练集使用SFA提取慢特征，x_train是训练集的输入，conponents是要提取的前n个慢特征
trainDataS = sfa.fit_transform(x_train, conponents=25)
# transfer函数是利用SFA提取数据的慢特征
testDataS = sfa.transfer(x_test)

# for i in range(len(trainDataS[0])):
#     print(i)
#     print(trainDataS[:,i])
# print(trainDataS.shape)
# print(testDataS.shape)
# print(trainDataS)
# print(testDataS)

for i in range(25):
    print(np.mean(trainDataS[:,i]))
    print(np.std(trainDataS[:,i]))
# plt.show()