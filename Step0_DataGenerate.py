import numpy as np
import torch

featuresetN = np.append(range(0,20),50)#'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'
featuresetN = np.append(featuresetN,range(20,40))# HMM
targetsetN = np.append(range(57,65),50)
ftsetN=np.append(featuresetN,targetsetN)

# ["A C D E F G H I K L M N P Q R S T V W Y X"]
ProtT5idx = torch.tensor([3, 22, 10,  9, 15,  5, 20, 12, 14,  4, 19, 17, 13, 16,  8,  7, 11,  6, 21, 18, 23])
ProtT5idx = ProtT5idx.unsqueeze(1).numpy()

Train = np.load(r'./DataSet/Train_HHblits.npz')
Traindata = Train['data']
Traindata = Traindata[:, :, ftsetN]
Traindata[:, :, 20] = Traindata[:, :, 20] - Traindata[:, :, 0:20].max(axis=2)
Traindata[:, :, -1] = abs(Traindata[:, :, -1]-1)
Traindata0=np.append(Traindata[:,:,0:21], np.zeros([Traindata.shape[0],1,21]), 1)
Traindatamask=Traindata0.sum(axis=2)
TraindataM = Train['data']
trainsum=Traindatamask[:,:,0].sum(axis=1)-1
trainsort=np.sort(trainsum)
trainidx=np.argsort(trainsum)
Traindata0=TraindataM[trainidx[:5000],:int(trainsort[4999]+1),:]
Traindata1=TraindataM[trainidx[5000:10000],:int(trainsort[9999]+1),:]
Traindata2=TraindataM[trainidx[10000:],:,:]
ff = r'./DataSet/Traindata0.npy'
np.save(ff,Traindata0)
ff = r'./DataSet/Traindata1.npy'
np.save(ff,Traindata1)
ff = r'./DataSet/Traindata2.npy'
np.save(ff,Traindata2)


Traindata = np.load(r'./DataSet/Traindata0.npy')
Traindata = Traindata[:, :, ftsetN]
Traindata[:, :, 20] = Traindata[:, :, 20] - Traindata[:, :, 0:20].max(axis=2)
Traindata[:, :, -1] = abs(Traindata[:, :, -1]-1)
Traindata0=np.append(Traindata[:,:,0:21], np.zeros([Traindata.shape[0],1,21]), 1)
Traindatamask=Traindata0.sum(axis=2)
Traindata0=np.dot(Traindata0,ProtT5idx)
Traindata1=np.append(Traindata[:,:,-1][:,:,np.newaxis], np.ones([Traindata.shape[0],1,1]), 1)
Traindata2=np.append(np.zeros([Traindata.shape[0],1,1]), Traindata[:,:,-1][:,:,np.newaxis], 1)
Traindata1=Traindata1-Traindata2
Traindata0=Traindata0+Traindata1
Traindatamask=Traindatamask[:,:,np.newaxis]+Traindata1
Traindata00 = np.append(Traindata0,Traindatamask,2)
ff = r'./ProtT5data\Traindata00.npy'
np.save(ff,Traindata00)

Traindata = np.load(r'./DataSet/Traindata1.npy')
Traindata = Traindata[:, :, ftsetN]
Traindata[:, :, 20] = Traindata[:, :, 20] - Traindata[:, :, 0:20].max(axis=2)
Traindata[:, :, -1] = abs(Traindata[:, :, -1]-1)
Traindata0=np.append(Traindata[:,:,0:21], np.zeros([Traindata.shape[0],1,21]), 1)
Traindatamask=Traindata0.sum(axis=2)
Traindata0=np.dot(Traindata0,ProtT5idx)
Traindata1=np.append(Traindata[:,:,-1][:,:,np.newaxis], np.ones([Traindata.shape[0],1,1]), 1)
Traindata2=np.append(np.zeros([Traindata.shape[0],1,1]), Traindata[:,:,-1][:,:,np.newaxis], 1)
Traindata1=Traindata1-Traindata2
Traindata0=Traindata0+Traindata1
Traindatamask=Traindatamask[:,:,np.newaxis]+Traindata1
Traindata10 = np.append(Traindata0,Traindatamask,2)
ff = r'./ProtT5data\Traindata10.npy'
np.save(ff,Traindata10)


Traindata = np.load(r'./DataSet/Traindata2.npy')
Traindata = Traindata[:, :, ftsetN]
Traindata[:, :, 20] = Traindata[:, :, 20] - Traindata[:, :, 0:20].max(axis=2)
Traindata[:, :, -1] = abs(Traindata[:, :, -1]-1)
Traindata0=np.append(Traindata[:,:,0:21], np.zeros([Traindata.shape[0],1,21]), 1)
Traindatamask=Traindata0.sum(axis=2)
Traindata0=np.dot(Traindata0,ProtT5idx)
Traindata1=np.append(Traindata[:,:,-1][:,:,np.newaxis], np.ones([Traindata.shape[0],1,1]), 1)
Traindata2=np.append(np.zeros([Traindata.shape[0],1,1]), Traindata[:,:,-1][:,:,np.newaxis], 1)
Traindata1=Traindata1-Traindata2
Traindata0=Traindata0+Traindata1
Traindatamask=Traindatamask[:,:,np.newaxis]+Traindata1
Traindata20 = np.append(Traindata0,Traindatamask,2)
ff = r'./ProtT5data\Traindata20.npy'
np.save(ff,Traindata20)


CB513 = np.load(r'./DataSet/CB513_HHblits.npz')
CB513Ndata = CB513['data']
CB513Ndata = CB513Ndata[:, :, ftsetN]
CB513Ndata[:, :, 20] = CB513Ndata[:, :, 20] - CB513Ndata[:, :, 0:20].max(axis=2)
CB513Ndata[:, :, -1] = abs(CB513Ndata[:, :, -1]-1)
CB513Ndata0=np.append(CB513Ndata[:,:,0:21], np.zeros([CB513Ndata.shape[0],1,21]), 1)
CB513Ndatamask=CB513Ndata0.sum(axis=2)
CB513Ndata0=np.dot(CB513Ndata0,ProtT5idx)
CB513Ndata1=np.append(CB513Ndata[:,:,-1][:,:,np.newaxis], np.ones([CB513Ndata.shape[0],1,1]), 1)
CB513Ndata2=np.append(np.zeros([CB513Ndata.shape[0],1,1]), CB513Ndata[:,:,-1][:,:,np.newaxis], 1)
CB513Ndata1=CB513Ndata1-CB513Ndata2
CB513Ndata0=CB513Ndata0+CB513Ndata1
CB513Ndatamask=CB513Ndatamask[:,:,np.newaxis]+CB513Ndata1
CB513Ndata0 = np.append(CB513Ndata0,CB513Ndatamask,2)
ff = r'./ProtT5data\CB513Ndata0.npy'
np.save(ff,CB513Ndata0)

TS115 = np.load(r'./DataSet/TS115_HHblits.npz')
TS115data = TS115['data']
TS115data = TS115data[:, :, ftsetN]
TS115data[:, :, 20] = TS115data[:, :, 20] - TS115data[:, :, 0:20].max(axis=2)
TS115data[:, :, -1] = abs(TS115data[:, :, -1]-1)
TS115data0=np.append(TS115data[:,:,0:21], np.zeros([TS115data.shape[0],1,21]), 1)
TS115datamask=TS115data0.sum(axis=2)
TS115data0=np.dot(TS115data0,ProtT5idx)
TS115data1=np.append(TS115data[:,:,-1][:,:,np.newaxis], np.ones([TS115data.shape[0],1,1]), 1)
TS115data2=np.append(np.zeros([TS115data.shape[0],1,1]), TS115data[:,:,-1][:,:,np.newaxis], 1)
TS115data1=TS115data1-TS115data2
TS115data0=TS115data0+TS115data1
TS115datamask=TS115datamask[:,:,np.newaxis]+TS115data1
TS115data0 = np.append(TS115data0,TS115datamask,2)
ff = r'./ProtT5data\TS115data0.npy'
np.save(ff,TS115data0)


CASP12 = np.load(r'./DataSet/CASP12_HHblits.npz')
CASP12data = CASP12['data']
CASP12data = CASP12data[:, :, ftsetN]
CASP12data[:, :, 20] = CASP12data[:, :, 20] - CASP12data[:, :, 0:20].max(axis=2)
CASP12data[:, :, -1] = abs(CASP12data[:, :, -1]-1)
CASP12data0=np.append(CASP12data[:,:,0:21], np.zeros([CASP12data.shape[0],1,21]), 1)
CASP12datamask=CASP12data0.sum(axis=2)
CASP12data0=np.dot(CASP12data0,ProtT5idx)
CASP12data1=np.append(CASP12data[:,:,-1][:,:,np.newaxis], np.ones([CASP12data.shape[0],1,1]), 1)
CASP12data2=np.append(np.zeros([CASP12data.shape[0],1,1]), CASP12data[:,:,-1][:,:,np.newaxis], 1)
CASP12data1=CASP12data1-CASP12data2
CASP12data0=CASP12data0+CASP12data1
CASP12datamask=CASP12datamask[:,:,np.newaxis]+CASP12data1
CASP12data0 = np.append(CASP12data0,CASP12datamask,2)
ff = r'./ProtT5data\CASP12data0.npy'
np.save(ff,CASP12data0)
