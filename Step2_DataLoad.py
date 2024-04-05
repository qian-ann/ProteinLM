

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

# 基本参数
seq_len = 700

featuresetN = range(0,50) # onehot20+HMM30
featuresetN = np.append(featuresetN,50)
targetsetN = np.append(np.array([51,56,65,66,67]),np.append(range(57,65),50),0)
ftsetN=np.append(featuresetN,targetsetN)

LMlen=1024

BATCH_SIZE = 64
BATCH_SIZE_TEST = 128


class NameDataset(Dataset):
    def __init__(self, is_train_set=0):

        if is_train_set == 6:
            CB513 = np.load(r'./DataSet/CB513_HHblits.npz')
            CB513data = CB513['data']
            CB513data = CB513data[:, :, ftsetN]
            CB513data[:, :, len(featuresetN)-1] = CB513data[:, :, len(featuresetN)-1] - CB513data[:, :, 0:20].max(axis=2)
            CB513data[:, :, -1] = abs(CB513data[:, :, -1]-1)
            data = np.load(r'./ProtT5data/CB513NdataLM.npy')
            data = np.append(data[:,:-1,:], CB513data,2)
        elif is_train_set == 7:
            TS115 = np.load(r'./DataSet/TS115_HHblits.npz')
            TS115data = TS115['data']
            TS115data = TS115data[:, :, ftsetN]
            TS115data[:, :, len(featuresetN)-1] = TS115data[:, :, len(featuresetN)-1] - TS115data[:, :, 0:20].max(axis=2)
            TS115data[:, :, -1] = abs(TS115data[:, :, -1]-1)
            data = np.load(r'./ProtT5data/TS115dataLM.npy')
            data = np.append(data[:,:-1,:], TS115data,2)
        elif is_train_set == 8:
            CASP12 = np.load(r'./DataSet/CASP12_HHblits.npz')
            CASP12data = CASP12['data']
            CASP12data = CASP12data[:, :, ftsetN]
            CASP12data[:, :, len(featuresetN)-1] = CASP12data[:, :, len(featuresetN)-1] - CASP12data[:, :, 0:20].max(axis=2)
            CASP12data[:, :, -1] = abs(CASP12data[:, :, -1]-1)
            data = np.load(r'./ProtT5data/CASP12dataLM.npy')
            data = np.append(data[:,:-1,:], CASP12data,2)
        elif is_train_set == 9:
            Traindata0 = np.load(r'./DataSet/Traindata0.npy')
            Traindata0 = Traindata0[:, :, ftsetN]
            Traindata0[:, :, len(featuresetN)-1] = Traindata0[:, :, len(featuresetN)-1] - Traindata0[:, :, 0:20].max(axis=2)
            Traindata0[:, :, -1] = abs(Traindata0[:, :, -1] - 1)
            data = np.load(r'./ProtT5data/Traindata0LM.npy')
            data = np.append(data[:,:-1,:], Traindata0,2)
        elif is_train_set == 10:
            Traindata0 = np.load(r'./DataSet/Traindata1.npy')
            Traindata0 = Traindata0[:, :, ftsetN]
            Traindata0[:, :, len(featuresetN)-1] = Traindata0[:, :, len(featuresetN)-1] - Traindata0[:, :, 0:20].max(axis=2)
            Traindata0[:, :, -1] = abs(Traindata0[:, :, -1] - 1)
            data = np.load(r'./ProtT5data/Traindata1LM.npy')
            data = np.append(data[:,:-1,:], Traindata0,2)
        elif is_train_set == 11:
            Traindata0 = np.load(r'./DataSet/Traindata2.npy')
            Traindata0 = Traindata0[:, :, ftsetN]
            Traindata0[:, :, len(featuresetN)-1] = Traindata0[:, :, len(featuresetN)-1] - Traindata0[:, :, 0:20].max(axis=2)
            Traindata0[:, :, -1] = abs(Traindata0[:, :, -1] - 1)
            data = np.load(r'./ProtT5data/Traindata2LM.npy')
            data = np.append(data[:,:-1,:], Traindata0,2)

        self.data = data
        self.len = len(self.data)

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float()

    def __len__(self):
        return self.len


trainset0 = NameDataset(is_train_set=9)
trainset1 = NameDataset(is_train_set=10)
trainset2 = NameDataset(is_train_set=11)
valiidx = np.arange(0, 11000, 25)
trainidx0 = np.setdiff1d(np.arange(0, 5000, 1), valiidx)
trainidx1 = np.setdiff1d(np.arange(0, 5000, 1), valiidx)
trainidx2 = np.setdiff1d(np.arange(0, 848, 1), valiidx)

trainloader0 = DataLoader(trainset0[trainidx0,:,:], batch_size=BATCH_SIZE, shuffle=True)
trainloader1 = DataLoader(trainset1[trainidx1,:,:], batch_size=BATCH_SIZE, shuffle=True)
trainloader2 = DataLoader(trainset2[trainidx2,:,:], batch_size=BATCH_SIZE, shuffle=True)

valiset = torch.cat([trainset0[np.arange(0, 5000, 25),:,:],
          torch.cat([torch.zeros(200,1632-207,len(ftsetN)+LMlen-1), torch.ones(200,1632-207,1)],2)],1)
valiset = torch.cat([valiset,torch.cat([trainset1[np.arange(0, 5000, 25),:,:],
          torch.cat([torch.zeros(200,1632-490,len(ftsetN)+LMlen-1),
                     torch.ones(200,1632-490,1)],2)],1), trainset2[np.arange(0, 848, 25),:,:]],0)
valiloader = DataLoader(valiset, batch_size=BATCH_SIZE_TEST, shuffle=False)

testset513N = NameDataset(is_train_set=6)
testloader513N = DataLoader(testset513N, batch_size=BATCH_SIZE_TEST, shuffle=False)
testsetTS115= NameDataset(is_train_set=7)
testloaderTS115 = DataLoader(testsetTS115, batch_size=BATCH_SIZE_TEST, shuffle=False)
testsetCASP12 = NameDataset(is_train_set=8)
testloaderCASP12 = DataLoader(testsetCASP12, batch_size=BATCH_SIZE_TEST, shuffle=False)

