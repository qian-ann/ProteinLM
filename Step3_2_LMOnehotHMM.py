

import torch
import torch.nn as nn
import numpy as np
import time
import math
from torch.optim.lr_scheduler import StepLR
import torch.nn.utils.rnn as rnn_utils

# 基本参数
seq_len = 700

featuresetN = range(0,50) # onehot20+HMM30
featuresetN = np.append(featuresetN,50)
targetsetN = np.append(np.array([51,56,65,66,67]),np.append(range(57,65),50),0)
ftsetN=np.append(featuresetN,targetsetN)

LMlen=1024
INPUTS_SIZE = LMlen+len(featuresetN)-11
LABELS_NUM = 8

USE_GPU = True #
dropout_rateLRI = 0.3
dropout_rate_Conv = 0.3
dropout_rateRNN = 0.3
LAM = 0
learning_rate = 1e-3
rate_decay = 0.5
lamterm = learning_rate* rate_decay* rate_decay* rate_decay
kernel_size = 9
pad_len = int((kernel_size-1)/2)
out_channels = 256
LRI_SIZE = 512
LRI_SIZE1 = 256
HIDDEN_SIZE = 256
N_LAYER = 2
bidirectional = True

N_EPOCHS = 400

# Q8      DiS    ASA    Phi    Psi    Q3
# 77.74%, 87.94, 81.93, 66.95, 17.73, 27.47 # modelpara-LM-HMM-1.pth
# 75.06%, 86.87, 81.99, 50.51, 20.44, 30.80
# 77.60%, 87.37, 76.50, 66.24, 17.34, 27.57
# 71.29%, 82.69, 66.60, 67.71, 21.20, 35.79

canshu_bian = True #  False  #  #  True denotes donot load the parameter in modelpara.pth
dir = r'.\parameter\modelpara.pth'
if canshu_bian:
    start_epoch = 1
else:
    checkpoint = torch.load(dir)
    start_epoch = checkpoint['epoch'] + 1

BATCH_SIZE = 64
BATCH_SIZE_TEST = 128

# dropout
dropoutLRI = nn.Dropout(p=dropout_rateLRI)
dropoutRNN = nn.Dropout(p=dropout_rateRNN)
dropoutConv = nn.Dropout(p=dropout_rate_Conv)


class CNNLayer(nn.Module):
    def __init__(self):
        super(CNNLayer, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=pad_len),
            torch.nn.ReLU(),
        )
        self.layernorm = nn.LayerNorm(out_channels)
        self.ReLu = torch.nn.ReLU()

    def forward(self, x):
        # x （batch_size, seqlen, out_channels）
        x = self.layernorm(x + self.conv(x.permute(0, 2, 1)).permute(0, 2, 1))
        x = self.ReLu(x)
        if self.training:
            x = dropoutConv(x)
        return x

class RNNLayer(nn.Module):
    def __init__(self):
        super(RNNLayer, self).__init__()
        self.n_directions = 2 if bidirectional else 1
        # input: (batchSize,seqLen,input_size)   output: (batchSize,seqLen,hiddenSize*nDirections)
        self.rnn = nn.LSTM(out_channels, HIDDEN_SIZE, N_LAYER, batch_first=True, bidirectional=bidirectional, dropout=dropout_rateRNN)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(N_LAYER * self.n_directions, batch_size, HIDDEN_SIZE)
        return (create_tensor(hidden), create_tensor(hidden))

    def forward(self, x, data_len):
        batch_size = x.size(0)
        seq_len = x.size(1)
        output = rnn_utils.pack_padded_sequence(x, data_len, batch_first=True)
        output, (_, _) = self.rnn(output, self._init_hidden(batch_size))
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)

        output = torch.cat([output, abs(output[:, 0, :].unsqueeze(1).repeat(1, seq_len-output.shape[1], 1)*0)], 1)
        if self.training:
            output = dropoutRNN(output)
        output = torch.cat([x, output], 2)

        return output

class CRNNmodel(nn.Module):
    def __init__(self):
        super(CRNNmodel, self).__init__()

        self.n_directions = 2 if bidirectional else 1

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(INPUTS_SIZE, out_channels, kernel_size=1, stride=1),
            torch.nn.ReLU(),
        )

        self.layers = nn.ModuleList([CNNLayer() for _ in range(1)])
        self.RNNlayer = RNNLayer()

        self.FC = torch.nn.Sequential(
            torch.nn.Linear(out_channels + (HIDDEN_SIZE * self.n_directions), LRI_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(LRI_SIZE, LRI_SIZE1),
            torch.nn.ReLU(),
        )

        self.FC0 = torch.nn.Linear(LRI_SIZE1, LABELS_NUM)
        self.FC1 = torch.nn.Linear(LRI_SIZE1, 2)  # disordered
        self.FC2 = torch.nn.Linear(LRI_SIZE1, 1)  # ASA
        self.FC3 = torch.nn.Linear(LRI_SIZE1, 2)  # Phi
        self.FC4 = torch.nn.Linear(LRI_SIZE1, 2)  # Psi
        self.FC5 = torch.nn.Linear(LRI_SIZE1, 3)  # Q3

        self.layernorm0 = nn.LayerNorm(LABELS_NUM,elementwise_affine=False)
        self.layernorm1 = nn.LayerNorm(2,elementwise_affine=False)
        self.layernorm5 = nn.LayerNorm(3,elementwise_affine=False)

    def forward(self, x, data_len):
        batch_size = x.shape[0]
        # (batch_size, seqlen, out_channels)
        x = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        # (batch_size, seqlen, out_channels+ hidden_size * nDirections)
        x = self.RNNlayer(x, data_len)
        # (batch_size, seqlen, out_channels+ hidden_size * nDirections+INPUTS_SIZE)
        x = self.FC(x)
        if self.training:
            x = dropoutLRI(x)

        x0 = self.FC0(x)
        # x0 = self.layernorm0(x0)
        x1 = self.FC1(x)
        # x1 = self.layernorm1(x1)
        x2 = self.FC2(x)
        x2 = torch.sigmoid(x2)
        x3 = self.FC3(x)
        x3 = torch.tanh(x3)
        x4 = self.FC4(x)
        x4 = torch.tanh(x4)
        x5 = self.FC5(x)
        # x5 = self.layernorm5(x5)

        return x0, x1, x2, x3, x4, x5


def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor



def make_tensors(data):
    inputs = data[:, :, :(len(featuresetN)+LMlen-11)]
    UnknownSeq = data[:, :, -1]

    target = data[:, :, -len(targetsetN):]

    targ = target[:,:,-LABELS_NUM-1:].max(dim=2)[1]
    target1 = torch.cat([abs(1-target[:,:,0]).unsqueeze(-1), target[:,:,0].unsqueeze(-1)],-1)
    targ1 = target1.max(dim=2)[1]
    target5 = torch.cat([target[:,:,-LABELS_NUM-1:-LABELS_NUM+2].sum(axis=2,keepdims=True),
                       target[:,:,-LABELS_NUM+2:-LABELS_NUM+4].sum(axis=2,keepdims=True),
                       target[:,:,-LABELS_NUM+4:-LABELS_NUM+7].sum(axis=2,keepdims=True)],-1)
    targ5 = target5.max(dim=2)[1]
    targidx = ~np.isin(targ, np.array([8]))
    targidxR = ((targidx * (UnknownSeq.numpy()<0.5) * (targ1.numpy()>0.5))>0.5)

    data_len = targidx.sum(axis=1)
    data_idx = np.argsort(-data_len)
    data_len = data_len[data_idx]
    inputs = inputs[data_idx,:]
    targidx = targidx[data_idx,:]
    targidxR = targidxR[data_idx,:]
    target = target[data_idx,:]
    targ3 = target[:,:,2]
    targ4 = target[:,:,3]
    targ3 = targ3 * (~(targ3==360))
    targ4 = targ4 * (~(targ4==360))
    ASAmax=target[:,:,4]

    targN = [create_tensor(targ[data_idx,:]), create_tensor(targ1[data_idx,:]),
             create_tensor(target[:,:,1]), create_tensor(targ3),
             create_tensor(targ4), create_tensor(targ5[data_idx,:])]

    target3 = torch.cat([torch.sin(target[:, :, 2]*math.pi/180).unsqueeze(-1), torch.cos(target[:, :, 2]*math.pi/180).unsqueeze(-1)],-1)
    target4 = torch.cat([torch.sin(target[:, :, 3]*math.pi/180).unsqueeze(-1), torch.cos(target[:, :, 3]*math.pi/180).unsqueeze(-1)],-1)

    targetN = [create_tensor(target[:,:,-LABELS_NUM-1:-1]), create_tensor(target1[data_idx,:,:]),
               create_tensor(target[:,:,1]), create_tensor(target3),
               create_tensor(target4), create_tensor(target5[data_idx,:,:])]

    return create_tensor(inputs), targN, targidx, targetN, data_len, ASAmax, targidxR

def time_since(since):  # 计算程序运行的时间
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def trainModel(epoch):
    # training =True
    CRNNmodel.train()
    total_loss = 0
    lossrate = 1
    for i, data in enumerate(trainloader0, 1):
        # inputs (batch_size, seqlen, INPUTS_SIZE)
        # target (batch_size, seqlen, LABELS_NUM)
        inputs, _, targidx, targetN, data_len, _, targidxR = make_tensors(data)
        targidx = targidx.reshape(-1)
        targidxR = targidxR.reshape(-1)
        output = CRNNmodel(inputs, data_len)
        loss0 = criterion(output[0].reshape(-1, LABELS_NUM)[targidx,:],
                          targetN[0].reshape(-1, LABELS_NUM)[targidx,:]) * lossrate
        loss1 = criterion(output[1].reshape(-1, 2)[targidx,:], targetN[1].reshape(-1, 2)[targidx,:]) * lossrate
        loss2 = criterionMSE(output[2].reshape(-1, 1)[targidxR,:], targetN[2].reshape(-1, 1)[targidxR,:]) * lossrate
        loss3 = criterionMSE(output[3].reshape(-1, 2)[targidxR,:], targetN[3].reshape(-1, 2)[targidxR,:]) * lossrate
        loss4 = criterionMSE(output[4].reshape(-1, 2)[targidxR,:], targetN[4].reshape(-1, 2)[targidxR,:]) * lossrate
        loss5 = criterion(output[5].reshape(-1, 3)[targidx,:], targetN[5].reshape(-1, 3)[targidx,:]) * lossrate
        loss = loss0+5*loss1+20*loss2+10*loss3+5*loss4+2*loss5
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(inputs)}/{len(trainset0)+len(trainset1)+len(trainset2)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
    i0=i+1
    for i, data in enumerate(trainloader1, i0):
        # inputs (batch_size, seqlen, INPUTS_SIZE)
        # target (batch_size, seqlen, LABELS_NUM)
        inputs, _, targidx, targetN, data_len, _, targidxR = make_tensors(data)
        targidx = targidx.reshape(-1)
        targidxR = targidxR.reshape(-1)
        output = CRNNmodel(inputs, data_len)
        loss0 = criterion(output[0].reshape(-1, LABELS_NUM)[targidx,:],
                          targetN[0].reshape(-1, LABELS_NUM)[targidx,:]) * lossrate
        loss1 = criterion(output[1].reshape(-1, 2)[targidx,:], targetN[1].reshape(-1, 2)[targidx,:]) * lossrate
        loss2 = criterionMSE(output[2].reshape(-1, 1)[targidxR,:], targetN[2].reshape(-1, 1)[targidxR,:]) * lossrate
        loss3 = criterionMSE(output[3].reshape(-1, 2)[targidxR,:], targetN[3].reshape(-1, 2)[targidxR,:]) * lossrate
        loss4 = criterionMSE(output[4].reshape(-1, 2)[targidxR,:], targetN[4].reshape(-1, 2)[targidxR,:]) * lossrate
        loss5 = criterion(output[5].reshape(-1, 3)[targidx,:], targetN[5].reshape(-1, 3)[targidx,:]) * lossrate
        loss = loss0+5*loss1+20*loss2+10*loss3+5*loss4+2*loss5
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(inputs)}/{len(trainset0)+len(trainset1)+len(trainset2)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
    i0=i+1
    for i, data in enumerate(trainloader2, i0):
        # inputs (batch_size, seqlen, INPUTS_SIZE)
        # target (batch_size, seqlen, LABELS_NUM)
        inputs, _, targidx, targetN, data_len, _, targidxR = make_tensors(data)
        targidx = targidx.reshape(-1)
        targidxR = targidxR.reshape(-1)
        output = CRNNmodel(inputs, data_len)
        loss0 = criterion(output[0].reshape(-1, LABELS_NUM)[targidx,:],
                          targetN[0].reshape(-1, LABELS_NUM)[targidx,:]) * lossrate
        loss1 = criterion(output[1].reshape(-1, 2)[targidx,:], targetN[1].reshape(-1, 2)[targidx,:]) * lossrate
        loss2 = criterionMSE(output[2].reshape(-1, 1)[targidxR,:], targetN[2].reshape(-1, 1)[targidxR,:]) * lossrate
        loss3 = criterionMSE(output[3].reshape(-1, 2)[targidxR,:], targetN[3].reshape(-1, 2)[targidxR,:]) * lossrate
        loss4 = criterionMSE(output[4].reshape(-1, 2)[targidxR,:], targetN[4].reshape(-1, 2)[targidxR,:]) * lossrate
        loss5 = criterion(output[5].reshape(-1, 3)[targidx,:], targetN[5].reshape(-1, 3)[targidx,:]) * lossrate
        loss = loss0+5*loss1+20*loss2+10*loss3+5*loss4+2*loss5
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(inputs)}/{len(trainset0)+len(trainset1)+len(trainset2)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
    torch.cuda.empty_cache()
    return total_loss


def valiModel():
    # training =False
    CRNNmodel.eval()
    correct0 = 0
    TP1 = 0
    TN1 = 0
    FP1 = 0
    FN1 = 0
    pred2 = torch.zeros([0])
    targ2 = torch.zeros([0])
    pred3 = torch.zeros([0])
    targ3 = torch.zeros([0])
    pred4 = torch.zeros([0])
    targ4 = torch.zeros([0])
    correct5 = 0
    total = 0
    print("validating trained model ...")
    with torch.no_grad():
        for i, data in enumerate(valiloader, 1):
            # inputs (batch_size, seqlen, INPUTS_SIZE)
            # target (batch_size, seqlen, LABELS_NUM)
            inputs, targN, targidx, _, data_len, _, targidxR = make_tensors(data)
            output= CRNNmodel(inputs, data_len)
            output0 = output[0].reshape(-1, LABELS_NUM)
            pred0 = output0.max(dim=1)[1][targidx.reshape(-1)].cpu()
            targ0 = targN[0].reshape(-1).cpu()[targidx.reshape(-1)]
            correct0 += pred0.eq(targ0.view_as(pred0)).sum().item()

            output1 = output[1].reshape(-1, 2)
            pred1 = output1.max(dim=1)[1][targidx.reshape(-1)].cpu()
            targ1 = targN[1].reshape(-1).cpu()[targidx.reshape(-1)]
            TP1 = TP1 + pred1[targ1==1].eq(targ1[targ1==1]).sum().item()
            FP1 = FP1+ (targ1==1).sum().item() - pred1[targ1==1].eq(targ1[targ1==1]).sum().item()
            TN1 = TN1 + pred1[targ1==0].eq(targ1[targ1==0]).sum().item()
            FN1 = FN1 + (targ1==0).sum().item() - pred1[targ1==0].eq(targ1[targ1==0]).sum().item()

            pred2 = torch.cat([pred2, output[2].cpu().reshape(-1)[targidxR.reshape(-1)]], 0)
            targ2 = torch.cat([targ2, targN[2].cpu().reshape(-1)[targidxR.reshape(-1)]], 0)

            output3 = output[3].cpu().numpy()
            pred3 = torch.cat([pred3, torch.tensor(np.arctan2(output3[:, :, 0], output3[:, :, 1])).reshape(-1)\
                    [targidxR.reshape(-1)] *180/math.pi], 0)
            targ3 = torch.cat([targ3, targN[3].cpu().reshape(-1)[targidxR.reshape(-1)]])

            output4 = output[4].cpu().numpy()
            pred4 = torch.cat([pred4, torch.tensor(np.arctan2(output4[:, :, 0], output4[:, :, 1])).reshape(-1)\
                    [targidxR.reshape(-1)] * 180 / math.pi], 0)
            targ4 = torch.cat([targ4, targN[4].cpu().reshape(-1)[targidxR.reshape(-1)]])

            output5 = output[5].reshape(-1, 3)
            pred5 = output5.max(dim=1)[1][targidx.reshape(-1)].cpu()
            targ5 = targN[5].reshape(-1).cpu()[targidx.reshape(-1)]
            correct5 += pred5.eq(targ5.view_as(pred5)).sum().item()

            total += len(targ0)
        MCC1 = '%.2f' % (
                (TP1 * TN1 - FP1 * FN1) / (np.sqrt((TP1 + FP1) * (TP1 + FN1) * (TN1 + FP1) * (TN1 + FN1)) + 10e-3)*100)
        pearson2 = '%.2f' % (100 * PCC(pred2 - pred2.mean(dim=-1, keepdim=True), targ2 - targ2.mean(dim=-1, keepdim=True)))
        MAE3 = '%.2f' % MAE(pred3, targ3)
        MAE4 = '%.2f' % MAE(pred4, targ4)
        percent0 = '%.2f' % (100 * correct0 / total)
        percent5 = '%.2f' % (100 * correct5 / total)
        print(f'Validation set: Accuracy {total} {percent0}%, {percent5}, {pearson2}, {MCC1}, {MAE3}, {MAE4}')
        torch.cuda.empty_cache()
    return float(percent0)


def testModel513N():
    # training =False
    CRNNmodel.eval()
    correct0 = 0
    TP1 = 0
    TN1 = 0
    FP1 = 0
    FN1 = 0
    pred2 = torch.zeros([0])
    targ2 = torch.zeros([0])
    pred3 = torch.zeros([0])
    targ3 = torch.zeros([0])
    pred4 = torch.zeros([0])
    targ4 = torch.zeros([0])
    correct5 = 0
    total = 0

    print("evaluating trained model ...")
    with torch.no_grad():
        for i, data in enumerate(testloader513N):
            # inputs (batch_size, seqlen, INPUTS_SIZE)
            # target (batch_size, seqlen, LABELS_NUM)
            inputs, targN, targidx, _, data_len, _, targidxR = make_tensors(data)
            output = CRNNmodel(inputs, data_len)
            output0 = output[0].reshape(-1, LABELS_NUM)
            pred0 = output0.max(dim=1)[1][targidx.reshape(-1)].cpu()
            targ0 = targN[0].reshape(-1).cpu()[targidx.reshape(-1)]
            correct0 += pred0.eq(targ0.view_as(pred0)).sum().item()

            output1 = output[1].reshape(-1, 2)
            pred1 = output1.max(dim=1)[1][targidx.reshape(-1)].cpu()
            targ1 = targN[1].reshape(-1).cpu()[targidx.reshape(-1)]
            TP1 = TP1 + pred1[targ1 == 1].eq(targ1[targ1 == 1]).sum().item()
            FP1 = FP1 + (targ1 == 1).sum().item() - pred1[targ1 == 1].eq(targ1[targ1 == 1]).sum().item()
            TN1 = TN1 + pred1[targ1 == 0].eq(targ1[targ1 == 0]).sum().item()
            FN1 = FN1 + (targ1 == 0).sum().item() - pred1[targ1 == 0].eq(targ1[targ1 == 0]).sum().item()

            pred2 = torch.cat([pred2, output[2].cpu().reshape(-1)[targidxR.reshape(-1)]], 0)
            targ2 = torch.cat([targ2, targN[2].cpu().reshape(-1)[targidxR.reshape(-1)]], 0)

            output3 = output[3].cpu().numpy()
            pred3 = torch.cat([pred3, torch.tensor(np.arctan2(output3[:, :, 0], output3[:, :, 1])).reshape(-1) \
                [targidxR.reshape(-1)] * 180 / math.pi], 0)
            targ3 = torch.cat([targ3, targN[3].cpu().reshape(-1)[targidxR.reshape(-1)]])

            output4 = output[4].cpu().numpy()
            pred4 = torch.cat([pred4, torch.tensor(np.arctan2(output4[:, :, 0], output4[:, :, 1])).reshape(-1) \
                [targidxR.reshape(-1)] * 180 / math.pi], 0)
            targ4 = torch.cat([targ4, targN[4].cpu().reshape(-1)[targidxR.reshape(-1)]])

            output5 = output[5].reshape(-1, 3)
            pred5 = output5.max(dim=1)[1][targidx.reshape(-1)].cpu()
            targ5 = targN[5].reshape(-1).cpu()[targidx.reshape(-1)]
            correct5 += pred5.eq(targ5.view_as(pred5)).sum().item()

            total += len(targ0)
        MCC1 = '%.2f' % (
                (TP1 * TN1 - FP1 * FN1) / (np.sqrt((TP1 + FP1) * (TP1 + FN1) * (TN1 + FP1) * (TN1 + FN1)) + 10e-3)*100)
        pearson2 = '%.2f' % (100 * PCC(pred2 - pred2.mean(dim=-1, keepdim=True), targ2 - targ2.mean(dim=-1, keepdim=True)))
        MAE3 = '%.2f' % MAE(pred3, targ3)
        MAE4 = '%.2f' % MAE(pred4, targ4)
        percent0 = '%.2f' % (100 * correct0 / total)
        percent5 = '%.2f' % (100 * correct5 / total)
        print(f'CB513 set: Accuracy {total} {percent0}%, {percent5}, {pearson2}, {MCC1}, {MAE3}, {MAE4}')
        torch.cuda.empty_cache()
    return correct0 / total

def testModelTS115():
    # training =False
    CRNNmodel.eval()
    correct0 = 0
    TP1 = 0
    TN1 = 0
    FP1 = 0
    FN1 = 0
    pred2 = torch.zeros([0])
    targ2 = torch.zeros([0])
    pred3 = torch.zeros([0])
    targ3 = torch.zeros([0])
    pred4 = torch.zeros([0])
    targ4 = torch.zeros([0])
    correct5 = 0
    total = 0

    print("evaluating trained model ...")
    with torch.no_grad():
        for i, data in enumerate(testloaderTS115):
            # 根据需求丢给GPU
            # inputs (batch_size, seqlen, INPUTS_SIZE)
            # target (batch_size, seqlen, LABELS_NUM)
            inputs, targN, targidx, _, data_len, _, targidxR = make_tensors(data)
            output = CRNNmodel(inputs, data_len)
            output0 = output[0].reshape(-1, LABELS_NUM)
            pred0 = output0.max(dim=1)[1][targidx.reshape(-1)].cpu()
            targ0 = targN[0].reshape(-1).cpu()[targidx.reshape(-1)]
            correct0 += pred0.eq(targ0.view_as(pred0)).sum().item()

            output1 = output[1].reshape(-1, 2)
            pred1 = output1.max(dim=1)[1][targidx.reshape(-1)].cpu()
            targ1 = targN[1].reshape(-1).cpu()[targidx.reshape(-1)]
            TP1 = TP1 + pred1[targ1 == 1].eq(targ1[targ1 == 1]).sum().item()
            FP1 = FP1 + (targ1 == 1).sum().item() - pred1[targ1 == 1].eq(targ1[targ1 == 1]).sum().item()
            TN1 = TN1 + pred1[targ1 == 0].eq(targ1[targ1 == 0]).sum().item()
            FN1 = FN1 + (targ1 == 0).sum().item() - pred1[targ1 == 0].eq(targ1[targ1 == 0]).sum().item()

            pred2 = torch.cat([pred2, output[2].cpu().reshape(-1)[targidxR.reshape(-1)]], 0)
            targ2 = torch.cat([targ2, targN[2].cpu().reshape(-1)[targidxR.reshape(-1)]], 0)

            output3 = output[3].cpu().numpy()
            pred3 = torch.cat([pred3, torch.tensor(np.arctan2(output3[:, :, 0], output3[:, :, 1])).reshape(-1) \
                [targidxR.reshape(-1)] * 180 / math.pi], 0)
            targ3 = torch.cat([targ3, targN[3].cpu().reshape(-1)[targidxR.reshape(-1)]])

            output4 = output[4].cpu().numpy()
            pred4 = torch.cat([pred4, torch.tensor(np.arctan2(output4[:, :, 0], output4[:, :, 1])).reshape(-1) \
                [targidxR.reshape(-1)] * 180 / math.pi], 0)
            targ4 = torch.cat([targ4, targN[4].cpu().reshape(-1)[targidxR.reshape(-1)]])

            output5 = output[5].reshape(-1, 3)
            pred5 = output5.max(dim=1)[1][targidx.reshape(-1)].cpu()
            targ5 = targN[5].reshape(-1).cpu()[targidx.reshape(-1)]
            correct5 += pred5.eq(targ5.view_as(pred5)).sum().item()

            total += len(targ0)
        MCC1 = '%.2f' % (
                (TP1 * TN1 - FP1 * FN1) / (np.sqrt((TP1 + FP1) * (TP1 + FN1) * (TN1 + FP1) * (TN1 + FN1)) + 10e-3)*100)
        pearson2 = '%.2f' % (100 * PCC(pred2 - pred2.mean(dim=-1, keepdim=True), targ2 - targ2.mean(dim=-1, keepdim=True)))
        MAE3 = '%.2f' % MAE(pred3, targ3)
        MAE4 = '%.2f' % MAE(pred4, targ4)
        percent0 = '%.2f' % (100 * correct0 / total)
        percent5 = '%.2f' % (100 * correct5 / total)
        print(f'TS115 set: Accuracy {total} {percent0}%, {percent5}, {pearson2}, {MCC1}, {MAE3}, {MAE4}')
        torch.cuda.empty_cache()
    return correct0 / total

def testModelCASP12():
    # training =False
    CRNNmodel.eval()
    correct0 = 0
    TP1 = 0
    TN1 = 0
    FP1 = 0
    FN1 = 0
    pred2 = torch.zeros([0])
    targ2 = torch.zeros([0])
    pred3 = torch.zeros([0])
    targ3 = torch.zeros([0])
    pred4 = torch.zeros([0])
    targ4 = torch.zeros([0])
    correct5 = 0
    total = 0

    print("evaluating trained model ...")
    with torch.no_grad():
        for i, data in enumerate(testloaderCASP12):
            # inputs (batch_size, seqlen, INPUTS_SIZE)
            # target (batch_size, seqlen, LABELS_NUM)
            inputs, targN, targidx, _, data_len, _, targidxR = make_tensors(data)
            output = CRNNmodel(inputs, data_len)
            output0 = output[0].reshape(-1, LABELS_NUM)
            pred0 = output0.max(dim=1)[1][targidx.reshape(-1)].cpu()
            targ0 = targN[0].reshape(-1).cpu()[targidx.reshape(-1)]
            correct0 += pred0.eq(targ0.view_as(pred0)).sum().item()

            output1 = output[1].reshape(-1, 2)
            pred1 = output1.max(dim=1)[1][targidx.reshape(-1)].cpu()
            targ1 = targN[1].reshape(-1).cpu()[targidx.reshape(-1)]
            TP1 = TP1 + pred1[targ1 == 1].eq(targ1[targ1 == 1]).sum().item()
            FP1 = FP1 + (targ1 == 1).sum().item() - pred1[targ1 == 1].eq(targ1[targ1 == 1]).sum().item()
            TN1 = TN1 + pred1[targ1 == 0].eq(targ1[targ1 == 0]).sum().item()
            FN1 = FN1 + (targ1 == 0).sum().item() - pred1[targ1 == 0].eq(targ1[targ1 == 0]).sum().item()

            pred2 = torch.cat([pred2, output[2].cpu().reshape(-1)[targidxR.reshape(-1)]], 0)
            targ2 = torch.cat([targ2, targN[2].cpu().reshape(-1)[targidxR.reshape(-1)]], 0)

            output3 = output[3].cpu().numpy()
            pred3 = torch.cat([pred3, torch.tensor(np.arctan2(output3[:, :, 0], output3[:, :, 1])).reshape(-1) \
                [targidxR.reshape(-1)] * 180 / math.pi], 0)
            targ3 = torch.cat([targ3, targN[3].cpu().reshape(-1)[targidxR.reshape(-1)]])

            output4 = output[4].cpu().numpy()
            pred4 = torch.cat([pred4, torch.tensor(np.arctan2(output4[:, :, 0], output4[:, :, 1])).reshape(-1) \
                [targidxR.reshape(-1)] * 180 / math.pi], 0)
            targ4 = torch.cat([targ4, targN[4].cpu().reshape(-1)[targidxR.reshape(-1)]])

            output5 = output[5].reshape(-1, 3)
            pred5 = output5.max(dim=1)[1][targidx.reshape(-1)].cpu()
            targ5 = targN[5].reshape(-1).cpu()[targidx.reshape(-1)]
            correct5 += pred5.eq(targ5.view_as(pred5)).sum().item()

            total += len(targ0)
        MCC1 = '%.2f' % (
                (TP1 * TN1 - FP1 * FN1) / (np.sqrt((TP1 + FP1) * (TP1 + FN1) * (TN1 + FP1) * (TN1 + FN1)) + 10e-3)*100)
        pearson2 = '%.2f' % (100 * PCC(pred2 - pred2.mean(dim=-1, keepdim=True), targ2 - targ2.mean(dim=-1, keepdim=True)))
        MAE3 = '%.2f' % MAE(pred3, targ3)
        MAE4 = '%.2f' % MAE(pred4, targ4)
        percent0 = '%.2f' % (100 * correct0 / total)
        percent5 = '%.2f' % (100 * correct5 / total)
        print(f'CASP12 set: Accuracy {total} {percent0}%, {percent5}, {pearson2}, {MCC1}, {MAE3}, {MAE4}')
        torch.cuda.empty_cache()
    return correct0 / total


CRNNmodel = CRNNmodel()
# 加载
if canshu_bian:
    start_epoch = 1
else:
    CRNNmodel.load_state_dict(checkpoint['net'])
    minacc = checkpoint['minacc']

criterion = torch.nn.CrossEntropyLoss()
criterionMSE = torch.nn.MSELoss()
PCC = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
MAE = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(CRNNmodel.parameters(), betas=[0.9,0.99], lr=learning_rate, weight_decay=LAM)
scheduler = StepLR(optimizer, step_size=1, gamma=rate_decay)

start = time.time()


if USE_GPU:
    device = torch.device("cuda:0")
    CRNNmodel.to(device)
    criterion.to(device)
    criterionMSE.to(device)

print("Training for %d epochs..." % N_EPOCHS)
acc_list = []
acc_list = list(acc_list)
valiacc_list = []
valiacc_list = list(valiacc_list)
for epoch in range(start_epoch, N_EPOCHS + 1):
    # Train cycle
    trainModel(epoch)
    valiacc = valiModel()
    valiacc_list.append(valiacc)
    if epoch == 1: minacc = valiacc
    if np.array(valiacc_list)[-1] < minacc:
        # break
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler.step()
        checkpoint = torch.load(dir)
        CRNNmodel.load_state_dict(checkpoint['net'])
    else:
        minacc = np.array(valiacc_list)[-1]
        state = {'net': CRNNmodel.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'minacc': minacc}
        torch.save(state, dir)
    if optimizer.param_groups[0]['lr'] < lamterm:
        break

# dir=r'.\parameter\modelpara-LM-HMM-1.pth'
checkpoint = torch.load(dir)
CRNNmodel.load_state_dict(checkpoint['net'])
acc4 = testModel513N()
acc5 = testModelTS115()
acc6 = testModelCASP12()