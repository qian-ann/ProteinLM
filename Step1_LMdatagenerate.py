from transformers import T5Tokenizer, T5Model
import torch
import numpy as np
USE_GPU=True
def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor

model = T5Model.from_pretrained("./prot_t5_xl_uniref50")



Traindata00 = np.load(r'./ProtT5data/Traindata00.npy')
Traindata01 = torch.LongTensor(Traindata00)
datalen = Traindata01.shape[0]
step = 2
Traindata0LM = np.zeros([0,Traindata01.shape[1],1024]).astype('float16')

for idx in range(int(np.ceil(datalen/step))):
    print(idx)
    input_ids=create_tensor(Traindata01[idx*step:(idx+1)*step,:,0])
    attention_mask=create_tensor(Traindata01[idx*step:(idx+1)*step,:,1])
    if USE_GPU:
        device = torch.device("cuda:0")
        model.to(device)
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=input_ids)
        # embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=None)
    # For feature extraction we recommend to use the encoder embedding
    encoder_embedding = embedding[2].cpu().numpy().astype('float16')
    Traindata0LM = np.append(Traindata0LM, encoder_embedding, 0)

ff = r'./ProtT5data/Traindata0LM.npy'
np.save(ff,Traindata0LM)


Traindata10 = np.load(r'./ProtT5data/Traindata10.npy')
Traindata11 = torch.LongTensor(Traindata10)
datalen = Traindata11.shape[0]
step = 7
Traindata1LM = np.zeros([0,Traindata11.shape[1],1024]).astype('float16')

for idx in range(int(np.ceil(datalen/step))):
    print(idx)
    input_ids=create_tensor(Traindata11[idx*step:(idx+1)*step,:,0])
    attention_mask=create_tensor(Traindata11[idx*step:(idx+1)*step,:,1])
    if USE_GPU:
        device = torch.device("cuda:0")
        model.to(device)
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=input_ids)
        # embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=None)
    # For feature extraction we recommend to use the encoder embedding
    encoder_embedding = embedding[2].cpu().numpy().astype('float16')
    Traindata1LM = np.append(Traindata1LM, encoder_embedding, 0)
Traindata1LM=Traindata1LM.astype('float16')
ff = r'./ProtT5data/Traindata1LM.npy'
np.save(ff,Traindata1LM)


Traindata20 = np.load(r'./ProtT5data/Traindata20.npy')
Traindata21 = torch.LongTensor(Traindata20)
datalen = Traindata21.shape[0]
step = 2
Traindata2LM = np.zeros([0,Traindata21.shape[1],1024]).astype('float16')

for idx in range(int(np.ceil(datalen/step))):
    print(idx)
    input_ids=create_tensor(Traindata21[idx*step:(idx+1)*step,:,0])
    attention_mask=create_tensor(Traindata21[idx*step:(idx+1)*step,:,1])
    if USE_GPU:
        device = torch.device("cuda:0")
        model.to(device)
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=input_ids)
        # embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=None)
    # For feature extraction we recommend to use the encoder embedding
    encoder_embedding = embedding[2].cpu().numpy().astype('float16')
    Traindata2LM = np.append(Traindata2LM, encoder_embedding, 0)

ff = r'./ProtT5data/Traindata2LM.npy'
np.save(ff,Traindata2LM)

CB513Ndata0 = np.load(r'./ProtT5data/CB513Ndata0.npy')
CB513Ndata1 = torch.LongTensor(CB513Ndata0)
datalen = CB513Ndata1.shape[0]
step = 2
CB513NdataLM = np.zeros([0,CB513Ndata1.shape[1],1024])

for idx in range(int(np.ceil(datalen/step))):
    print(idx)
    input_ids=create_tensor(CB513Ndata1[idx*step:(idx+1)*step,:,0])
    attention_mask=create_tensor(CB513Ndata1[idx*step:(idx+1)*step,:,1])
    if USE_GPU:
        device = torch.device("cuda:0")
        model.to(device)
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=input_ids)
        # embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=None)
    # For feature extraction we recommend to use the encoder embedding
    encoder_embedding = embedding[2].cpu().numpy()
    CB513NdataLM = np.append(CB513NdataLM, encoder_embedding, 0)
ff = r'.\CB513NdataLM.npy'
np.save(ff,CB513NdataLM)

TS115data0 = np.load(r'./ProtT5data/TS115data0.npy')
TS115data1 = torch.LongTensor(TS115data0)
datalen = TS115data1.shape[0]
step = 2
TS115dataLM = np.zeros([0,TS115data1.shape[1],1024])

for idx in range(int(np.ceil(datalen/step))):
    print(idx)
    input_ids=create_tensor(TS115data1[idx*step:(idx+1)*step,:,0])
    attention_mask=create_tensor(TS115data1[idx*step:(idx+1)*step,:,1])
    if USE_GPU:
        device = torch.device("cuda:0")
        model.to(device)
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=input_ids)
        # embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=None)
    # For feature extraction we recommend to use the encoder embedding
    encoder_embedding = embedding[2].cpu().numpy()
    TS115dataLM = np.append(TS115dataLM, encoder_embedding, 0)
ff = r'.\TS115dataLM.npy'
np.save(ff,TS115dataLM)

CASP12data0 = np.load(r'./ProtT5data/CASP12data0.npy')
CASP12data1 = torch.LongTensor(CASP12data0)
datalen = CASP12data1.shape[0]
step = 2
CASP12dataLM = np.zeros([0,CASP12data1.shape[1],1024])

for idx in range(int(np.ceil(datalen/step))):
    print(idx)
    input_ids=create_tensor(CASP12data1[idx*step:(idx+1)*step,:,0])
    attention_mask=create_tensor(CASP12data1[idx*step:(idx+1)*step,:,1])
    if USE_GPU:
        device = torch.device("cuda:0")
        model.to(device)
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=input_ids)
        # embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=None)
    # For feature extraction we recommend to use the encoder embedding
    encoder_embedding = embedding[2].cpu().numpy()
    CASP12dataLM = np.append(CASP12dataLM, encoder_embedding, 0)
ff = r'.\CASP12dataLM.npy'
np.save(ff,CASP12dataLM)
