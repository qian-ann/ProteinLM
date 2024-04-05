# Fast and Accurate multi-task prediction of protein structural features by using a pretrained language model 

<div align="left">


</div>



## Datasets
Large files are supplied via google drive: 
https://drive.google.com/drive/folders/1NcerEtJUn6eULDLdu2l-WPdzvTTw6mFE?usp=sharing

In this google drive, folder "data" contains all the files ( 'Train_HHblits.npz', 'CASP12_HHblits.npz', 
'TS115_HHblits.npz', 'CB513_HHblits.npz') used for training and testing our models.

This files should be palced in `./data` folder.


## Protein T5

Dolowd `ProtT5-XL-UniRef50` from [ProtTrans](https://github.com/agemagician/ProtTrans)

##Usage 

1. Run `Step0_DataGenerate.py` to generate data.
2. Run `Step1_LMdatagenerate.py` to generate ProteinT5 data.
3. Run `Step2_DataLoad.py` to generate data loader.
4. Run `Step3_0_LM.py` to train and test.
