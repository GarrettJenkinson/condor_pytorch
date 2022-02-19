# CONDOR MLP for predicting poker hands

This tutorial explains how to equip a deep neural network with the CONDOR layer and loss function for ordinal regression in the context of predicting poker hands.

## 0 -- Obtaining and preparing the Poker Hand dataset from the UCI ML repository

First, we are going to download and prepare the UCI Poker Hand dataset from [https://archive.ics.uci.edu/ml/datasets/Poker+Hand](https://archive.ics.uci.edu/ml/datasets/Poker+Hand) and save it as CSV files locally. This is a general procedure that is not specific to CONDOR.

This dataset has 10 ordinal labels, 

```
0: Nothing in hand; not a recognized poker hand 
1: One pair; one pair of equal ranks within five cards 
2: Two pairs; two pairs of equal ranks within five cards 
3: Three of a kind; three equal ranks within five cards 
4: Straight; five cards, sequentially ranked with no gaps 
5: Flush; five cards with the same suit 
6: Full house; pair + different rank three of a kind 
7: Four of a kind; four equal ranks within five cards 
8: Straight flush; straight + flush 
9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush 
```

where 0 < 1 < 2 ... < 9.

Download training examples and test dataset:


```python
import pandas as pd


train_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data", header=None)
train_features = train_df.loc[:, 0:10]
train_labels = train_df.loc[:, 10]

print('Number of features:', train_features.shape[1])
print('Number of training examples:', train_features.shape[0])
```

    Number of features: 11
    Number of training examples: 25010



```python
test_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data", header=None)
test_df.head()

test_features = test_df.loc[:, 0:10]
test_labels = test_df.loc[:, 10]

print('Number of test examples:', test_features.shape[0])
```

    Number of test examples: 1000000


Standardize features:


```python
from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
train_features_sc = sc.fit_transform(train_features)
test_features_sc = sc.transform(test_features)
```

Save training and test set as CSV files locally


```python
pd.DataFrame(train_features_sc).to_csv('train_features.csv', index=False)
train_labels.to_csv('train_labels.csv', index=False)

pd.DataFrame(test_features_sc).to_csv('test_features.csv', index=False)
test_labels.to_csv('test_labels.csv', index=False)

# don't need those anymore
del test_features
del train_features
del train_labels
del test_labels
```

## 1 -- Setting up the dataset and dataloader

In this section, we set up the data set and data loaders using PyTorch utilities. This is a general procedure that is not specific to CONDOR.


```python
import torch


##########################
### SETTINGS
##########################

# Hyperparameters
random_seed = 1
learning_rate = 0.001
num_epochs = 20
batch_size = 128

# Architecture
NUM_CLASSES = 10

# Other
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on', DEVICE)
```

    Training on cpu



```python
from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):

    def __init__(self, csv_path_features, csv_path_labels, dtype=np.float32):
    
        self.features = pd.read_csv(csv_path_features).values.astype(np.float32)
        self.labels = pd.read_csv(csv_path_labels).values.flatten()

    def __getitem__(self, index):
        inputs = self.features[index]
        label = self.labels[index]
        return inputs, label

    def __len__(self):
        return self.labels.shape[0]
```


```python
import torch
from torch.utils.data import DataLoader


# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = MyDataset('train_features.csv', 'train_labels.csv')
test_dataset = MyDataset('test_features.csv', 'test_labels.csv')


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True, # want to shuffle the dataset
                          num_workers=0) # number processes/CPUs to use

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True, # want to shuffle the dataset
                         num_workers=0) # number processes/CPUs to use

# Checking the dataset
for inputs, labels in train_loader:  
    print('Input batch dimensions:', inputs.shape)
    print('Input label dimensions:', labels.shape)
    break
```

    Input batch dimensions: torch.Size([128, 11])
    Input label dimensions: torch.Size([128])


## 2 - Equipping MLP with CONDOR layer

In this section, we are using  `condor_pytorch` to outfit a multilayer perceptron for ordinal regression. Note that the CONDOR method only requires replacing the last (output) layer, which is typically a fully-connected layer, by the CONDOR layer with one fewer node.



```python
class CondorMLP(torch.nn.Module):

    def __init__(self, num_classes):
        super(CondorMLP, self).__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Linear(11, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5,num_classes-1) #THIS IS KEY OUTPUT SIZE
        )


    def forward(self, x):
        logits = self.features(x)
        return logits

    
    
    
torch.manual_seed(random_seed)
model = CondorMLP(num_classes=NUM_CLASSES)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters())
```

## 3 - Using the CONDOR loss for model training

During training, all you need to do is to 

1) convert the integer class labels into the extended binary label format using the `levels_from_labelbatch` provided via `condor_pytorch`:

```python
        levels = levels_from_labelbatch(class_labels, 
                                        num_classes=NUM_CLASSES)
```

2) Apply the CONDOR loss (also provided via `condor_pytorch`):

```python
        cost = condor_negloglikeloss(logits, levels)
```



```python
from condor_pytorch.dataset import levels_from_labelbatch
from condor_pytorch.losses import condor_negloglikeloss


for epoch in range(num_epochs):
    
    model = model.train()
    for batch_idx, (features, class_labels) in enumerate(train_loader):

        ##### Convert class labels for CONDOR
        levels = levels_from_labelbatch(class_labels, 
                                        num_classes=NUM_CLASSES)
        ###--------------------------------------------------------------------###

        features = features.to(DEVICE)
        levels = levels.to(DEVICE)
        logits = model(features)
        
        #### CONDOR loss 
        cost = cost = condor_negloglikeloss(logits, levels)
        ###--------------------------------------------------------------------###   
        
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 200:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), cost))
```

    Epoch: 001/020 | Batch 000/196 | Cost: 1.1676
    Epoch: 002/020 | Batch 000/196 | Cost: 1.1595
    Epoch: 003/020 | Batch 000/196 | Cost: 0.5821
    Epoch: 004/020 | Batch 000/196 | Cost: 0.2216
    Epoch: 005/020 | Batch 000/196 | Cost: 0.1225
    Epoch: 006/020 | Batch 000/196 | Cost: 0.1025
    Epoch: 007/020 | Batch 000/196 | Cost: 0.0678
    Epoch: 008/020 | Batch 000/196 | Cost: 0.0501
    Epoch: 009/020 | Batch 000/196 | Cost: 0.0422
    Epoch: 010/020 | Batch 000/196 | Cost: 0.0219
    Epoch: 011/020 | Batch 000/196 | Cost: 0.0082
    Epoch: 012/020 | Batch 000/196 | Cost: 0.0158
    Epoch: 013/020 | Batch 000/196 | Cost: 0.0144
    Epoch: 014/020 | Batch 000/196 | Cost: 0.0032
    Epoch: 015/020 | Batch 000/196 | Cost: 0.0238
    Epoch: 016/020 | Batch 000/196 | Cost: 0.0434
    Epoch: 017/020 | Batch 000/196 | Cost: 0.0112
    Epoch: 018/020 | Batch 000/196 | Cost: 0.0018
    Epoch: 019/020 | Batch 000/196 | Cost: 0.0230
    Epoch: 020/020 | Batch 000/196 | Cost: 0.0011


## 4 -- Evaluate model

Finally, after model training, we can evaluate the performance of the model. For example, via the mean absolute error and mean squared error measures.

For this, we are going to use the `logits_to_label` utility function from `condor_pytorch` to convert the probabilities back to the orginal label.



```python
from condor_pytorch.dataset import logits_to_label
from condor_pytorch.activations import ordinal_softmax
from condor_pytorch.metrics import earth_movers_distance
from condor_pytorch.metrics import ordinal_accuracy
from condor_pytorch.metrics import mean_absolute_error

def compute_mae_and_acc(model, data_loader, device):

    with torch.no_grad():
    
        emd, mae, acc, acc1, num_examples = 0., 0., 0., 0., 0

        for i, (features, targets) in enumerate(data_loader):
            ##### Convert class labels for CONDOR
            levels = levels_from_labelbatch(targets, 
                                        num_classes=NUM_CLASSES)
            features = features.to(device)
            levels = levels.to(device)
            targets = targets.float().to(device)
            ids = targets.long()

            logits = model(features)
            predicted_labels = logits_to_label(logits).float()
            predicted_probs = ordinal_softmax(logits).float()

            num_examples += targets.size(0)
            mae  += mean_absolute_error(logits,levels,reduction='sum')
            acc  += ordinal_accuracy(logits,levels,tolerance=0,reduction='sum')
            acc1 += ordinal_accuracy(logits,levels,tolerance=1,reduction='sum')
            emd  += earth_movers_distance(logits,levels,reduction='sum')

        mae  = mae / num_examples
        acc  = acc / num_examples
        acc1 = acc1 / num_examples
        emd  = emd / num_examples
        return mae, acc, acc1, emd
```


```python
train_mae, train_acc, train_acc1, train_emd = compute_mae_and_acc(model, train_loader, DEVICE)
test_mae, test_acc, test_acc1, test_emd = compute_mae_and_acc(model, test_loader, DEVICE)
```


```python
print(f'Mean absolute error (train/test): {train_mae:.2f} | {test_mae:.2f}')
print(f'Accuracy tolerance 0 (train/test): {train_acc:.2f} | {test_acc:.2f}')
print(f'Accuracy tolerance 1 (train/test): {train_acc1:.2f} | {test_acc1:.2f}')
print(f'Earth movers distance (train/test): {train_emd:.3f} | {test_emd:.3f}')
```

    Mean absolute error (train/test): 0.00 | 0.00
    Accuracy tolerance 0 (train/test): 1.00 | 1.00
    Accuracy tolerance 1 (train/test): 1.00 | 1.00
    Earth movers distance (train/test): 0.005 | 0.004

