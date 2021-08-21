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


```python
test_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data", header=None)
test_df.head()

test_features = test_df.loc[:, 0:10]
test_labels = test_df.loc[:, 10]

print('Number of test examples:', test_features.shape[0])
```

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
        cost = CondorOrdinalCrossEntropy(logits, levels)
```



```python
from condor_pytorch.dataset import levels_from_labelbatch
from condor_pytorch.losses import CondorOrdinalCrossEntropy


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
        cost = cost = CondorOrdinalCrossEntropy(logits, levels)
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

## 4 -- Evaluate model

Finally, after model training, we can evaluate the performance of the model. For example, via the mean absolute error and mean squared error measures.

For this, we are going to use the `logits_to_label` utility function from `condor_pytorch` to convert the probabilities back to the orginal label.



```python
from condor_pytorch.dataset import logits_to_label


def compute_mae_and_mse(model, data_loader, device):

    with torch.no_grad():
    
        pwrong, mae, mse, acc, num_examples = 0., 0., 0., 0., 0

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.float().to(device)
            ids = targets.long()

            logits = model(features)
            predicted_labels = logits_to_label(logits).float()
            predicted_probs = ordinal_softmax(logits).float()

            num_examples += targets.size(0)
            mae += torch.sum(torch.abs(predicted_labels - targets))
            mse += torch.sum((predicted_labels - targets)**2)
            pwrong += torch.sum(1. - predicted_probs.gather(1,ids.view(-1,1)))

        mae = mae / num_examples
        mse = mse / num_examples
        pwrong = pwrong / num_examples
        return mae, mse, pwrong
```


```python
train_mae, train_mse, train_pwrong = compute_mae_and_mse(model, train_loader, DEVICE)
test_mae, test_mse, test_pwrong = compute_mae_and_mse(model, test_loader, DEVICE)
```


```python
print(f'Mean absolute error (train/test): {train_mae:.2f} | {test_mae:.2f}')
print(f'Mean squared error (train/test): {train_mse:.2f} | {test_mse:.2f}')
print(f'Mean probability on wrong class (train/test): {train_pwrong:.3f} | {test_pwrong:.3f}')
```
