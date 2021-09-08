# CONDOR CNN for predicting handwritten digits (MNIST)

This tutorial explains how to equip a deep neural network with the CONDOR layer and loss function for ordinal regression. Please note that **MNIST is not an ordinal dataset**. The reason why we use MNIST in this tutorial is that it is included in the PyTorch's `torchvision` library and is thus easy to work with, since it doesn't require extra data downloading and preprocessing steps.

## 1 -- Setting up the dataset and dataloader

In this section, we set up the data set and data loaders. This is a general procedure that is not specific to CONDOR.


```python
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

##########################
### SETTINGS
##########################

# Hyperparameters
random_seed = 1
learning_rate = 0.05
num_epochs = 10
batch_size = 128

# Architecture
NUM_CLASSES = 10

# Other
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on', DEVICE)

##########################
### MNIST DATASET
##########################


# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.MNIST(root='data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          drop_last=True,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         drop_last=True,
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break
```

## 2 - Equipping CNN with CONDOR layer

In this section, we are using  `condor_pytorch` to outfit a convolutional neural network for ordinal regression. Note that the CONDOR method only requires replacing the last (output) layer, which is typically a fully-connected layer, by the CONDOR layer.

Using the `Sequential` API, we specify the CORAl layer as 

```python
        self.fc = torch.nn.Linear(size_in=294, num_classes=num_classes-1)
```

This is because the convolutional and pooling layers 

```python
            torch.nn.Conv2d(1, 3, (3, 3), (1, 1), 1),
            torch.nn.MaxPool2d((2, 2), (2, 2)),
            torch.nn.Conv2d(3, 6, (3, 3), (1, 1), 1),
            torch.nn.MaxPool2d((2, 2), (2, 2)))
```


produce a flattened feature vector of 294 units. Then, when using the CONDOR layer in the forward function

```python
        logits =  self.fc(x)
```

please use the `sigmoid` not softmax function (since the CONDOR method uses a concept known as extended binary classification as described in the paper).


```python
class ConvNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, (3, 3), (1, 1), 1),
            torch.nn.MaxPool2d((2, 2), (2, 2)),
            torch.nn.Conv2d(3, 6, (3, 3), (1, 1), 1),
            torch.nn.MaxPool2d((2, 2), (2, 2)))
        
        self.fc = torch.nn.Linear(294,num_classes-1) #THIS IS KEY OUTPUT SIZE 
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # flatten
        logits =  self.fc(x)
        
        return logits
    
    
    
torch.manual_seed(random_seed)
model = ConvNet(num_classes=NUM_CLASSES)
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
