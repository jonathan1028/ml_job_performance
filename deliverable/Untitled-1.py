#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'deliverable'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # FS Consulting
# 
# ## Overview
# 
# 
# ---
#%% [markdown]
# ### Load Preprocessed Data
# ---

#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
import model_1
from torch import optim
from torch.utils.data.sampler import RandomSampler

# Test, test, test
# Read in preprocessed files
train_set = pd.read_csv('../data/Preprocessed_TrainingSet_1815.csv')
train_targets = pd.read_csv('../data/Preprocessed_TrainingTargets.csv')
validation_set = pd.read_csv('../data/Preprocessed_ValidationSet_1815.csv')
test_targets = pd.read_csv('../data/Preprocessed_TestingTargets.csv')

# Setup data loaders
train_set = torch.utils.data.TensorDataset(torch.Tensor(np.array(train_set)), torch.Tensor(np.array(train_targets)))
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle = True)

validation_set = torch.utils.data.TensorDataset(torch.Tensor(np.array(validation_set)), torch.Tensor(np.array(test_targets)))
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = 32, shuffle = True)

# Verify shape of tensors
dataiter = iter(train_loader)
features, targets = dataiter.next()
print("Training Set Tensors:")
print("  Features:",features.shape)
print("  Targets: ",targets.shape)
dataiter = iter(validation_loader)
features, targets = dataiter.next()
print("Test Set Tensors:")
print("  Features:",features.shape)
print("  Targets: ",targets.shape)

#%% [markdown]
# ### Load Model

#%%
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = model_1.JobPerformance(checkpoint['input_size'],checkpoint['output_size'],checkpoint['hidden_layers'],checkpoint['drop_out'])
    model.load_state_dict(checkpoint['state_dict'])
    criterion1 = checkpoint['criterion']
    optimizer = checkpoint['optimizer']
    learning_rate = checkpoint['learning_rate']
    
    return model, criterion1, optimizer, learning_rate

model, criterion1, optimizer, learning_rate = load_checkpoint('checkpoint_v1.pth')
criterion2 = torch.nn.MSELoss()
print("Our Model: \n",model,"\n")
print("State Dict Keys:\n",model.state_dict().keys(),"\n")
print("Criterion:",criterion1,"\n")
print("Optimizer:",optimizer,"\n")
# Set criterion & optimizer
# criterion = exec(checkpoint['criterion']
# optimizer = optim.Adam(model.parameters(), lr=0.003)

print("")

#%% [markdown]
# ### Create New Model
# 
# Pass in model configuration parameters.

#%%
# Instantiate model
learning_rate = 0.003
dropout_percentage = 0.0
model = model_1.JobPerformance(1815,1,[1054,512,256,64],dropout_percentage)
# criterion1 = torch.nn.SmoothL1Loss()
criterion1 = torch.nn.MSELoss()
criterion2 = torch.nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(model)

#%% [markdown]
# ### Train Model
# 

#%%
train_losses, validation_losses = model_1.train(model, train_loader, validation_loader, criterion1, criterion2, optimizer, epochs=100)

#%% [markdown]
# ### Plot Results 

#%%
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.legend(frameon=False)

#%% [markdown]
# ### Save Model

#%%
checkpoint = {
    'input_size': model.input_size,
    'output_size': model.output_size,
    'hidden_layers': [each.out_features for each in model.hidden_layers],
    'drop_out': model.dropout_percentage,
    'criterion': criterion1,
    'optimizer': optimizer,
    'learning_rate': learning_rate,
    'state_dict': model.state_dict()
}
print("State dict keys:", model.state_dict().keys())

torch.save(checkpoint, 'checkpoint_v1.pth')

#%% [markdown]
# ### Applying the Model

#%%
# Run new data through model and collect inferences
# Export to CSV
# Import new data & create empty DataFrame with shape[test_set.shape[0],1]
test_set = pd.read_csv('./data/1814_Preprocessed_TestSet.csv')
test_targets = pd.DataFrame(index=range(test_set.shape[0]),columns=range(1),dtype='float')

test_set = torch.utils.data.TensorDataset(torch.Tensor(np.array(test_set)), torch.Tensor(np.array(test_targets)))
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 32, shuffle = False)


outputs_tensor = model_1.predict(model, test_loader)


#%%
print(len(outputs_tensor))


#%%
df = pd.DataFrame(outputs.detach().numpy().round(2))
df.shape


#%%
raw_test_set = pd.read_csv('./data/TestSet.csv')
raw_test_set['job_performance'] = df.iloc[:,0]
raw_test_set
raw_test_set.to_csv(r'./predictions.csv', index=False)


#%%
raw_test_set.to_csv(r'./predictions.csv', index=False)

#%% [markdown]
# ### Best Results
#%% [markdown]
# ### Ideas
# - Try with replacement sampling
# - Try 90/10 split of data
# - Setup cross validation
# - Try different architectures
# - Try different loss functions
# - Try different learning rates
# - Try different set of variables
# 
# 
# 
# ### Minimum Objective MSE = 172,400
# ### Target Objective MSE = 52,500
# 
# ### Best MSE = 31,000
# - DataSet: _1815
# - Preprocessing: See notebook
# - Layers: [1815,1054,512,256,64,1]
# - Equations: [MSE,Adam,lr=0.003,do=0.0]
# - Epochs: 30
# 
# ### Best MSE = 35,000
# - Preprocessing: Made the mistake of stripping out one hot encoded data with variance function
# - Layers: [908,780,524,256,64,1]
# - Equations: [MSE,Adam,lr=0.003,do=0.0]
# - Epochs: 50

#%%



