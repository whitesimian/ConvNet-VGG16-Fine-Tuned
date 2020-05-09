import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFile
import gc
import os

# =============================================================================
# Format of data folders:
#
# ./training
#     .../class1
#     .../class2
#     .
#     .
#     .
# ./validation
#     .../class1
#     .../class2
#     .
#     .
#     .
# ./test
#     .../class1
#     .../class2
#     .
#     .
#     .
# =============================================================================
 
ImageFile.LOAD_TRUNCATED_IMAGES = True

folder_path = r'C:\Users\Pichau\PycharmProjects\manage_files\global_patterns'
save_file = '../../Desktop/Results/ResNet50/'
model_name = 'resnet50'

im_size = 224
epochs = 20  # Epochs before fine-tuning.
epochs_ft = 50  # Epochs for fine-tuning.
batch_size = 32
testing_with = 'validation'

num_classes = len(os.listdir(folder_path + r'\training'))
print('Found ' + str(num_classes) + ' classes.')

#torch.hub.list('pytorch/vision')  # Modelos disponíveis no github

train_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(p=0.5), # used for data augmentation
    # transforms.RandomVerticalFlip(p=0.5), # used for data augmentation
    transforms.Resize((im_size,im_size)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225] )                            
])

img_transforms = transforms.Compose([
    transforms.Resize((im_size,im_size)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225] )
    ])

def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False

gc.collect()  # Summon the garbage collector

train_data_path = folder_path + "/training/"
train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=img_transforms, is_valid_file=check_image)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True)

val_data_path = folder_path + "/validation/"
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=img_transforms, is_valid_file=check_image)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,shuffle=True)

test_data_path = folder_path + "/test/"
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=img_transforms, is_valid_file=check_image)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=True)

transfer_model = models.resnet50(pretrained=True)
    
# print(transfer_model)

# Freezing the convolutional layers, except batch normalizations.
for name, param in transfer_model.named_parameters():
    if("bn" not in name):
        param.requires_grad = False

transfer_model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features,500), # Other models supplied with PyTorch use either 'fc' or 'classifier'. You can
                                                                                # ...also use 'out_features' to discover the activations coming out.
nn.ReLU(),                                 
nn.Dropout(), nn.Linear(500,num_classes))

torch.save(transfer_model, './tmp/' + model_name)
    
if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")  

# =============================================================================
# Remember that because the next function does actually train the model and messes with
# the optimizer’s learning rate settings, you should save and reload your model beforehand
# to get back to the state it was in before you called find_lr() and also
# reinitialize the optimizer you’ve chosen, which you can do now, passing in the learning
# rate you’ve determined from looking at the graph!
# =============================================================================

# Heuristic to gather the data of loss values through a given learning rate range.
def find_lr(model, loss_fn, optimizer, train_loader, init_value=1e-8, final_value=10.0, device="cpu"):
    number_in_epoch = len(train_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    for data in train_loader:
        batch_num += 1
        
        # Progress
        if batch_num%100==0:
            print('Finding lr... nof data: ' + str(batch_num))
            
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Crash out if loss explodes
        if batch_num > 1 and loss > 4 * best_loss:
            if(len(log_lrs) > 20):
                return log_lrs[10:-5], losses[10:-5]
            else:
                return log_lrs, losses

        # Record the best loss
        if loss < best_loss or batch_num == 1:
            best_loss = loss

        # Store the values
        losses.append(loss.item())
        log_lrs.append((lr))

        # Do the backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update the lr for the next step and store
        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr
    return log_lrs[10:-5], losses[10:-5]

optimizer = optim.Adam(transfer_model.parameters())

logs,losses = find_lr(transfer_model, torch.nn.CrossEntropyLoss(), optimizer, train_data_loader)
plt.plot(logs, losses)  # Plotting learning rate graph.
plt.savefig(save_file + 'lr.png')

# Finding best learning rate using the heuristics above.
best = 0
found_lr = None
for i in range(len(logs)):
    if i != 0:
        cur = (losses[i-1] - losses[i]) / (logs[i] - logs[i-1])
        if cur > best and logs[i-1] > 1e-4:  # Minimum 1e-4.
            best = cur
            found_lr = logs[i-1]
print('Reportedly optimal learning rate: ' + str(found_lr))

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=epochs, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        
        model.eval()
        num_correct = 0 
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets) 
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))
        
        
gc.collect() # Summon the garbage collector

# Reload
transfer_model = torch.load('./tmp/' + model_name)
transfer_model.to(device)

if found_lr == None:
    raise ValueError('Error finding learning rate!')

optimizer = optim.Adam(transfer_model.parameters(), lr=found_lr)

# Training
train(transfer_model, optimizer,torch.nn.CrossEntropyLoss(), train_data_loader, val_data_loader, epochs=epochs, device=device)

# Unfreezing some convolutional layers for fine-tuning.
unfreeze_layers = [transfer_model.layer3, transfer_model.layer4]
for layer in unfreeze_layers:
    for param in layer.parameters():
            param.requires_grad = True

optimizer = optim.Adam([
        { 'params': transfer_model.layer4.parameters(), 'lr': found_lr /3},
        { 'params': transfer_model.layer3.parameters(), 'lr': found_lr /9},
        ], lr=found_lr)

# Fine-tuning
train(transfer_model, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, val_data_loader, epochs=epochs_ft, device=device)
   
# Saving model
torch.save(transfer_model, './models/' + model_name + '_' + str(epochs + epochs_ft) + 'epochs')
torch.save(transfer_model.state_dict(), './models/' + model_name + '_dict_' + str(epochs + epochs_ft) + 'epochs')
 

model = transfer_model
num_correct = 0 
num_examples = 0
valid_loss = 0.0
loss_fn = torch.nn.CrossEntropyLoss()

confusion_matrix = np.zeros((num_classes, num_classes))

if testing_with == 'validation':
    data_loader = val_data_loader
elif testing_with == 'test':
    data_loader = test_data_loader
else:
    raise ValueError('Test dataset not defined or invalid!')

# Results for validation dataset!
for batch in data_loader:
  inputs, targets = batch
  inputs = inputs.to(device)
  output = model(inputs)

  targets = targets.to(device)
  loss = loss_fn(output,targets) 
  valid_loss += loss.data.item() * inputs.size(0)
  
  classes = torch.max(F.softmax(output), dim=1)[1]
  
  for c,t in zip(classes,targets):
    confusion_matrix[t][c] = confusion_matrix[t][c]+1

  correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], targets).view(-1)
  num_correct += torch.sum(correct).item()
  num_examples += correct.shape[0]
valid_loss /= len(test_data_loader.dataset)

print(confusion_matrix)

from math import trunc           

# Metrics
   
# Saving confusion matrix to text file
f = open(save_file + '/CM.txt', "a")
f.write('Confusion matrix for ' + testing_with + ' dataset:\n[[')
first = True
for row in confusion_matrix:
    if not first:
        f.write('\n [')
    else:
        first = False
    for number in row:
        f.write("{: >4}".format(trunc(number)))
    f.write(']')
f.write(']\n\n') 
            
target_names = []; longer = 0
for folder in os.listdir(folder_path + r'/training'):
    target_names.append(folder.capitalize())
    longer = max(longer, len(folder))

total = 0
for row in confusion_matrix:
    total += sum(row)

f.write('METRICS:\n')
f.write(' '*longer + 3*' ')
f.write('{:>12}'.format('Accuracy') + ' ')
f.write('{:>12}'.format('Sensitivity') + ' ')
f.write('{:>12}'.format(' Specificity\n'))

acc_avg = 0
sens_avg = 0
specf_avg = 0
for i in range(len(confusion_matrix)):
    f.write(('{:<' + str(longer) + '}').format(target_names[i]) + '   ')
    TP = 0; TN = 0; FP = 0; FN = 0
    for j in range(len(confusion_matrix[i])):
        if i==j:
            TP += confusion_matrix[i][j]
        else:
            FP += confusion_matrix[i][j]
    for j in range(len(confusion_matrix)):
        if j!=i:
            FN += confusion_matrix[i][j]
    TN = total - confusion_matrix[i][i] - FN
    acc = float(TN+TP)/(TN+TP+FN+FP)
    sens = float(TP)/(TP+FN)
    specf = float(TN)/(TN+FP)
    acc_avg += acc
    sens_avg += sens
    specf_avg += specf
    f.write('{:>12.4f}'.format(acc) + ' ')
    f.write('{:>12.4f}'.format(sens) + ' ')
    f.write('{:>12.4f}'.format(specf) + '\n')

# For balanced datasets.
acc_avg /= num_classes
sens_avg /= num_classes
specf_avg /= num_classes

f.write('\n\n===============\nAccuracy avg:    ' + '{:.4f}'.format(acc_avg) + '\nSensitivity avg: ' +'{:.4f}'.format(sens_avg) + '\nSpecificity avg: ' + '{:.4f}'.format(specf_avg) + '\n')

f.close()


