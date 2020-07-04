import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import numpy as np
import gc
import os
from torchvision import transforms
from PIL import Image, ImageFile
from math import trunc
import time
import csv

 
ImageFile.LOAD_TRUNCATED_IMAGES = True

#criterion = torch.nn.BCEWithLogitsLoss()
criterion = torch.nn.BCELoss()

test_number = 4
model_name = 'squeezenet1_1'
folder_path = './ISIC2020/dataset'
save_file = '../../Desktop/Results/PyTorch/' + model_name + '_ISIC2020/' + str(test_number)

if not os.path.isdir(save_file):
    os.mkdir(save_file)
    
if not os.path.isdir('./tmp'):
    os.mkdir('./tmp')
    
if not os.path.isdir('./models'):
    os.mkdir('./models')

INFO = 'Todo classifier mais as duas últimas da convolução (como na IC), lr: /9, /27.\nUsando BCELoss e LogSigmoid na última camada.'
if len(INFO) > 0:    
    f = open(save_file + '/INFO.txt', "w")
    f.write(INFO)
    f.close()   

im_size = 224
epochs = 30
#epochs_ft = 30
batch_size = 64
testing_with = 'validation'

print('Running ' + model_name + '...')
num_classes = len(os.listdir(folder_path + '/training'))
print('Found ' + str(num_classes) + ' classes.')

#torch.hub.list('pytorch/vision')  # Available models on github.

img_transforms = transforms.Compose([
    transforms.Resize((im_size,im_size)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225] )
    ])

def check_image(path):
    try:
        Image.open(path)
        return True
    except:
        return False

gc.collect()  # Summon the garbage collector

train_data_path = folder_path + "/training/"
train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=img_transforms, is_valid_file=check_image)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True)

#print(train_data.class_to_idx)

val_data_path = folder_path + "/validation/"
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=img_transforms, is_valid_file=check_image)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,shuffle=True)

# =============================================================================
# test_data_path = './ISIC2020/test'
# test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=img_transforms, is_valid_file=check_image)
# test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=True)
# =============================================================================

# Load structure.
transfer_model = models.squeezenet1_1(pretrained=True #,progress=True
                                      )
transfer_model.classifier[1] = nn.Conv2d(512, 8, kernel_size=(1,1), stride=(1,1))

# Loading weights.
transfer_model.load_state_dict(torch.load('./models/squeezenet1_1_dict_50epochs'))
print('Model weights loaded.')

# Freezing.
wt, bs = None, None
for name, param in transfer_model.named_parameters():
    if name == 'classifier.1.weight':
        wt = param
    elif name == 'classifier.1.bias':
        bs = param
    param.requires_grad = False
    
transfer_model.classifier = nn.Sequential(
    nn.Dropout(p=0.5, inplace=False),
    nn.Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1)),
    nn.ReLU(inplace=True),
    nn.Conv2d(8, 1, kernel_size=(1, 1), stride=(1, 1)),
    nn.LogSigmoid(),
    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    )

# Copying weights.
           
transfer_model.state_dict()['classifier.1.weight'].data.copy_(wt)
transfer_model.state_dict()['classifier.1.bias'].data.copy_(bs)

for name, param in transfer_model.named_parameters():
# =============================================================================
#     if name == 'classifier.1.weight' or name == 'classifier.1.bias':  # APENAS A ÚLTIMA CAMADA!
#         param.requires_grad = False
# =============================================================================
    if name.startswith('classifier'):  # TODO O BLOCO CLASSIFIER!
        param.requires_grad = True
        
# =============================================================================
# for name, param in transfer_model.named_parameters():
#     print(name + ' ' + str(param.requires_grad))
# =============================================================================

#print(transfer_model)

#torch.save(transfer_model, './tmp/' + model_name)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Using CUDA...')
else:
    device = torch.device("cpu")  


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=epochs, device="cpu"):
    for epoch in range(epochs):
        start = time.time()
        
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        print('batch ',end='')
        for batch in train_loader:
            print('#',end='')
            
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            #targets = targets.unsqueeze(1)  # Added
            loss = loss_fn(output, targets.float())
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)        
        training_loss /= len(train_loader.dataset)
        
        print('\nval ',end='')
        model.eval()
        num_correct = 0 
        num_examples = 0
        for batch in val_loader:
            print('#',end='')
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            #targets = targets.unsqueeze(1)  # Added
            loss = loss_fn(output,targets.float())
            #loss = loss_fn(output,targets.unsqueeze(1))
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], targets).view(-1)
            #correct = np.round(output.detach())
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)
        
        end = time.time()
        print('\nTime elapsed: {:.1f}'.format(end - start), end='s, ')
        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))

try:
    
    found_lr = 0.0038
       
    gc.collect() # Summon the garbage collector
    
    
    unfreeze_layers = [transfer_model.features[-1], transfer_model.features[-2]]
    for layer in unfreeze_layers:
        for param in layer.parameters():
            param.requires_grad = True
    
    #optimizer = optim.Adam(transfer_model.parameters(), lr=found_lr)
    optimizer = optim.Adam([
        { 'params': transfer_model.features[-1].parameters(), 'lr': found_lr /9},
        { 'params': transfer_model.features[-2].parameters(), 'lr': found_lr /27},
        ], lr=found_lr)
    
    # =============================================================================
    # transfer_model = torch.load('./tmp/' + model_name)
    # os.remove('./tmp/' + model_name)
    # =============================================================================
    transfer_model.to(device)
    
    print('Training...')
    # Training
    beg_train = time.time()
    
    train(transfer_model, optimizer, criterion, train_data_loader, val_data_loader, epochs=epochs, device=device)
    
    end_train = time.time()
    total_time = end_train - beg_train
       
    # Saving model
    #torch.save(transfer_model, './models/' + model_name + '_' + str(epochs + epochs_ft) + 'epochs')
    torch.save(transfer_model.state_dict(), './models/' + model_name + '_dict_ISIC2020_' + str(test_number))
    
    model = transfer_model
    num_correct = 0 
    num_examples = 0
    valid_loss = 0.0
    #loss_fn = torch.nn.BCEWithLogitsLoss()
    
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    if testing_with == 'validation':
        data_loader = val_data_loader
    # =============================================================================
    # elif testing_with == 'test':
    #     data_loader = test_data_loader
    # =============================================================================
    else:
        raise ValueError('Test dataset not defined or invalid!')
    
    # Results for validation dataset!
    print('Validation results... ')
    with torch.no_grad():
        for batch in data_loader:
          print('#')
          inputs, targets = batch
          inputs = inputs.to(device)
          output = model(inputs)
        
          targets = targets.to(device)
          #targets = targets.unsqueeze(1)  # Added
          loss = criterion(output,targets.float()) 
          valid_loss += loss.data.item() * inputs.size(0)
          
          #classes = torch.max(F.softmax(output), dim=1)[1]
          classes = torch.round(torch.sigmoid(output)).cpu().numpy()
          
          for c,t in zip(classes,targets):
            c = list(map(int, list(map(round, c))))
            confusion_matrix[t][c] = confusion_matrix[t][c]+1
        
          correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], targets).view(-1)
          num_correct += torch.sum(correct).item()
          num_examples += correct.shape[0]
        valid_loss /= len(data_loader.dataset)
        
        print(confusion_matrix)
              
    
    # Metrics
       
    # Saving confusion matrix to text file
    f = open(save_file + '/CM.txt', "a")
    f.write('Confusion matrix for ' + testing_with + ' dataset:\n\n[[')
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
    target_sizes = []
    soma = 0
    for folder in os.listdir(folder_path + '/' + testing_with):
        target_names.append(folder.capitalize())
        target_sizes.append(len(os.listdir(folder_path + '/' + testing_with + '/' + folder)))
        soma += target_sizes[-1]
        longer = max(longer, len(folder))    

    total = 0
    for row in confusion_matrix:
        total += sum(row)
    
    f.write('METRICS:\n')
    f.write(' '*longer + 3*' ')
    f.write('{:>12}'.format('Accuracy') + ' ')
    f.write('{:>12}'.format('Sensitivity') + '  ')
    f.write('{:>12}'.format('Specificity\n'))
    
    acc_avg = 0
    sens_avg = 0
    specf_avg = 0
    for i in range(len(confusion_matrix)):
        f.write(('{:<' + str(longer) + '}').format(target_names[i]) + '   ')
        TP = 0; TN = 0; FP = 0; FN = 0
        for j in range(len(confusion_matrix[i])):  # Within the images of the class.
            if i == j:
                TP += confusion_matrix[i][j]  # Incorrectly identified.
            else:
                FN += confusion_matrix[i][j]  # All the others.
        for j in range(len(confusion_matrix)):  # Within all identified as the class.
            if j != i:
                FP += confusion_matrix[j][i]  # Column of the class.
        TN = total - (TP + FN + FP)
        acc = float(TN + TP) / (TN + TP + FN + FP)
        sens = float(TP) / (TP + FN)
        specf = float(TN) / (TN + FP)
        acc_avg += target_sizes[i] * acc
        sens_avg += target_sizes[i] * sens
        specf_avg += target_sizes[i] * specf
        f.write('{:>12.3f}'.format(acc) + ' ')
        f.write('{:>12.3f}'.format(sens) + ' ')
        f.write('{:>12.3f}'.format(specf) + '\n')
    
    # For unbalanced datasets
    acc_avg /= soma
    sens_avg /= soma
    specf_avg /= soma
    
    f.write('\n\n=========================\n\nAccuracy avg:    ' + '{:.3f}'.format(acc_avg) + '\nSensitivity avg: ' +'{:.3f}'.format(sens_avg) + '\nSpecificity avg: ' + '{:.3f}'.format(specf_avg) + '\n')
    f.write('\nlr: ' + str(found_lr) + '\ntime: ' + '{:.1f}'.format(total_time/3600) + 'h\n')
    
    f.close()
    
    # CSV result for ISIC2020
    
    print('\nWriting CSV...', end='')
    #from torch.autograd import Variable
    
    def image_loader(image_name):
        image = Image.open(image_name)
        #image = img_transforms(image).float()
        image = img_transforms(image)
        #image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        return image.cuda()  #assumes that you're using GPU
    
    count = 0
    with open('submission' + str(test_number) +'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name','target'])
        cur_dir = './ISIC2020/test'
        for image in os.listdir(cur_dir):
            im = image_loader(cur_dir + '/' + image)
            output = torch.round(torch.sigmoid(transfer_model(im)))
            output = int(round(output.item()))
            #output = transfer_model(im).argmax()
            writer.writerow([image[:-4], output])
            count += 1
            if count % 100 == 0:
                print(count / 100, end=' ')
except:
    import traceback
    f = open(save_file + '/ERROR.txt', "w")
    f.write(traceback.format_exc())
    f.close()

os.system('shutdown -s -f -t 60')
