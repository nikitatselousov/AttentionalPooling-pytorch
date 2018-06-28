from __future__ import print_function, division
import torch
import torch.nn as nn
from datasets import CSDataSet
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
from sklearn.metrics import confusion_matrix
import os
import cv2
import time
import sys
# from model.residual_attention_network_pre import ResidualAttentionModel
# based https://github.com/liudaizong/Residual-Attention-Network
import model.resnet_att_pool

model_file = 'model_92_sgd.pkl'


# for test
def test(model, test_loader, btrain=False, model_file='model_92_sgd.pkl', epoch=-1):
    # Test
    if not btrain:
        model.load_state_dict(torch.load(model_file))
    model.eval()

    correct = 0
    total = 0
    #
    class_correct = list(0. for i in range(4))
    class_total = list(0. for i in range(4))
    per_patient = list([] for i in range(206))
    true_patient = list(-1 for i in range(206))
    conf_matrix = np.zeros((4,4))

    for images, labels, patients in test_loader:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        for i, p in enumerate(patients):
            per_patient[p].append(predicted[i])
            true_patient[p] = labels.cpu().data.numpy()[i]
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        c = (predicted == labels.data).squeeze().int()

        for i in range(len(labels.data)):
            label = labels.data[i]
            class_correct[label] += c[i]
            class_total[label] += 1

        
            
    print('Accuracy of the model on the test images: %d %%' % (100 * float(correct) / total))
    print('Accuracy of the model on the test images:', float(correct)/total)

    ptn_pred = list(-1 for i in range(206))
    emp = 0
    for i, p in enumerate(per_patient):
        if p:
            a = np.bincount(p)
            ptn_pred[i]=np.argmax(a)
        else:
            emp+=1
    ptn_pred = np.asarray(ptn_pred)
    true_patient = np.asarray(true_patient)    
    ptn_pred = ptn_pred[ptn_pred != -1]
    true_patient = true_patient[true_patient != -1]
    print(ptn_pred)
    print(true_patient)
    conf_matrix = conf_matrix + confusion_matrix(true_patient, ptn_pred)
    ptn_acc = 100* float((np.array(ptn_pred) == np.array(true_patient)).sum()) / (206-emp)


    with open("fold_"+str(sys.argv[1])+".txt", "a") as file:
        file.write('\nEpoch {0}\n'.format(epoch))
        file.write("Accuracy patient-wise: {0}%".format(ptn_acc))        
        print("Accuracy patient-wise: {0}%".format(ptn_acc))
        print('Accuracy of the model on the test images: {0}\n'.format((100 * float(correct) / total)))
        for i in range(4):
            file.write('Accuracy of {0} : {1}\n'.format(classes[i], 100 * float(class_correct[i]) / float(class_total[i])))
            print('Accuracy of {0} : {1}\n'.format(classes[i], 100 * float(class_correct[i]) / float(class_total[i])))
        print("Confusion matrix:\n")
        print(conf_matrix)
        file.write("Confusion matrix:\n")
        file.write(np.array2string(conf_matrix))
    return correct / total


# Image Preprocessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((32, 32), padding=4),   #left, top, right, bottom
    # transforms.Scale(224),
    transforms.Grayscale(),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

input_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

# when image is rgb, totensor do the division 255
# CIFAR-10 Dataset
#train_dataset = datasets.CIFAR10(root='./data/', train=True, transform=transform,download=True)

#test_dataset = datasets.CIFAR10(root='./data/', train=False, transform=test_transform)

train_dataset = CSDataSet("./data/fold_"+str(sys.argv[1])+"/", split="train",img_transform=input_transform, label_transform=None)
test_dataset = CSDataSet("./data/fold_"+str(sys.argv[1])+"/", split="test", img_transform=input_transform, label_transform=None)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=30,
                                          shuffle=False)

#classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes = ('one', 'two', 'three', 'four')

model = resnet18().cuda()
print(model)

lr = 0.1  # 0.1
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
is_train = True
is_pretrain = False
acc_best = 0
total_epoch = 50
if is_train is True:
    if is_pretrain == True:
        model.load_state_dict((torch.load(model_file)))
    # Training
    for epoch in range(total_epoch):
        model.train()
        tims = time.time()
        for i, (images, labels, _) in enumerate(train_loader):
            images = Variable(images.cuda())
            # print(images.data)
            labels = Variable(labels.cuda())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print("hello")
            if (i+1) % 25 == 0:
                print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" %(epoch+1, total_epoch, i+1, len(train_loader), loss.data[0]))
        print('the epoch takes time:',time.time()-tims)
        print('evaluate test set:')
        acc = test(model, test_loader, btrain=True, epoch=epoch)
        if acc > acc_best:
            acc_best = acc
            print('current best acc,', acc_best)
            torch.save(model.state_dict(), model_file)
        # Decaying Learning Rate
        if (epoch+1) / float(total_epoch) == 0.3 or (epoch+1) / float(total_epoch) == 0.6 or (epoch+1) / float(total_epoch) == 0.9:
            lr /= 10
            print('reset learning rate to:', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                print(param_group['lr'])
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
    # Save the Model
    torch.save(model.state_dict(), 'last_model_92_sgd.pkl')

else:
    test(model, test_loader, btrain=False)

