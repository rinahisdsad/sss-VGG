# -*- coding: utf-8 -*-
# @Time    : 2021/1/30
# @Author  : Li
# @FileName: semi_vgg.py
# @Software: PyCharm
import torch.nn as nn
import torch
from torch.nn import functional as F

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

import os

learning_rate = 0.0001

root = os.getcwd() + '/picture/'

def default_loader(path):
    image=Image.open(path).convert('RGB')
    image=image.resize((224,224),Image.ANTIALIAS)

    return image
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.imgs)
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2))])
train_data = MyDataset(txt=root + 'train.txt', transform=transform)
test_data = MyDataset(txt=root + 'train.txt', transform=transform)
nolabel_data = MyDataset(txt=root + 'nolabel.txt', transform=transform)
nolabel_loader=DataLoader(dataset=nolabel_data,batch_size=2,shuffle=True,num_workers=2)
train_loader=DataLoader(dataset=train_data,batch_size=2,shuffle=True,num_workers=2)
test_loader=DataLoader(dataset=test_data,batch_size=2,shuffle=True,num_workers=2)
valte_data = MyDataset(txt=root + 'test.txt', transform=transform)
valte_loader=DataLoader(dataset=valte_data,batch_size=2,shuffle=True,num_workers=2)

def Conv3x3BNReLU(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )
class VGGNet(nn.Module):
    def __init__(self, block_nums,num_classes=2):
        super(VGGNet, self).__init__()
        self.stage1 = self._make_layers(in_channels=3, out_channels=64, block_num=block_nums[0])
        self.stage2 = self._make_layers(in_channels=64, out_channels=128, block_num=block_nums[1])
        self.stage3 = self._make_layers(in_channels=128, out_channels=256, block_num=block_nums[2])
        self.stage4 = self._make_layers(in_channels=256, out_channels=512, block_num=block_nums[3])
        self.stage5 = self._make_layers(in_channels=512, out_channels=512, block_num=block_nums[4])
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7,out_features=4096),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=num_classes)
        )
    def _make_layers(self, in_channels, out_channels, block_num):
        layers = []
        layers.append(Conv3x3BNReLU(in_channels,out_channels))
        for i in range(1,block_num):
            layers.append(Conv3x3BNReLU(out_channels,out_channels))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = x.view(x.size(0),-1)
        out = self.classifier(x)
        return out
def VGG16():
    block_nums = [2, 2, 3, 3, 3]
    model = VGGNet(block_nums)
    return model

def VGG19():
    block_nums = [2, 2, 4, 4, 4]
    model = VGGNet(block_nums)
    return model

model = VGG16()
device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

a=[]
b=[]
c=[]
d=[]
e=[]
ff=[]
def tran(epoch):
    if epoch<=20:
        alpha = 0
    elif 20<epoch and epoch<80:
        alpha = 0.5 * (epoch-20)/(80-20)
    else:
        alpha = 0.5
    train_loss_label=0
    train_loss = 0
    laleb_index = 0
    index = 0
    print('epoch:' + str(epoch + 1))
    model.train()
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target = data
        inputs, target=inputs.to(device),target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,target)
        laleb_index = laleb_index + 1
        train_loss_label = train_loss_label + loss.item()
        loss.backward()
        optimizer.step()
    train_loss_label = train_loss_label/laleb_index
    for batch_idx, data in enumerate(nolabel_loader, 0):
        model.eval()
        index = index + 1
        images, _ = data
        images = images.to(device)
        outputs_image = model(images)
        _, predicted = torch.max(outputs_image.data, dim=1)
        model.train()
        predicted = predicted.to(device)
        optimizer.zero_grad()
        outputs_nolabel = model(images)
        loss_nolabel = criterion(outputs_nolabel, predicted)
        loss = train_loss_label + alpha * loss_nolabel
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()
    print('train_loss:',train_loss/index)
    a.append(loss.item())

def val():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images,label = data
            images, label = images.to(device), label.to(device)
            outputs = model(images)
            _,predicted = torch .max(outputs.data,dim=1)
            total += label.size(0)
            correct += (predicted == label).cpu().sum()
    print('train_accyracy:%.3f %%'%float(100*correct/total))
    b.append(float(100*correct/total))

def valtest():
    correct = 0
    total = 0
    val_loss = 0
    index = 0
    with torch.no_grad():
        for data in valte_loader:
            images,label = data
            images, label = images.to(device), label.to(device)
            outputs = model(images)
            valloss = criterion(outputs,label)
            index += 1
            val_loss = val_loss + valloss.item()
            _,predicted = torch .max(outputs.data,dim=1)
            total += label.size(0)

            correct += (predicted == label).cpu().sum()
    print('test_loss:',val_loss/index)
    c.append(val_loss/index)
    print('test_accuracy:%.3f %%'%float(100*correct/total))
    d.append(float(100*correct/total))
import xlwt


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style

def add_sheet(excel_obj, sheet_name):
    new_sheet = excel_obj.add_sheet(sheet_name, cell_overwrite_ok=True)
    return new_sheet
def add_sheet_row(sheet_obj, fieldnames, row_index=0):
    for i in range(len(fieldnames)):
        sheet_obj.write(row_index, i, fieldnames[i], set_style('Times New Roman', 220, True))
def add_sheet_col(sheet_obj, fieldnames, col_index=0, row_index=1):
    for i in range(len(fieldnames)):
        sheet_obj.write(row_index + i, col_index, fieldnames[i], set_style('Times New Roman', 220, True))
def write_excel():
    f = xlwt.Workbook()
    sheet1 = add_sheet(f, 'vgg16')
    row0 = ["train_loss", "train_accuracy", "test_loss", "test_accuracy","nolable_loss", "nolable_accuracy"]
    add_sheet_row(sheet1, row0)
    add_sheet_col(sheet1, a, 0, 1)
    add_sheet_col(sheet1, b, 1, 1)
    add_sheet_col(sheet1, c, 2, 1)
    add_sheet_col(sheet1, d, 3, 1)
    add_sheet_col(sheet1, e, 4, 1)
    add_sheet_col(sheet1, ff, 5, 1)
    f.save('./semi_vgg_a/semivggdata.xls')
if __name__ == '__main__':
    dir = './semi_vgg_a/'
    for epoch in range(100):
        tran(epoch)
        val()
        valtest()
        if (epoch+1)%20==0:
            torch.save(model.state_dict(), dir + str(epoch+1)+'net.pkl')
            write_excel()

