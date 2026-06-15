from dataclasses import dataclass  , asdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , recall_score,precision_score ,root_mean_squared_error, confusion_matrix , mean_squared_error , r2_score ,accuracy_score,log_loss,f1_score
from typing import Literal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split ,cross_val_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder , StandardScaler ,LabelEncoder
import optuna
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset , DataLoader
import torch.optim as optim
import joblib
import os
import random


from nami import Torch #-----> new
from lion_pytorch import Lion #-----> new














def set_seed (seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    
set_seed(42)



import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader , TensorDataset
from matplotlib import pyplot as plt

device = torch.device('cuda'  if torch.cuda.is_available() else 'cpu')















transform_2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307 ,) , (0.3081 ,))
])


train_Dataset = torchvision.datasets.MNIST(root='./data' , train=True , download=True , transform=transform_2)
test_dataset = torchvision.datasets.MNIST(root='./data' , train=False , download=True ,transform=transform_2)


train_loader = DataLoader(train_Dataset , batch_size=32 , shuffle=True)
test_loader = DataLoader(test_dataset , batch_size=32 , shuffle=False)



class netModel (nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1 , out_channels=16 , kernel_size=3 , stride=1 , padding=1) #(1,28,28)-->(16,28,28)

        self.relu1 = Torch.Nami() # non Linear 

        self.pool1 = nn.MaxPool2d(kernel_size=2 , stride=2) # (16,28,28)-->28 / 2 ==> (16,14,14)

        self.conv2 = nn.Conv2d(in_channels=16 , out_channels=32 , kernel_size=3 , stride=1 , padding=1) #(16,14,14)-->(32,14,14)

        self.relu2 = Torch.Nami() # non Linear 

        self.pool2 = nn.MaxPool2d(kernel_size=2 , stride=2) #(32,14,14)-->14 / 2 ==> (32,7,7)

        self.layer1 = nn.Linear(in_features=32*7*7 , out_features=150)

        self.layer2 = nn.Linear(in_features=150 , out_features=120)

        self.layer3 = nn.Linear(in_features=120 , out_features=10)

        self.relu3 = Torch.Nami()


    
    def forward (self , x):

        x = self.pool1(self.relu1(self.conv1(x)))

        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(x.size(0) , -1)

        x = self.relu3(self.layer1(x))

        x = self.relu3(self.layer2(x))

        x = self.layer3(x)

        return x
    

lossis = []
accuracyis = []

total = 0
correct = 0

Y_ture = []
Y_prds = []

model = netModel().to(device)

criterion = nn.CrossEntropyLoss()


# optimizer = optim.AdamW(model.parameters() , lr=1e-4)
optimizer = Lion(model.parameters() , lr=1e-4 , weight_decay=0.01)#--> حداقل شروع مقدار وزن دهی ام 
# optimizer = optim.Adam(model.parameters() , lr=1e-4)    

epochs = 5
for i in range(epochs):

    model.train()

    runing_loss = 0.0

    for images , labels in train_loader :

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(images)
        
        loss = criterion(output , labels)

        loss.backward()

        optimizer.step()

        runing_loss += loss.item()

    avg_loss = runing_loss / len(train_loader)

    lossis.append(avg_loss)

    print(f'Epoch :{epochs} , Loss:{runing_loss / len(train_loader):.4f}')



    model.eval()

    with torch.no_grad():

        prds = model(images)

        _ , predict = torch.max(prds , 1)

        total += labels.size(0)

        correct += (predict == labels).sum().item()

        Y_ture.extend(labels.cpu().numpy())
        Y_prds.extend(predict.cpu().numpy())


    acc = 100 * correct / total

    accuracyis.append(acc)

    precision = precision_score(Y_ture , Y_prds ,average='macro')
    recall = recall_score(Y_ture , Y_prds , average='macro')
    confusion_matrixs = confusion_matrix(Y_ture , Y_prds)

    print(f'accuracy Test : {100 * correct / total:.2f}%')
    print('precision :' , precision)
    print('recall :' , recall)
    print(confusion_matrixs)





plt.figure(figsize=(12,5))
plt.subplot(1,2,1).plot(lossis , color='red')
plt.title('Loss Test')

plt.subplot(1,2,2).plot(accuracyis , color='blue')
plt.title('Accuracy Train')
plt.legend()
plt.show()



#_________________
   # with AdamW  + Nami   ----> ترکیب این دو بهتر از ترکیب Adam + Nami
# Epoch :5 , Loss:0.0492
# accuracy Test : 98.75%
# precision : 0.9873076923076922
# recall : 0.9881944444444445
# [[15  0  0  0  0  0  0  0  1  0]
#  [ 0 19  0  0  0  0  0  0  0  0]
#  [ 0  0 16  0  0  0  0  0  0  0]
#  [ 0  0  0 16  0  1  0  0  0  1]
#  [ 0  0  0  0 17  0  0  0  0  0]
#  [ 0  0  0  0  0 12  0  0  0  0]
#  [ 0  0  0  0  0  0 16  0  0  0]
#  [ 0  0  0  0  0  0  0 12  0  0]
#  [ 0  0  0  0  0  0  0  0 18  1]
#  [ 0  0  0  0  0  0  0  0  0 15]]
#_________________


#_________________
   # with Lion  -------> best methode
# Epoch :5 , Loss:0.0250
# accuracy Test : 99.38%
# precision : 0.9949999999999999
# recall : 0.99375
# [[16  0  0  0  0  0  0  0  0  0]
#  [ 0 19  0  0  0  0  0  0  0  0]
#  [ 0  0 16  0  0  0  0  0  0  0]
#  [ 0  0  0 18  0  0  0  0  0  0]
#  [ 0  0  0  0 17  0  0  0  0  0]
#  [ 0  0  0  0  0 12  0  0  0  0]
#  [ 0  1  0  0  0  0 15  0  0  0]
#  [ 0  0  0  0  0  0  0 12  0  0]
#  [ 0  0  0  0  0  0  0  0 19  0]
#  [ 0  0  0  0  0  0  0  0  0 15]]
        
#_________________

#_________________
   # with Adam  + Nami 
# Epoch :5 , Loss:0.0491
# accuracy Test : 98.12%
# precision : 0.9807945344129555
# recall : 0.9829312865497076
# [[15  0  0  0  0  0  0  0  1  0]
#  [ 0 19  0  0  0  0  0  0  0  0]
#  [ 0  0 16  0  0  0  0  0  0  0]
#  [ 0  0  0 17  0  1  0  0  0  0]
#  [ 0  0  0  0 17  0  0  0  0  0]
#  [ 0  0  0  0  0 12  0  0  0  0]
#  [ 0  0  0  0  0  0 16  0  0  0]
#  [ 0  0  0  0  0  0  0 12  0  0]
#  [ 0  0  0  0  0  0  0  0 18  1]
#  [ 0  0  0  0  0  0  0  0  0 15]]
#_________________


#_______________________________________
# With Lion + Nami
# Epoch :5 , Loss:0.0200
# accuracy Test : 98.75%
# precision : 0.99
# recall : 0.9884868421052632
# [[16  0  0  0  0  0  0  0  0  0]
#  [ 0 19  0  0  0  0  0  0  0  0]
#  [ 0  0 15  1  0  0  0  0  0  0]
#  [ 0  0  0 18  0  0  0  0  0  0]
#  [ 0  0  0  0 17  0  0  0  0  0]
#  [ 0  0  0  0  0 12  0  0  0  0]
#  [ 0  0  0  0  0  0 16  0  0  0]
#  [ 0  0  0  0  0  0  0 12  0  0]
#  [ 0  0  0  1  0  0  0  0 18  0]
#  [ 0  0  0  0  0  0  0  0  0 15]]
# اگر کمترین خطا اولویت باشه این ترکیب خیلی خوبه ولی اگر اولویت مون دقت تست بالا باشه اون وقت Lion بدون Nami بهتره.
#_______________________________________

# ترکیب Adam and AdamW + Nami بهترین نتایج ممکن رو دادند ولی به پای اون ترکیب Lion + Nami نمیرسند






import matplotlib.pyplot as plt
import numpy as np

# ==========================
# داده‌ها (بر اساس نتایج شما)
# ==========================
categories = ['Lion+ReLU', 'Lion+Nami', 'AdamW+Nami', 'Adam+Nami']
accuracy = [99.38, 98.75, 98.75, 98.12]
precision = [0.995, 0.990, 0.987, 0.981]
recall = [0.994, 0.988, 0.988, 0.983]
loss = [0.0250, 0.0200, 0.0492, 0.0491]

x = np.arange(len(categories))
width = 0.25

# استایل حرفه‌ای
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
fig.patch.set_facecolor('white')

# ========== نمودار چپ: Accuracy, Precision, Recall ==========
bars1 = ax1.bar(x - width, accuracy, width, label='Accuracy (%)', color='#2c7bb6', edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x, precision, width, label='Precision', color='#abd9e9', edgecolor='black', linewidth=0.5)
bars3 = ax1.bar(x + width, recall, width, label='Recall', color='#fdae61', edgecolor='black', linewidth=0.5)

ax1.set_xticks(x)
ax1.set_xticklabels(categories, rotation=15, ha='right', fontsize=10)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Comparison of Accuracy, Precision, and Recall', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# افزودن مقادیر روی میله‌ها (اختیاری)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=7)

# ========== نمودار راست: Loss ==========
bars4 = ax2.bar(x, loss, width*2, color='#d7191c', edgecolor='black', linewidth=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(categories, rotation=15, ha='right', fontsize=10)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Comparison of Loss (lower is better)', fontsize=13, fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars4:
    height = bar.get_height()
    ax2.annotate(f'{height:.4f}',
                 xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points",
                 ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('optimizer_benchmark.png', dpi=300, bbox_inches='tight')
plt.show()


