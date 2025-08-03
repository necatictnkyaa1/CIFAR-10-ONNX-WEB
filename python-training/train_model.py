import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN,self).__init__()
        self.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.dropout1=nn.Dropout(0.25)
        self.dropout2=nn.Dropout(0.5)
        self.fc1=nn.Linear(128*4*4,512)
        self.fc2=nn.Linear(512,10)
        self.relu=nn.ReLU()
    
    def forward(self,x):
        x=self.pool(self.relu(self.conv1(x)))
        x=self.pool(self.relu(self.conv2(x)))
        x=self.pool(self.relu(self.conv3(x)))
        x=self.dropout1(x)
        x=x.view(-1,128*4*4)
        x=self.relu(self.fc1(x))
        x=self.dropout2(x)
        x=self.fc2(x)
        return x
    
def load_data():
    transform_train=transforms.Compose([
        transforms.RandomHorizontalFlip(),#resimleri yatay çeviriyor, sağa bakan kedi=sola bakan kedi
        transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))#bu degerde iyi calisir
    ])
    
    transform_test=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    trainset=torchvision.datasets.CIFAR10(root="../data",train=True,download=True,transform=transform_train)
    testset=torchvision.datasets.CIFAR10(root="../data",train=False,download=True,transform=transform_test)
    
    trainloader=DataLoader(trainset,batch_size=128,shuffle=True)
    testloader=DataLoader(testset,batch_size=128,shuffle=False)
    
    return testloader,trainloader


def train_model():
    
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    
    
    trainloader, testloader = load_data()
    
    
    model = CIFAR10CNN().to(device)
    
    
    criterion = nn.CrossEntropyLoss()
    
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
   
    num_epochs = 50
    train_losses = []     
    train_accuracies = []  
    
    print("Eğitim başlıyor...")
    
   
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            
            inputs, labels = inputs.to(device), labels.to(device)
            
           
            optimizer.zero_grad()
            
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()  
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)  # en yüksek skor alan classı alıyoruz
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            
            if i % 100 == 99:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], '
                      f'Loss: {running_loss/100:.4f}, '
                      f'Acc: {100*correct/total:.2f}%')
                running_loss = 0.0  
        
        
        
        
        epoch_acc = 100 * correct / total
        train_accuracies.append(epoch_acc)
        
    test_accuracy = evaluate_model(model, testloader, device)
    
    
    torch.save(model.state_dict(), '../models/cifar10_cnn.pth')
    print(f"Model kaydedildi! Test doğruluğu: {test_accuracy:.2f}%")
    
    return model

def evaluate_model(model, testloader, device):
    
    
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

if __name__ == "__main__":
    model = train_model()
