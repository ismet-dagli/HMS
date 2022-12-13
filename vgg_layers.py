import torch
from datetime import datetime
# from torchsummary import summary
total_time=0
layer_execs={}


class TheModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Get a resnet50 backbone
        self.layer1 =  torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer2 =  torch.nn.ReLU()
        self.layer3 =  torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer4 =  torch.nn.ReLU()
        self.layer5 =  torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer6 =  torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer7 =  torch.nn.ReLU()
        self.layer8 =  torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer9 =  torch.nn.ReLU()
        self.layer10 =  torch.nn.MaxPool2d(kernel_size=1, stride=2, padding=0, dilation=1, ceil_mode=False)   
        self.layer11 =  torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer12 =  torch.nn.ReLU()
        self.layer13 =  torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer14 =  torch.nn.ReLU()
        self.layer15 =  torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer16 =  torch.nn.ReLU()
        self.layer17 =  torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer18 =  torch.nn.ReLU()
        self.layer19 =  torch.nn.MaxPool2d(kernel_size=1, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer20 =  torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer21 =  torch.nn.ReLU()
        self.layer22 =  torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer23 =  torch.nn.ReLU()
        self.layer24 =  torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer25 =  torch.nn.ReLU()
        self.layer26 =  torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer27 =  torch.nn.ReLU()
        self.layer28 =  torch.nn.MaxPool2d(kernel_size=1, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer29 =  torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer30 =  torch.nn.ReLU()
        self.layer31 =  torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer32 =  torch.nn.ReLU()
        self.layer33 =  torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer34 =  torch.nn.ReLU()
        self.layer35 =  torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer36 =  torch.nn.ReLU()

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)
        x9 = self.layer9(x8)  
        x10 = self.layer10(x9)
        x11 = self.layer11(x10)
        x12 = self.layer12(x11)
        x13 = self.layer13(x12)
        x14 = self.layer14(x13)
        x15 = self.layer15(x14)
        x16 = self.layer16(x15)
        x17 = self.layer17(x16)
        x18 = self.layer18(x17)
        x19 = self.layer19(x18)
        x20 = self.layer20(x19)
        x21 = self.layer21(x20)
        x22 = self.layer22(x21)
        x23 = self.layer23(x22)
        x24 = self.layer24(x23)
        x25 = self.layer25(x24)
        x26 = self.layer26(x25)
        x27 = self.layer27(x26) 
        x28 = self.layer27(x27) 
        x29 = self.layer29(x28)
        x30 = self.layer30(x29)
        x31 = self.layer31(x30)
        x32 = self.layer32(x31)
        x33 = self.layer33(x32)
        x34 = self.layer34(x33)
        x35 = self.layer35(x34)
        x36 = self.layer36(x35)        
        return x36     

import time
## Define hook functions
take_time_dict = {}

def take_time_pre(layer_name,module, input):
    take_time_dict[layer_name] = time.time() 

def take_time(layer_name,module, input, output):
    take_time_dict[layer_name] =  time.time() - take_time_dict[layer_name]
    ## for TensorBoard you should use writter




from functools import partial


# Create Model
model = TheModel()

# Register function for every 
for layer in model.children():
    layer.register_forward_pre_hook( partial(take_time_pre, layer) )
    layer.register_forward_hook( partial(take_time, layer) )



#CPU
batch=1
input = torch.rand(batch,3,224,224)

# #GPU
# device_2 = 'cuda'
# model = model.to(device=device_2) 
# x = torch.rand(4,3,224,224).to(device_2)

first_time = datetime.now()
for i in range(10):
    model(input)
last_time = datetime.now()
print("warm-up time:", str((last_time - first_time)/10))

first_time = datetime.now()
for i in range(10):
    model(input)
last_time = datetime.now()
print("average of final exec time:", str((last_time - first_time)/10))

layer_no=0
for i in take_time_dict.values():
    layer_no+=1
    # print(f'{i:f}')
    # print("layer no ", count, " : ",i)
    total_time+=i



