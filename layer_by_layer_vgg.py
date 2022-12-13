import torch
from datetime import datetime
from torchsummary import summary

import time


class TheModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
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
        return x10


class TheModel2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer11 =  torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer12 =  torch.nn.ReLU()
        self.layer13 =  torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer14 =  torch.nn.ReLU()
        self.layer15 =  torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer16 =  torch.nn.ReLU()
        self.layer17 =  torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer18 =  torch.nn.ReLU()
        self.layer19 =  torch.nn.MaxPool2d(kernel_size=1, stride=2, padding=0, dilation=1, ceil_mode=False)

    def forward(self, x):
        x11 = self.layer11(x)
        x12 = self.layer12(x11)
        x13 = self.layer13(x12)
        x14 = self.layer14(x13)
        x15 = self.layer15(x14)
        x16 = self.layer16(x15)
        x17 = self.layer17(x16)
        x18 = self.layer18(x17)
        x19 = self.layer19(x18)
        return x19

class TheModel3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer20 =  torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer21 =  torch.nn.ReLU()
        self.layer22 =  torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer23 =  torch.nn.ReLU()
        self.layer24 =  torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer25 =  torch.nn.ReLU()
        self.layer26 =  torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer27 =  torch.nn.ReLU()
        self.layer28 =  torch.nn.MaxPool2d(kernel_size=1, stride=2, padding=0, dilation=1, ceil_mode=False)

    def forward(self, x):
        x20 = self.layer20(x)
        x21 = self.layer21(x20)
        x22 = self.layer22(x21)
        x23 = self.layer23(x22)
        x24 = self.layer24(x23)
        x25 = self.layer25(x24)
        x26 = self.layer26(x25)
        x27 = self.layer27(x26) 
        x28 = self.layer27(x27)        
        return x28



class TheModel4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer29 =  torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer30 =  torch.nn.ReLU()
        self.layer31 =  torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer32 =  torch.nn.ReLU()
        self.layer33 =  torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer34 =  torch.nn.ReLU()
        self.layer35 =  torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer36 =  torch.nn.ReLU()

    def forward(self, x):
        x29 = self.layer29(x)
        x30 = self.layer30(x29)
        x31 = self.layer31(x30)
        x32 = self.layer32(x31)
        x33 = self.layer33(x32)
        x34 = self.layer34(x33)
        x35 = self.layer35(x34)
        x36 = self.layer36(x35)        
        return x36     


## Define hook functions
take_time_dict = {}
layer_execs={}
total_time_gpu=0
total_time_cpu=0

def take_time_pre(layer_name,module, input):
    take_time_dict[layer_name] = time.time() 

def take_time(layer_name,module, input, output):
    take_time_dict[layer_name] =  time.time() - take_time_dict[layer_name]
    ## for TensorBoard you should use writter

def inference(model, input):

    from functools import partial


    # # Create Model
    # print("model shape:",type(model))

    # Register function for every 
    for layer in model.children():
        layer.register_forward_pre_hook( partial(take_time_pre, layer) )
        layer.register_forward_hook( partial(take_time, layer) )

    # x = torch.rand(1,5)

    first_time = datetime.now()
    #warm-up iterations
    for i in range(50):
        model(x)
    last_time = datetime.now()
    print("average warm-up time:", str((last_time - first_time)/50))

    #The data we collect
    for i in range(500):
        model(x)
    last_time = datetime.now()
    print("average of final exec time:", str((last_time - first_time)/500))

    total = 0
    layer_execs.update(take_time_dict)
    for i in take_time_dict.values():
        total += i
    # print(total)
    print() 

    return total



#CPU INFERENCE
model = TheModel()
summary(model)
x = torch.rand(1,3,224,224)
total_time_cpu+=inference(model, x)


#GPU INFERENCE
device_2 = 'cuda'
model = model.to(device=device_2) 
x = torch.rand(1,3,224,224).to(device_2)
total_time_gpu+=inference(model, x)

model = TheModel2()
x = torch.rand(1,128,56,56)
total_time_cpu+=inference(model, x)
device_2 = 'cuda'
x = torch.rand(1,128,56,56).to(device_2)
model = model.to(device=device_2) 
total_time_gpu+=inference(model, x)

model = TheModel3()
x = torch.rand(1,256,28,28)
total_time_cpu+=inference(model, x)
device_2 = 'cuda'
x = torch.rand(1,256,28,28).to(device_2)
model = model.to(device=device_2)
total_time_gpu+=inference(model, x)

model = TheModel4()
x = torch.rand(1,512,14,14)
total_time_cpu+=inference(model, x)
device_2 = 'cuda'
x = torch.rand(1,512,14,14).to(device_2)
model = model.to(device=device_2) 
total_time_gpu+=inference(model, x)


print("total time cpu: ",total_time_cpu)
print("total time gpu: ",total_time_gpu)
exit()



##I was playing around and leave this here it might need to check later.


take_time_dict = {}

def take_time_pre(layer_name,module, input):
    take_time_dict[layer_name] = time.time() 

def take_time(layer_name,module, input, output):
    take_time_dict[layer_name] =  time.time() - take_time_dict[layer_name]
    ## for TensorBoard you should use writter
# Create Model
model = TheModel2()
print("model shape:",type(model))

# Register function for every 
for layer in model.children():
    layer.register_forward_pre_hook( partial(take_time_pre, layer) )
    layer.register_forward_hook( partial(take_time, layer) )

x = torch.rand(1,128,56,56)
# x = torch.rand(1,5)

first_time = datetime.now()
model(x) 
last_time = datetime.now()
print("first warm-up time:",str(last_time - first_time))

first_time = datetime.now()
for i in range(50):
    model(x)
last_time = datetime.now()
print("warm-up time:", str(last_time - first_time))

first_time = datetime.now()
model(x) 
last_time = datetime.now()
print("final exec",str(last_time - first_time))

# print(take_time_dict)
total = 0
layer_execs.update(take_time_dict)
for i in take_time_dict.values():
    # print(i)
    total += i
print(total)
total_time+=total


print("\n\n\n")


model = TheModel3()
print("model shape:",type(model))
take_time_dict = {}

def take_time_pre(layer_name,module, input):
    take_time_dict[layer_name] = time.time() 

def take_time(layer_name,module, input, output):
    take_time_dict[layer_name] =  time.time() - take_time_dict[layer_name]
    ## for TensorBoard you should use writter
# Create Model

# Register function for every 
for layer in model.children():
    layer.register_forward_pre_hook( partial(take_time_pre, layer) )
    layer.register_forward_hook( partial(take_time, layer) )

x = torch.rand(1,256,28,28)
# x = torch.rand(1,5)
first_time = datetime.now()
model(x) 
last_time = datetime.now()
print("first warm-up time:",str(last_time - first_time))

first_time = datetime.now()
for i in range(50):
    model(x)
last_time = datetime.now()
print("warm-up time:", str(last_time - first_time))

first_time = datetime.now()
model(x) 
last_time = datetime.now()
print("final exec",str(last_time - first_time))

# print(take_time_dict)
total = 0
layer_execs.update(take_time_dict)
for i in take_time_dict.values():
    # print(i)
    total += i
print(total)
total_time+=total



print("\n\n\n")


model = TheModel4()
print("model shape:",type(model))

take_time_dict = {}

def take_time_pre(layer_name,module, input):
    take_time_dict[layer_name] = time.time() 

def take_time(layer_name,module, input, output):
    take_time_dict[layer_name] =  time.time() - take_time_dict[layer_name]
    ## for TensorBoard you should use writter

# Register function for every 
for layer in model.children():
    layer.register_forward_pre_hook( partial(take_time_pre, layer) )
    layer.register_forward_hook( partial(take_time, layer) )

x = torch.rand(1,512,14,14)
# x = torch.rand(1,5)
first_time = datetime.now()
model(x) 
last_time = datetime.now()
print("first warm-up time:",str(last_time - first_time))

first_time = datetime.now()
for i in range(50):
    model(x)
last_time = datetime.now()
print("warm-up time:", str(last_time - first_time))

first_time = datetime.now()
model(x) 
last_time = datetime.now()
print("final exec",str(last_time - first_time))

# print(take_time_dict)
total = 0
layer_execs.update(take_time_dict)
for i in take_time_dict.values():
    # print(i)
    total += i
print(total)
total_time+=total

print("\n\n\n\n ---------------Total-------------\n")
count=0
for i in layer_execs.values():
    count+=1
    print("count ", count, " : ",i)

print("total execution time of all groups:", total_time)


print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
  
# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")
        
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")