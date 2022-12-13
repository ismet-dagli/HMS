
This repository contains the code for pytorch to run on CPU and GPU on Xavier AGX. The JetPack(JP) version is 4.5. (R32.5)

# SETUP

Prerequisites: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html ( Follow the steps until installing PyTorch.)
Installing the pytorch on original source did not work for me (E.g. Don't use here: https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/)

Install GPU pytorch wheel https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

If you receive numpy dependency error, install numpy 1.19.4: pip3 install numpy==1.19.4


## To use GPU
No changes is required for CPU. To be able to use GPU, modify the model as such
```
device = 'cuda'
x = torch.rand(1,3,32,32).to(device)
net= net.to("cuda")
```

To update the batch size, change the first parameter of input data: x = torch.rand(batch,3,32,32)

## Layer Profiling
For individual layer profiling, models may need to be re-written as seperate layers rather than blocks/groups etc.

This code can 

```
import time
take_time_dict = {}

def take_time_pre(layer_name,module, input):
    take_time_dict[layer_name] = time.time() 

def take_time(layer_name,module, input, output):
    take_time_dict[layer_name] =  time.time() - take_time_dict[layer_name]
    ## for TensorBoard you should use writter

from functools import partial
for layer in net.children():
    layer.register_forward_pre_hook( partial(take_time_pre, layer) )
    layer.register_forward_hook( partial(take_time, layer) )


#RUN THE MODEL
TO-DO

#PRINTS THE LAYER EXEC TIMES.
for i in take_time_dict.values():
    print(f'{i:f}')

```

take_time_dict keeps the layer info and execution time of per sequential block. This sequential block can be defined layer-wise (like vgg_layers.py) or can be group wise (resnet_torch.py) or can be whole neural network(vgg.py). To be able to get layer-wise results from each DNN, sequential layer definitions should be updated to similar pattern on vgg_layers.py


## CPU core selection

Use taskset. Taskset -c $core_numbers $command_to_run. (e.g. taskset -c 4,5,6,7 python3 resnet_torch.py)