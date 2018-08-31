from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import predict_transform
import cv2

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img,(416,416))
    img_ = img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_
    

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    #x is the input, CUDA is a GPU flag.
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {} #cache outputs for route layer

        write = 0

        for i, module in enumerate(modules):
            
            module_type = (module["type"])
    
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
                

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
        
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
    
                    x = torch.cat((map1,map2), 1)
                

            elif module_type =="shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
                

            elif module_type == "yolo":
                
                anchors = self.module_list[i][0].anchors
                #get input dimensions
                inp_dim = int (self.net_info["height"])
                
                #get # of classes
                num_classes = int(module["classes"])
                
                #Transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x 
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x 

    
        return detections

    def load_weights(self, weightfile):
        fp = open(weightfile, "rb") 

        #The first five weight values are header information
        # 1. Major Version Number
        # 2. Minor Version Number
        # 3. Subversion Number
        # 4,5 Images seen by the network (during training)
    
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype = np.float32)

        ptr = 0
    
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            #If module_type is convolutinal load weights
            #Otherwise ignore

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]
                
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
    
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    #Cast the loaded weights in dims of model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)                  

                    #Copy the data to the model.

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    #Number of biases
                    num_biases = conv.bias.numel()

                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)


                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

def parse_cfg(cfgfile):
    '''
    Parses YOLO configuration file with each block giving a dictionary of the type of block and parameters.
    '''
    file = open(cfgfile,"r")
    layers = []
    layer = {}


    for line in file:
        if line == '\n':
            continue

        if line[0] == '#':
            continue

        line = line.strip('\n')
        
        if line.startswith('['):
            if len(layer) != 0:
                layers.append(layer)
                layer = {}
            line = line.strip('[]')
            layer["type"] = line
        else:
            key, value  = line.rsplit('=',1)
            
            #rstrip, and lstrip remove whitespace
            layer[key.rstrip()] = value.lstrip()
    layers.append(layer)

    return layers


def create_modules(blocks):
    '''
    Creates the pytorch modules for the 5 different layers in Config File
    [1a] Net - stores network information and training parameters
    [1] Convolutional - normal conv layer with activation
    [2] Shortcut - resenet style shortcut layer
    [3] Upsample - Upsamples featuremap in previous layer by factor of stride
    [4] Route - layers value = output the feature map of layer indexing that value, layers value has 2, means concat layers from the layers specified along the depth dimension
    [5] Yolo - yolo detection layer. Only uses anchors specified by mask.
    '''
    
    #Stores the information about the network.
    net_info = blocks[0]

    #Creates a module list object which will add all of our parameters to the network as a member.
    module_list = nn.ModuleList()

    #Previous depth of filters, keep track of this so the convolution dimensions of sucessive layers match up.
    prev_filters = 3

    #Add the filter depth to output_filters list.  We need this information to match convolution dimensions when there is a convolutional layer in front of a route layer.
    output_filters = []
    
    #Iterate over each block after the net, enumerating to keep an index.
    for index, x in enumerate(blocks[1:]):
        
        
        #Sequential, sequentially executes module objects
        #Conv layer has batch norm and leaky relu, so we need to add multiple modules per layer
        module = nn.Sequential()
        
        #Check type of block, create new module for block add module to the list.

        if (x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            #Create the convolutional layer in pytorch
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            #Name the layer based on the enumerated index
            module.add_module("conv_{0}".format(index), conv)

            #Create the batch norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index),bn)

            #Do a check of the activation, YOLO use linear or leaky relu activation in its layers
            if activation == "leaky":
                active = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), active)
                
        # Next type of layer to check for is upsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            module.add_module("upsample_{}".format(index),upsample)

        # Need custom code for routing and shortcut layers
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            
            #Find the start of the route
            start = int(x["layers"][0])
            
            #Test if there is an end to route otherwise return 0
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
                       
            route = EmptyLayer()

            module.add_module("route_{0}".format(index), route)
    
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:

                filters = output_filters[index + start]

        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
            
            #Split by ,
            anchors = x["anchors"].split(",")
            #convert to ints
            anchors = [int(a) for a in anchors]
            #Group pairs of 2
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            #Only keep anchors based on mask
            anchors = [anchors[i] for i in mask]
            
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)


        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)    

class DetectionLayer(nn.Module):
    '''
    Yolo layer
    '''
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class EmptyLayer(nn.Module):
    '''
    Empty Layer for shortcut and routing layers.
    '''
    def __init__(self):
        super(EmptyLayer, self).__init__()

'''
def main():
    #layers = parse_cfg("yolov3.cfg")    
    #print(layers)
    #print(create_modules(layers))

    #Darknet("yolov3.cfg")


    model = Darknet("yolov3.cfg")
    model.load_weights("yolov3.weights")
    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.cuda()

    inp = get_test_input()
    inp = inp.cuda()
    pred = model(inp, CUDA)
    print(pred)
    print(pred.shape)

if __name__ == "__main__":
    main()
'''
