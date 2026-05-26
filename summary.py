# from models import ResNetModel
from torchinfo import summary
# model = ResNetModel()
# summary(model, input_size=(1, 3, 224, 224))

import torch
import torch.nn as nn
import torch.optim as optim
from open_set_training import training,get_data_loaders
import argparse
from open_set_dataset_loader import datasetLoader
from models import DenseNetModel
import time
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

parser = argparse.ArgumentParser()
parser.add_argument('-batchSize', type=int, default=32)
parser.add_argument('-nEpochs', type=int, default=50)
parser.add_argument('-csvPath', required=False, default= 'Images/combined_dataset.csv',type=str)
parser.add_argument('-datasetPath', required=False, default= 'Images/',type=str)
parser.add_argument('-outputPath', required=False, default= 'weights/',type=str)
parser.add_argument('-nClasses', default= 2,type=int)
parser.add_argument('-use_amp', default= False,type=bool)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

attacks_list = ['Artifact']
start = time.time()
for attack in attacks_list:
    print(attack)
    dataseta = datasetLoader(args.csvPath,args.datasetPath,train_test='train',attack_type=attack)
    datasetb = datasetLoader(args.csvPath,args.datasetPath, train_test='val', c2i=dataseta.class_to_id,attack_type=attack)
    datasetc = datasetLoader(args.csvPath,args.datasetPath, train_test='test', c2i=dataseta.class_to_id,attack_type=attack)

    print(f'len of train = {len(dataseta)} , val = {len(datasetb)}, test = {len(datasetc)}')
    print('\n\n')
    train,val,test = get_data_loaders(dataseta, datasetb, datasetc, args.batchSize)
    dataloader = {'train': train, 'val':val, 'test':test}

    model1 = DenseNetModel()
    
    def print_layer_names(mod):
        """
        Print names of all submodules/layers of a model.
        If model wraps an underlying model in attribute 'model', use that.
        """
        target = getattr(mod, 'model', mod)
        for name, module in target.named_modules():
            if name == '':
                continue
            print(name)
    print("Model layer names:")
    print_layer_names(model1)
    

#     for param in model1.parameters():
#         param.requires_grad = False
#     for param in model1.model.layer4.parameters():
#         param.requires_grad = True
    
#     criterion = nn.CrossEntropyLoss()
	
#     optimizer = torch.optim.SGD(model1.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
#     lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
#     t1=training(model1, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','ResNet34_LastBlock',attack)
# end = time.time()
# print(f'Total Time taken: {end-start} seconds')