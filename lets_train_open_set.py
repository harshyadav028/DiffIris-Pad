import os
# os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from open_set_training import training,get_data_loaders
import argparse
from open_set_dataset_loader import datasetLoader
from models import ResNetModel , DINOResNetModel, DenseNetModel, ResNet34Model, ResNet101Model, EfficientNetV2SModel, MobileNetV2Model , MobileNetV3LargeModel, ResNeXtModel, WideResNetModel, Res2NetModel, SENetModel, SKNetModel ,ViTModel,DeiTModel, PVTModel, CvTModel, TNTModel, CrossViTModel, SwinV2Model, VisionMambaModel, MAEModel, BEiTModel, SimMIMModel, MobileViTModel, MaxViTModel ,DinoV2ViTModel, EnsembleModel1, EnsembleModel2
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

#attacks_list = ['Artifact','CL','E-display','Fake with Add On','Generated','PostMortem','Print and E-display','Printed']
attacks_list = ['Printed']
#attacks_list = ['Artifact','CL','E-display','Fake with Add On','Generated','PostMortem','Print and E-display',]
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

    #model = DinoVisionTransformerClassifier()
    #Deep Learning models
    # model1 = ResNet34Model()
    # model2 = ResNet101Model()
    # model3 = EfficientNetV2SModel()
    # model4 = MobileNetV2Model()
    # model5 = MobileNetV3LargeModel()
    # model6 = ResNeXtModel()
    # model7 = WideResNetModel()
    # model8 = Res2NetModel()
    # model9 = SENetModel()
    # model10 = SKNetModel()
    # model11 = ResNetModel()
    # model12 = DenseNetModel()
    #Vision Transformer models
    model13 = ViTModel(variant='vit_base_patch16_224')
    model14 = DeiTModel()
    model15 = PVTModel()
    # model16 = CvTModel() --
    model17 = TNTModel()
    model18 = CrossViTModel()
    model19 = SwinV2Model(variant='swinv2_base_window12_192_22k')
    # model20 = VisionMambaModel() --
    model21 = MAEModel()
    model22 = BEiTModel()
    model23 = SimMIMModel()
    model24 = MobileViTModel()
    model25 = MaxViTModel()
    # model26 = ViTModel(variant='vit_large_patch16_224')
    # model27 = ViTModel(variant='vit_huge_patch14_224')
    model28 = SwinV2Model(variant='swinv2_small_window8_256.ms_in1k')
    # model29 = SwinV2Model(variant='swinv2_large_window12_192_22k')
    criterion = nn.CrossEntropyLoss()

    if attack!='Printed':
        optimizer = torch.optim.SGD(model13.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
        lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
        t13=training(model13, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','vit_base_patch16_224',attack)

        optimizer = torch.optim.SGD(model14.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
        lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
        t14=training(model14, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','DeiTModel',attack)

        optimizer = torch.optim.SGD(model15.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
        lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
        t15=training(model15, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','PVTModel',attack)

        # optimizer = torch.optim.SGD(model16.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
        # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
        # t16=training(model16, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','CvTModel',attack)

        optimizer = torch.optim.SGD(model17.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
        lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
        t17=training(model17, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','TNTModel',attack)

        optimizer = torch.optim.SGD(model18.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
        lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
        t18=training(model18, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','CrossViTModel',attack)

        optimizer = torch.optim.SGD(model19.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
        lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
        t19=training(model19, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','swinv2_base_window12_192_22k',attack)

        # optimizer = torch.optim.SGD(model20.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
        # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
        # t20=training(model20, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','VisionMambaModel',attack)

        optimizer = torch.optim.SGD(model21.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
        lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
        t21=training(model21, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','MAEModel',attack)

        optimizer = torch.optim.SGD(model22.parameters(),lr=0.0005, weight_decay=1e-6, momentum=0.9)
        lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
        t22=training(model22, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','BEiTModel',attack)

        optimizer = torch.optim.SGD(model23.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
        lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
        t23=training(model23, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','SimMIMModel',attack)

        optimizer = torch.optim.SGD(model24.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
        lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
        t24=training(model24, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','MobileViTModel',attack)

    optimizer = torch.optim.SGD(model25.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    t25=training(model25, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','MaxViTModel',attack)

    # optimizer = torch.optim.SGD(model26.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # t26=training(model26, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','vit_large_patch16_224',attack)

    # optimizer = torch.optim.SGD(model27.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # t27=training(model27, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','vit_huge_patch14_224',attack)

    optimizer = torch.optim.SGD(model28.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    t28=training(model28, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','swinv2_small_window8_256_22k',attack)

    # optimizer = torch.optim.SGD(model29.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # t29=training(model29, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','swinv2_large_window12_192_22k',attack)

#########################################################################################################################################
    # optimizer = torch.optim.SGD(model1.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # t1=training(model1, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','ResNet34_LastBlock',attack)

    # optimizer = torch.optim.SGD(model2.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # t2=training(model2, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','ResNet101_LastBlock',attack)

    # optimizer = torch.optim.SGD(model3.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # t3=training(model3, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','EfficientNetV2SModel_LastBlock',attack)

    # optimizer = torch.optim.SGD(model4.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # t4=training(model4, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','MobileNetV2Model_LastBlock',attack)

    # optimizer = torch.optim.SGD(model5.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # t5=training(model5, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','MobileNetV3LargeModel_LastBlock',attack)

    # optimizer = torch.optim.SGD(model6.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # t6=training(model6, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','ResNeXtModel_LastBlock',attack)

    # optimizer = torch.optim.SGD(model7.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # t7=training(model7, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','WideResNetModel_LastBlock',attack)

    # optimizer = torch.optim.SGD(model8.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # t8=training(model8, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','Res2NetModel_LastBlock',attack)

    # optimizer = torch.optim.SGD(model9.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # t9=training(model9, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','SENetModel_LastBlock',attack)

    # optimizer = torch.optim.SGD(model10.parameters(),lr=0.0005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # t10=training(model10, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','SKNetModel_LastBlock',attack)

    # optimizer = torch.optim.SGD(model11.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # t11=training(model11, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','ResNet50_LastBlock',attack)

    # optimizer = torch.optim.SGD(model12.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # t12=training(model12, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','DenseNet121_LastBlock',attack)


    # optimizer = torch.optim.SGD(model2.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # t2=training(model2, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','DinoResNet50',attack)

    # optimizer = torch.optim.SGD(model3.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # t3=training(model3, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','DenseNet121',attack)

    # optimizer = optim.Adam(model4.parameters(), lr=0.0000005)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=25, gamma=0.1)
    # t4=training(model4, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','DinoV2',attack)







    #model5 = EnsembleModel1(base_models=model_dict, thresholds=thresholds)
    # model6 = EnsembleModel2(base_models=model_dict, thresholds=thresholds)

    # optimizer = torch.optim.SGD(model5.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # training(model5, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','Ensemble1',attack)

    # optimizer = torch.optim.SGD(model6.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    # lr_sched = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
    # training(model6, dataloader, args, criterion, optimizer, lr_sched,'Open_Set','Ensemble2',attack)

end = time.time()
print(f'Total Time taken: {end-start} seconds')
