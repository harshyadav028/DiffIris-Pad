import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
# Enable cuDNN autotuner for optimized performance
torch.backends.cudnn.benchmark = True
import os
import json
import matplotlib.pyplot as plt
from Evaluation import evaluation
from sklearn.metrics import classification_report
# Optional: Enable torch.compile for model graph optimization (requires PyTorch 2.x)
# model = torch.compile(model)

def training(model,dataloader, args, criterion, optimizer, scheduler,root,m,o):
    torch.cuda.empty_cache()
    #model = torch.compile(model)
    
    # Creation of Log folder: used to save the trained model
    log_path = os.path.join(root,m, o, 'Logs')
    os.makedirs(log_path, exist_ok=True)

    # Creation of result folder: used to save the performance of trained model on the test set
    result_path = os.path.join(root,m, o, 'Results')
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(log_path,'params.json'), 'w') as out:
        hyper = vars(args)
        json.dump(hyper, out)
    log = {'iterations':[], 'epoch':[], 'validation':[], 'train_acc':[], 'val_acc':[], 'BestThresh':0.5, 'total params': sum(p.numel() for p in model.parameters()), 'trainable params': sum(p.numel() for p in model.parameters() if p.requires_grad), 'non-trainable params': sum(p.numel() for p in model.parameters() if not p.requires_grad)}

    if torch.cuda.is_available():
        model.cuda()

    scaler = torch.amp.GradScaler('cuda') if args.use_amp else None
    train_loss=[]
    val_loss=[]
    bestAccuracy=0
    print(f'Training {m} with {o} ...')
    for epoch in range(args.nEpochs):
        start=time.time()
        for phase in ['train', 'val']:
            loader=dataloader[phase]
            if phase == 'train':
                model.train()
            else:
                model.eval()

            tloss = 0.   #avg loss over a batch
            acc = 0.
            tot = 0         #stores toal number of samples
            c = 0           #stores number of batches
            testPredScore = []
            testTrueLabel = []
            imgNames=[]

            iterator = tqdm(loader, desc=f"Epoch {epoch+1}/{args.nEpochs} {phase} {m} {o}", leave=False)

            for data, labels ,imageName ,attack in iterator:
                data, labels = data.cuda(non_blocking=True), labels.cuda(non_blocking=True)

                with torch.amp.autocast('cuda',enabled=args.use_amp):
                    outputs = model(data)
                    pred = torch.max(outputs,dim=1)[1]
                    corr = torch.sum((pred == labels).int())
                    acc += corr.item()
                    tot += data.size(0)  
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    log['iterations'].append(loss.item())
                elif phase == 'val':
                    temp = outputs.detach().cpu().numpy()
                    scores = np.stack((temp[:,0], np.amax(temp[:,1:args.nClasses], axis=1)), axis=-1)
                    testPredScore.extend(scores)
                    testTrueLabel.extend((labels.detach().cpu().numpy()>0)*1)
                    imgNames.extend(imageName)


                tloss += loss.item()
                c+=1
                iterator.set_postfix(loss=loss.item())

            if phase == 'train':
                scheduler.step()
                log['epoch'].append(tloss/c)
                log['train_acc'].append(acc/tot)
                print(f"Epoch {epoch+1}/{args.nEpochs}, Training Loss: {tloss/c:.4f}, Accuracy: {acc/tot:.4f}")
                train_loss.append(tloss / c)
            elif phase == 'val':
                log['validation'].append(tloss / c)
                log['val_acc'].append(acc / tot)
                print(f"Epoch {epoch+1}/{args.nEpochs}, Validation Loss: {tloss/c:.4f}, Accuracy: {acc/tot:.4f}")
                val_loss.append(tloss / c)
                if (acc/tot > bestAccuracy):
                    bestAccuracy = acc/tot
                    TrueLabels = testTrueLabel
                    PredScores = testPredScore
                    save_best_model = os.path.join(log_path,m+'_best.pth')
                    states = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(states, save_best_model)
                    ImageNames= imgNames
        end = time.time()
        print(f"Epoch {epoch+1}/{args.nEpochs} completed in {end - start:.2f} seconds\n")
        torch.save(model.state_dict(), os.path.join(log_path,m+'_epoch.pt'))
        torch.cuda.empty_cache()
    print(f"Training completed. Best Validation Accuracy: {bestAccuracy:.4f}")

    # Plotting of train and test loss
    plt.figure()
    plt.xlabel('Epoch Count')
    plt.ylabel('Loss')
    plt.plot(np.arange(0, args.nEpochs), train_loss[:], color='r')
    plt.plot(np.arange(0, args.nEpochs), val_loss[:], 'b')
    plt.legend(('Train Loss', 'Validation Loss'), loc='upper right')
    plt.title(f'{m} with {o} Loss')
    plt.savefig(os.path.join(result_path,m+'model_Loss.jpg'))

    # Evaluation of test set utilizing the trained model
    all_attacks=[]
    for _,_,_,attack in dataloader['val']:
        all_attacks.extend(attack)  # assuming batch is a dict
    
    #print(all_attacks)
    obvResult = evaluation()
    errorIndex, predictScore, bestThreshold = obvResult.get_result(m, ImageNames, TrueLabels, PredScores,all_attacks, result_path)
    print("best Threshold :", bestThreshold)
    log['BestThresh']=bestThreshold
    with open(os.path.join(log_path,m+'_log.json'), 'w') as out:
        json.dump(log, out)
    # Testing phase
    imgNames=[]

    torch.cuda.empty_cache()
    print(f'Testing {m} with {o}...')
    model.eval()
    with torch.no_grad():
        all_outputs, all_labels, all_pai = [], [], []
        for data, labels,imageName, attack in tqdm(dataloader['test'], desc="Testing", leave=False):
            data, labels = data.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            with torch.amp.autocast('cuda',enabled=args.use_amp):
                outputs = model(data)

            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_pai.extend(attack)
            imgNames.extend(imageName)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        #all_pai=torch.cat(all_pai,dim=0)

    obvResult = evaluation()
    errorIndex, predictScore, threshold = obvResult.get_result(m, imgNames, all_labels, all_outputs,all_pai, result_path,minThreshold = bestThreshold)
    y_pred = (all_outputs > bestThreshold).int().max(dim=1)[1]
    report = classification_report(all_labels, y_pred, target_names=['Live','Spoof'], output_dict=True)
    with open(os.path.join(log_path,m+'_report.json'), 'w') as out:
        json.dump(report, out, indent=4)
    return threshold

def get_data_loaders(dataset_train, dataset_val,dataset_test, batch_size):
    num_workers = min(32, multiprocessing.cpu_count())
    return (
        torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
    )
