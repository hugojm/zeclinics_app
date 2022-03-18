import copy
import csv
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from evaluate import evaluate_sample


def train_model(model, criterion, dataloaders, optimizer, metrics, masks_names, bpath,
                num_epochs, device='cpu', best_loss=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    if best_loss is None:
        best_loss = float('inf')  # Initialize best loss to +oo if not best_loss is passed as an input

    # Use gpu if available and send model to device
    model.to(device)

    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Test_{m}_{mask_name}' for m in metrics.keys() for mask_name in masks_names] + \
        [f'Test_{m}' for m in metrics.keys()]

    # Open a log file to store the metrics
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary -> Contains information about metrics
        batchsummary = {a: [] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            
            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                masks = sample['masks'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs['out'], masks)
                    
                    if phase == 'Test':
                        masks_pred_by_sample = torch.split(outputs['out'], 1, dim = 0)
                        masks_true_by_sample = torch.split(masks, 1, dim = 0)

                        for i in range(len(masks_pred_by_sample)):
                            #print("evaluate sample")
                            metrics_sample = evaluate_sample(masks_true_by_sample[i],
                                                             masks_pred_by_sample[i],
                                                             masks_names, metrics)
                            for metric_name, metric_value in metrics_sample.items():
                                batchsummary[f'{phase}_{metric_name}'].append(metric_value)
                            #print('finished sample', i)
                    else:
                        loss.backward()
                        optimizer.step()

            batchsummary['epoch'] = epoch
            batchsummary[f'{phase}_loss'] = loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)

        # Write in log.csv batchsummary
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)

        # Deep copy the model only if loss is decreased in Test set
        if phase == 'Test' and loss < best_loss:
            best_loss = loss
            best_model_wts = copy.deepcopy(model.state_dict())
            # model.load_state_dict(best_model_wts) # Descomentar para que te vaya guardando

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return best_loss


#####################################################################################################


def train_clasif_model(model, feno_names, criterion, dataloaders, optimizer, metrics, bpath,
                num_epochs, bs, batch_acum=1, class_weights = None):
    if class_weights is None:
        class_weights = [1/len(feno_names) for f in feno_names]
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_weights = torch.Tensor(class_weights).to(device)
    model.to(device)
    #feno_names = ['bodycurvature', 'yolkedema', 'necrosis', 'tailbending', 'notochorddefects', 'craniofacialedema', 'finabsence', 'scoliosis', 'snoutjawdefects', 'otoliths$
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Test_{m}_{feno_name}' for m in metrics.keys() for feno_name in feno_names] + \
        [f'Test_{m}' for m in metrics.keys()] + \
        [f'Train_{m}_{feno_name}' for m in metrics.keys() for feno_name in feno_names] + \
        [f'Train_{m}' for m in metrics.keys()] + ['Test_hmean_f1','Train_hmean_f1']

    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            #store phase predictions to later compute metrics
            test_pred = [[] for i in range(10)]
            test_true = [[] for i in range(10)]
            train_pred = [[] for i in range(10)]
            train_true = [[] for i in range(10)]
            epoch_loss = 0

            acum_steps = 0
            # Iterate over data.
            num_samples = len(dataloaders[phase])
            for j, sample in enumerate(tqdm(iter(dataloaders[phase]))):
                inputs = sample['image'].to(device)
                feno = torch.Tensor(torch.stack([v for k,v in sample['fenotypes'].items()]).float()).to(device)
                feno = torch.transpose(feno,0,1)
                # zero the parameter gradients
                if acum_steps == 0:
                    optimizer.zero_grad()

                # track gradient history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, feno)
                    loss = loss.mean(dim = 0)
                    loss = torch.dot(loss,class_weights)
                    epoch_loss += loss
                    pred_by_feno = torch.split(outputs, 1, dim = 1)
                    true_by_feno = torch.split(feno, 1, dim = 1)
                    for i in range(len(pred_by_feno)):
                        #print(test_pred)
                        #print(torch.squeeze(torch.transpose(feno_pred_by_feno[i],0,1)))
                        if phase == 'Test':
                            test_pred[i] += list(torch.squeeze(torch.transpose(pred_by_feno[i],0,1)).cpu().data.numpy().ravel())
                            test_true[i] += list(torch.squeeze(torch.transpose(true_by_feno[i],0,1)).cpu().data.numpy().ravel())
                        else:
                            train_pred[i] += list(torch.squeeze(torch.transpose(pred_by_feno[i],0,1)).cpu().data.numpy().ravel())
                            train_true[i] += list(torch.squeeze(torch.transpose(true_by_feno[i],0,1)).cpu().data.numpy().ravel())
                    acum_steps += 1
                    if phase == 'Train':
                        loss.backward()
                        if acum_steps == batch_acum or j == num_samples-1:
                            optimizer.step()
                            acum_steps = 0
            '''
            Update batchsummary
            --------------------------------------------------------------------
            '''
            batchsummary['epoch'] = epoch
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            for name, metric in metrics.items():
                for i,feno in enumerate(feno_names):
                    #print("true:",torch.Tensor(test_true[i]))
                    #print("pred:",torch.sigmoid(torch.Tensor(test_pred[i])))
                    if phase == 'Test':
                        value = metric(torch.Tensor(test_true[i]), torch.sigmoid(torch.Tensor(test_pred[i])) > 0.5)
                    else:
                        value = metric(torch.Tensor(train_true[i]), torch.sigmoid(torch.Tensor(train_pred[i])) > 0.5)
                    batchsummary[f'{phase}_{name}_{feno}'] = value
                    batchsummary[f'{phase}_{name}'].append(value)
            batchsummary[f'{phase}_hmean_f1'] = hmean(batchsummary[f'{phase}_f1_score'])
            batchsummary[f'{phase}_f1_score'] = np.mean(batchsummary[f'{phase}_f1_score'])
            batchsummary[f'{phase}_precision'] = np.mean(batchsummary[f'{phase}_precision'])
            batchsummary[f'{phase}_recall'] = np.mean(batchsummary[f'{phase}_recall'])
            '''
            --------------------------------------------------------------------
            '''
        #print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
        # deep copy the model
        if phase == 'Test' and batchsummary[f'{phase}_hmean_f1'] >= best_f1:
            best_f1 = batchsummary[f'{phase}_hmean_f1']
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model, str(bpath) +'/weights.pt')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Bigges F1: {:4f}'.format(best_f1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


