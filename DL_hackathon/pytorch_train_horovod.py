# Copyright (c) 2017, PyTorch contributors
# Modifications copyright (C) Microsoft Corporation
# Licensed under the BSD license
# Adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torch.utils.data.distributed
import numpy as np
import time
import os
import copy
import argparse
import pickle
from tensorboardX import SummaryWriter
import horovod.torch as hvd

hvd.init()

from azureml.core.run import Run
# get the Azure ML run object
run = Run.get_context()

def load_data(data_dir):
    """Load the train/val data."""

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    samplers = {x: torch.utils.data.distributed.DistributedSampler(
        image_datasets[x], num_replicas=hvd.size(), rank=hvd.rank())
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                        sampler=samplers[x], num_workers=1)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) / hvd.size() for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

def train_model(model, criterion, optimizer, scheduler, num_epochs, data_dir, writer):
    """Train the model."""

    # load training/validation data
    dataloaders, dataset_sizes, class_names = load_data(data_dir)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(hvd.local_rank())
    else:
        device = torch.device('cpu')
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                corrects = torch.sum(preds == labels.data)
                running_corrects += corrects
                niter = epoch * len(dataloaders[phase]) + batch_idx
                if hvd.rank() == 0:
                    writer.add_scalar(f'{phase}/Loss', loss.item(), niter)
                    writer.add_scalar(f'{phase}/Accuracy', (corrects / inputs.size(0)).item(), niter)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_loss = metric_average(epoch_loss, 'epoch_loss')
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc = metric_average(epoch_acc, 'epoch_acc')

            if hvd.rank() == 0:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                writer.add_scalar(f'{phase}/Epoch_accuracy', epoch_acc, (epoch+1) * len(dataloaders[phase]))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # log the best val accuracy to AML run
            if hvd.rank() == 0 and phase == 'val':
                run.log('best_val_acc', np.float(best_acc))

        print()

    time_elapsed = time.time() - since
    if hvd.rank() == 0:
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return (model, class_names)


def fine_tune_model(num_epochs, data_dir, learning_rate, momentum, writer):
    """Load a pretrained model and reset the final fully connected layer."""
    _, _, class_names = load_data(data_dir)
    num_classes = len(class_names)
    # log the hyperparameter metrics to the AML run
    if hvd.rank() == 0:
        run.log('lr', np.float(learning_rate))
        run.log('momentum', np.float(momentum))
        run.log('num_classes', num_classes)

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)  # 40 classes to predict

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(hvd.local_rank())
    else:
        device = torch.device('cpu')
    model_ft = model_ft.to(device)

    # Horovod: broadcast parameters.
    hvd.broadcast_parameters(model_ft.state_dict(), root_rank=0)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate * hvd.size(), momentum=momentum)

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer_ft = hvd.DistributedOptimizer(optimizer_ft,
                                     named_parameters=model_ft.named_parameters())

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs, data_dir, writer)

    return model
  
def fixed_feature_model(num_epochs, data_dir, learning_rate, momentum, writer):
    _, _, class_names = load_data(data_dir)
    num_classes = len(class_names)
    if hvd.rank() == 0:
        run.log('lr', np.float(learning_rate))
        run.log('momentum', np.float(momentum))
        run.log('num_classes', num_classes)

    model_conv = models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_classes)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(hvd.local_rank())
    else:
        device = torch.device('cpu')
    model_conv = model_conv.to(device)

    # Horovod: broadcast parameters.
    hvd.broadcast_parameters(model_conv.state_dict(), root_rank=0)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr =learning_rate,momentum=momentum)

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer_conv = hvd.DistributedOptimizer(optimizer_conv,
                                     named_parameters=model_conv.named_parameters())

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    model = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs, data_dir, writer)

    return model

def main():
    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='breeds-10', help='directory of training data')
    parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs to train')
    parser.add_argument('--output_dir', type=str, default='outputs', help='output directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='log directory')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--mode', type=str, default='fixed_feature', 
                        choices=['fixed_feature', 'fine_tune'], 
                        help='fixed feature model or fine tune based on existing weights')
    args = parser.parse_args()

    print(f'Horovod info: size={hvd.size()}, rank={hvd.rank()}, local_rank={hvd.local_rank()}')
    if hvd.rank() == 0:
        print("data directory is: " + args.data_dir)
        # Tensorboard
        writer = SummaryWriter(f'{args.log_dir}/{run.id}')
        run.log('mode', args.mode)
    else:
        writer = None

    if args.mode == 'fixed_feature':
        model, class_names = fixed_feature_model(args.num_epochs, args.data_dir, args.learning_rate, args.momentum, writer)
    else: 
        model, class_names = fine_tune_model(args.num_epochs, args.data_dir, args.learning_rate, args.momentum, writer)

    if hvd.rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model, os.path.join(args.output_dir, 'model.pt'))
        classes_file = open(os.path.join(args.output_dir, 'class_names.pkl'), 'wb')
        pickle.dump(class_names, classes_file)
        classes_file.close()

if __name__ == "__main__":
    main()
