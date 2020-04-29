import os
from models.c3d import *
import torch
from dataset.Sports1mDataset import Sports1mDataset
import argparse


def main():
    parser = argparse.ArgumentParser(description="C3D Sports1M Training")

    parser.add_argument('--learning_rate', type=float, help='Initial Learning Rate', default=3e-4)
    parser.add_argument('--momentum', type=float, help="Momentum for SGD", default=0.9)
    parser.add_argument('--batch_size', type=int, help='Batch Size', default=8)
    parser.add_argument('--epochs', type=int, help='Number of Epochs to Train the Model for', default=300)
    parser.add_argument('--subsample', type=int, help='Subsample every N frames')
    parser.add_argument('--num_classes', type=int, help="Number of classes tp predict", default=487)

    args = parser.parse_args()

    train(args)

def train(args):

    lr = args.learning_rate
    momentum = args.momentum
    batch_size = args.batch_size
    epochs = args.epochs
    frame_sample = args.subsample

    n_classes = args.num_classes

    train_dataset = Sports1mDataset("dataset/sport1m_training_data.json", "dataset/training_videos", subsample=frame_sample)
    val_dataset = Sports1mDataset("dataset/sport1m_validation_data.json", "dataset/validation_videos", subsample=frame_sample)
    print("built datasets...")

    model = C3D(3, n_classes).cuda()
    print("built model...")

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    print("built data loaders...")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("built optimizer...")

    print("Initializing training...")
    train_epoch(model, train_loader, optimizer, 1)
    # for epoch in range(epochs):
    #     train_epoch(model, train_loader, optimizer, epoch+1)
    #     break


def train_epoch(model, train_loader, optimizer, epoch):
    model.train()

    total_correct = torch.tensor(0.0) #.cuda()
    total_examples = torch.tensor(0.0) #.cuda()

    for batch_idx, example in enumerate(train_loader):
        # Metrics
        correct = torch.tensor(0.0).cuda()
        examples = torch.tensor(0.0).cuda()

        # Model data setup
        data = example['video'] # B, F, H, W, C
        target = example['class'] 

        data, target = data.cuda(), target.cuda()
        data = torch.transpose(data, 1, 4) #B, C, H, W, F
        data = torch.transpose(data, 2, 4) #B, C, F, W, H

        # Compute the forward pass
        optimizer.zero_grad()
        
        target = target.view(-1,)

        print("data", data.shape)
        print("label", target.shape)


        print("forward pass in model...")

        output = model(data)
        print(output.shape)
        loss = torch.nn.functional.cross_entropy(output, target)
        break

        # Compute training accuracy
        # pred = output.argmax(dim=1, keepdim=True)
        # correct += pred.eq(target.view_as(pred)).sum()
        # examples += target.size()[0]

        # optimizer.step()

        # # Reduction as sum
        # torch.distributed.reduce(examples, dst=0)
        # torch.distributed.reduce(correct, dst=0)
        # total_correct += correct
        # total_examples += examples

if __name__ == "__main__":
    main()