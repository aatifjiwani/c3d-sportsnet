import os
from models.c3d import *
import torch
from dataset.Sports1mDataset import Sports1mDataset
import argparse
import logging
from tqdm import tqdm



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

def createLogger():
    console_logging_format = "%(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    return logger

def train(args):

    lr = args.learning_rate
    momentum = args.momentum
    batch_size = args.batch_size
    epochs = args.epochs
    frame_sample = args.subsample

    n_classes = args.num_classes

    logger = createLogger()
    logger.info("Starting training...")

    train_dataset = Sports1mDataset("dataset/sport1m_training_data.json", "dataset/training_videos", subsample=frame_sample)
    val_dataset = Sports1mDataset("dataset/sport1m_validation_data.json", "dataset/validation_videos", subsample=frame_sample)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=None)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=None)

    logger.info("Datasets modules built...")

    model = C3D(3, n_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info("Experiment modules built...")

    print("Initializing training...")
    train_epoch(model, train_loader, optimizer, 1)


def train_epoch(model, train_loader, optimizer, epoch):
    model.train()

    total_loss = 0
    total_accuracy = 0

    loader = tqdm(train_loader)
    loader.set_description("Epoch {} - Training".format(epoch))

    for batch_idx, example in enumerate(loader):
        # Model data setup
        data = example['video'] # B, F, H, W, C
        target = example['class'] 

        data, target = data.cuda(), target.cuda()
        data = torch.transpose(data, 1, 4) #B, C, H, W, F
        data = torch.transpose(data, 2, 4) #B, C, F, W, H
        target = target.view(-1,) #B,

        # Compute the forward pass
        optimizer.zero_grad()

        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)

        loss.backward()
        optimizer.step()

        # Compute training accuracy
        pred = output.argmax(dim=1, keepdim=True)
        total_accuracy += pred.eq(target.view_as(pred)).sum().item() / float(target.shape[0])
        total_loss += loss.item()

        avg_loss = total_loss / (batch_idx + 1)
        avg_accuracy = total_accuracy / (batch_idx + 1)
        loader.set_description("TRAIN - Avg Loss: %.4f; Avg. Accuracy: %.6f;" % (avg_loss, avg_accuracy*100) )

if __name__ == "__main__":
    main()