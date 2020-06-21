import os
from models.c3d import *
from models.utils.lr_scheduler import *
import torch
from dataset.Sports1mDataset import Sports1mDataset
import argparse
import logging
from tqdm import tqdm



def main():
    parser = argparse.ArgumentParser(description="C3D Sports1M Training")

    parser.add_argument('--downsample_fps', type=int, help="Downsample FPS for data loading", default=3)

    parser.add_argument('--learning_rate', type=float, help='Initial Learning Rate', default=3e-4)
    parser.add_argument('--batch_size', type=int, help='Batch Size', default=8)
    parser.add_argument('--epochs', type=int, help='Number of Epochs to Train the Model for', default=300)
    parser.add_argument('--num_classes', type=int, help="Number of classes tp predict", default=487)

    args = parser.parse_args()

    train(args)

def createLogger():
    console_logging_format = "%(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    return logger

def train(args):

    downsample_fps = args.downsample_fps

    lr = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    frame_sample = args.subsample

    n_classes = args.num_classes

    logger = createLogger()
    logger.info("Starting training...")

    train_dataset = Sports1mDataset("dataset/cleaned_dataset/sports1m_training_cleaned.json", "dataset/training_videos", downsample_fps = downsample_fps)
    train_dataset.filter_videos('duration', lambda x: x <= 120)

    #val_dataset = Sports1mDataset("dataset/sport1m_validation_data.json", "dataset/validation_videos")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    logger.info("Datasets modules built...")

    model = C3D(3, n_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = LRScheduler(optimizer, init_lr = 3e-3, iters_per_step = 150000, decay_factor = 0.5)

    logger.info("Experiment modules built...")

    print("Initializing training...")
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, epoch + 1)
        

        print("Epoch {} Statistics:".format(epoch + 1))
        print("  Train Loss: {}; Train Accuracy {}".format(train_loss, train_acc))
        break


def train_epoch(model, train_loader, optimizer, scheduler, epoch):
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
        if scheduler is not None:
            scheduler.step()

        # Compute training accuracy
        pred = output.argmax(dim=1, keepdim=True)

        curr_accuracy = pred.eq(target.view_as(pred)).sum().item() / float(target.shape[0])
        total_accuracy += curr_accuracy
        total_loss += loss.item()

        avg_loss = total_loss / (batch_idx + 1)
        avg_accuracy = total_accuracy / (batch_idx + 1)
        loader.set_description("TRAIN - Avg Loss: %.4f; Curr. Accuracy: %.6f; Curr Learning Rate: %.6f" % (
                                avg_loss, curr_accuracy*100, optimizer.param_groups[0]['lr']
                                ))

    
    return avg_loss, avg_accuracy

if __name__ == "__main__":
    pass
    #main()