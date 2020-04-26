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
    parser.add_argument('--num_params_factor', type=float, help='Factor for number of params', default=1.0)
    parser.add_argument('--num_classes', type=int, help="Number of classes tp predict", default=487)

    args = parser.parse_args()

    train(args)

def train(args):

    lr = args.learning_rate
    momentum = args.momentum
    batch_size = args.batch_size
    epochs = args.epochs
    frame_sample = args.subsample
    num_params_factor = args.num_params_factor

    n_classes = args.num_classes

    train_dataset = Sports1mDataset("dataset/sport1m_training_data.json", "dataset/training_videos")
    val_dataset = Sports1mDataset("dataset/sport1m_validation_data.json", "dataset/validation_videos")

    model = C3D(3, n_classes, num_params_factor=num_params_factor)

    print(len(train_dataset))
    print(len(val_dataset))

if __name__ == "__main__":
    main()