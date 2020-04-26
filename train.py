import os
from models.c3d import *
import torch
from dataset.Sports1mDataset import Sports1mDataset
import argparse


def main():
    parser = argparse.ArgumentParser(description="C3D Sports1M Training")

    # @click.option('-l', '--learning_rate', help='Initial Learning Rate', default=3e-4)
    # @click.option('-m', '--momentum', help="Momentum for SGD", default=0.9)
    # @click.option('-b', '--batch_size', help='Batch Size', default=8)
    # @click.option('-e', '--epochs', help='Number of Epochs to Train the Model for', default=300)
    # @click.option('-s', '--subsample', help='Subsample every N frames')
    # @click.option('-n', '--num_params_factor', help='Factor for number of params', default=1.0)

def train():
    train_dataset = Sports1mDataset("dataset/sport1m_training_data.json", "dataset/training_videos")
    val_dataset = Sports1mDataset("dataset/sport1m_validation_data.json", "dataset/validation_videos")
    # model = C3D(3, out_channels, num_params_factor=1.0)

    print(len(train_dataset))
    print(len(val_dataset))

if __name__ == "__main__":
    train()