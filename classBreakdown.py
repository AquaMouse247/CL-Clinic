import torch
import numpy as np
from captum.attr import Saliency
from Saliency.imports import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from saliency_generator import SalGenArgs, iTAMLArgs, MnistArgs
import saliency_dataloader as sdl
#from attack.data_bd_V5 import PoisonedCIFAR10, PoisonedCIFAR10_train, SubsetWithAttributes

def main(dataset):
    SalGenArgs.args = iTAMLArgs

    if dataset == "cifar100":
        SalGenArgs.class_per_task = 10
        SalGenArgs.num_class = 100
    else:
        SalGenArgs.class_per_task = 2
        SalGenArgs.num_class = 10

    SalGenArgs.algorithm = "iTAML"
    SalGenArgs.dataset = dataset
    SalGenArgs.args.dataset = SalGenArgs.dataset
    SalGenArgs.args.num_class = SalGenArgs.num_class


    sal_dataloader = sdl.SalDataloader(SalGenArgs)
    print(f"-------{dataset.upper()} Dataset-------")

    for i in range(SalGenArgs.num_class):
        dataloader = sal_dataloader.load_data([i])
        samples = len(dataloader.dataset)
        print(f"Class {i} Samples: {samples}")


if __name__ == '__main__':
  algorithm = "iTAML"
  dataset = "cifar100"
  main(dataset)