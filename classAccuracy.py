import torch
import numpy as np
from captum.attr import Saliency
from Saliency.imports import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from saliency_generator import load_model, generate_predictions
from saliency_generator import SalGenArgs, iTAMLArgs, MnistArgs
import saliency_dataloader as sdl
from rps_net import generate_path
#from attack.data_bd_V5 import PoisonedCIFAR10, PoisonedCIFAR10_train, SubsetWithAttributes

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_task(label, sessions) -> int:
    task = None
    if sessions == 5:
        match label:
            case 0 | 1: task = 0
            case 2 | 3: task = 1
            case 4 | 5: task = 2
            case 6 | 7: task = 3
            case 8 | 9: task = 4
    return task

def main(algorithm, dataset, sessions):
    if algorithm == "iTAML":
        SalGenArgs.args = iTAMLArgs
    else:
        SalGenArgs.args = MnistArgs

    if dataset == "cifar100":
        SalGenArgs.class_per_task = 10
        SalGenArgs.num_class = 100
    else:
        SalGenArgs.class_per_task = 2
        SalGenArgs.num_class = 10

    SalGenArgs.algorithm = algorithm
    SalGenArgs.dataset = dataset
    SalGenArgs.args.dataset = SalGenArgs.dataset
    SalGenArgs.args.num_class = SalGenArgs.num_class

    session_acc = {}

    for ses in range(sessions):
        model = load_model(algorithm, dataset, ses, args=SalGenArgs.args)
        sal_dataloader = sdl.SalDataloader(SalGenArgs)

        accLoader = sal_dataloader.load_data(range(SalGenArgs.class_per_task*(ses+1)))

        if algorithm == "RPSnet":
            infer_path = generate_path(ses, SalGenArgs.dataset, SalGenArgs.args)


        correct_total = 0
        total_samples = 0

        from collections import defaultdict
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        task_correct = defaultdict(int)
        task_total = defaultdict(int)

        from tqdm import tqdm
        for images, labels in tqdm(accLoader):
            images = images.to(device)
            labels = labels.to(device)

            if algorithm == "RPSnet":
                preds = generate_predictions(algorithm, model, ses, images, infer_path=infer_path)
            else:
                preds = generate_predictions(algorithm, model, ses, images, args=SalGenArgs.args)

            # Overall accuracy
            correct_total += (preds == labels).sum().item()
            total_samples += labels.size(0)

            # Per-class accuracy
            '''
            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                if pred.item() == label.item():
                    class_correct[label.item()] += 1
            #'''

            # Per-task accuracy
            #'''
            for label, pred in zip(labels, preds):
                accTask = get_task(label.item(), sessions)
                task_total[accTask] += 1
                if pred.item() == label.item():
                    task_correct[accTask] += 1
            #'''
        if ses == sessions-1: overall_acc = correct_total / total_samples
        per_task_acc = {
            task: task_correct[task] / task_total[task] if task_total[task] > 0 else 0.0
            for task in range(ses+1)
        }
        session_acc[ses] = per_task_acc

    # Create Acc Matrix
    matrix = []
    for ses in range(sessions):
        s = []
        for task in range(ses+1):
            s.append(f"{session_acc[ses][task] * 100:.2f}")
        matrix.append(s)

    print(f"-------{dataset.upper()} Dataset-------")
    print(f"Overall Accuracy: {overall_acc * 100:.2f}%")
    print("Task Accuracy per Session:")

    # Print Matrix for task-wise
    #'''
    #print(matrix)
    np_acctable = np.zeros([ses + 1, ses + 1])
    for idxx, line in enumerate(matrix):
        idxy = len(line)
        np_acctable[idxx, :idxy] = np.array(line)
    np_acctable = np_acctable.T
    print(np_acctable)
    #'''

    #for task in range(SalGenArgs.class_per_task*(ses+1)):
    #    print(f"Acc Task {task}: {per_task_acc[task] * 100:.2f}%")



if __name__ == '__main__':
  algorithm = "iTAML"
  dataset = "mnist"
  sessions = 10 if dataset == "cifar100" else 5
  main(algorithm, dataset, sessions)


