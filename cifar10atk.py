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


def main(poison=False):
    SalGenArgs.args = iTAMLArgs
    SalGenArgs.class_per_task = 2
    SalGenArgs.num_class = 10

    SalGenArgs.algorithm = "iTAML"
    SalGenArgs.dataset = "cifar10"
    SalGenArgs.args.dataset = SalGenArgs.dataset
    SalGenArgs.args.num_class = SalGenArgs.num_class

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load model
    #model_path = f"attack/session_4_model_best.pth.tar"
    model_path = f"Saliency/iTAML/cifar10/session_4_model_best.pth.tar"
    model = BasicNet1(SalGenArgs.args, 0, device=device)
    model_data = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(model_data)
    model.eval()

    sal_dataloader = sdl.SalDataloader(SalGenArgs)

    if poison:
        poisonDataset = torch.load('poison_datasets/test_poisoned_V2.pth')

        # Create the transform
        Transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        poisonDataset.transform = Transform

        # Create a DataLoader for the subset
        saliencyLoader = DataLoader(poisonDataset, batch_size=256)
    else:
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])

        saliency_set = datasets.CIFAR10(root=f"Datasets/cifar10/", train=False,
                                        download=True,
                                        transform=transforms.Compose(
                                            [transforms.ToTensor(),
                                             transforms.Normalize(mean, std)]))
        saliencyLoader = DataLoader(saliency_set, batch_size=256)

    #for images, labels in saliencyLoader:
    #images, labels = next(iter(saliencyLoader))
    #predicted = generate_predictions("iTAML", model, 4, images, labels=labels, args=SalGenArgs.args)

    correct_total = 0
    total_samples = 0

    from collections import defaultdict
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    from tqdm import tqdm
    for images, labels in tqdm(saliencyLoader):
        images = images.to(device)
        labels = labels.to(device)

        outputs2, outputs = model(images)
        preds = torch.argmax(outputs2[:, 0:10], dim=1)

        # Overall accuracy
        correct_total += (preds == labels).sum().item()
        total_samples += labels.size(0)

        # Per-class accuracy
        for label, pred in zip(labels, preds):
            class_total[label.item()] += 1
            if pred.item() == label.item():
                class_correct[label.item()] += 1

    overall_acc = correct_total / total_samples
    per_class_acc = {
        cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
        for cls in range(10)
    }

    print(f"-------CIFAR10 Dataset-------")
    print(f"Overall Accuracy: {overall_acc * 100:.2f}%")
    for cls in range(10):
        print(f"Acc Class {cls}: {per_class_acc[cls] * 100:.2f}%")


if __name__ == '__main__':
    main(True)
    pass


