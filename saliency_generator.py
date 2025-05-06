## SALIENCY MAP-MAKING

import os
import numpy as np

# Custom Imports
from image_crop import crop_and_combine_images, combine_cropped, combine_cropped_100
from Saliency.imports import * # Imports needed for each model
import saliency_dataloader as sdl


# Saliency Imports
from captum.attr import Saliency
from captum.attr import visualization as viz
import matplotlib.pyplot as plt



device = "cuda:0" if torch.cuda.is_available() else "cpu"


def compute_accuracy(predictions, targets):
    correct, total = 0, 0
    correct += (predictions.cpu() == targets).sum()
    total += len(targets)
    correct = correct.cpu().data.numpy()
    return np.around(correct * 100 / total, decimals=2)


def generate_predictions(algorithm, model, ses, images, labels=None, **kwargs):
    if algorithm == "iTAML":
        model.set_saliency(True)
        outputs2 = model(images)
        pred = torch.argmax(outputs2[:, 0:SalGenArgs.class_per_task * (1 + ses)], 1, keepdim=False)
    elif algorithm == "RPSnet":
        # Reshape MNIST data
        if SalGenArgs.dataset == "mnist":
            images = images.detach().numpy().reshape(-1, 784)
            images = torch.from_numpy(images)
        outputs = model(images, kwargs['infer_path'], -1)
        _, pred = torch.max(outputs, 1)
    elif algorithm == "DGR":
        real_scores = model.forward(images)
        _, pred = real_scores[:, 0: SalGenArgs.class_per_task * (1 + ses)].max(1)
    elif algorithm == "foster" or algorithm == "memo" or algorithm == "der":
        with torch.no_grad():
            outputs = model(images)["logits"]
        pred = torch.max(outputs, dim=1)[1]

    predicted = pred.squeeze()
    if labels is not None:
        acc = compute_accuracy(predicted, labels)
        print("Acc:", acc)

    return predicted


def load_model(algorithm, dataset, ses, **kwargs):
    model = None
    scholar = None
    model_path = ""
    match algorithm:
        case "iTAML":
            model_path = f"Saliency/{algorithm}/{dataset}/session_{ses}_model_best.pth.tar"
            model = BasicNet1(kwargs['args'], 0, device=device)
        case "RPSnet":
            model_path = f"Saliency/{algorithm}/{dataset}/session_{ses}_0_model_best.pth.tar"
            if dataset == "mnist":
                model = RPS_net_mlp(kwargs['args'])
            else:
                model = RPS_net_cifar(kwargs['args'])
        case "DGR":
            model_path = f"Saliency/{algorithm}/{dataset}/split-mnist-generative-replay-r0.3_ses_{ses}"
            cnn = CNN(image_size=32, image_channel_size=1, classes=10,
                      depth=5, channel_size=1024, reducing_layers=3)
            wgan = WGAN(z_size=100, image_size=32, image_channel_size=1,
                        c_channel_size=64, g_channel_size=64)

            scholar = Scholar('', generator=wgan, solver=cnn, dataset_config=None)
        case "foster":
            model_path = f"Saliency/{algorithm}/{dataset}/{algorithm}_ses_{ses}.pth"
            #model = factory.get_model(algorithm, fosterArgs)._network
            if dataset == "mnist":
                fosterArgs['convnet_type'] = "resnet32mnist"
            model = FOSTERNet(fosterArgs, False)
            # Update the model architecture to match the task
            if ses > 0:
                for i in range(ses-1, ses + 1):
                    model.update_fc(SalGenArgs.class_per_task * (i + 1))
            else: model.update_fc(SalGenArgs.class_per_task * (ses + 1))
        case "memo":
            model_path = f"Saliency/{algorithm}/{dataset}/{algorithm}_ses_{ses}.pth"
            if dataset == "mnist":
                memoArgs['convnet_type'] = "memo_resnet32mnist"
            model = AdaptiveNet(memoArgs, False)
            # Update the model architecture to match the task
            if ses > 0:
                for i in range(ses + 1):
                    model.update_fc(SalGenArgs.class_per_task * (i + 1))
            else:
                model.update_fc(SalGenArgs.class_per_task * (ses + 1))

        case "der":
            model_path = f"Saliency/{algorithm}/{dataset}/{algorithm}_ses_{ses}.pth"
            if dataset == "mnist":
                derArgs['convnet_type'] = "resnet32mnist"
            model = DERNet(derArgs, False)
            # Update the model architecture to match the task
            if ses > 0:
                for i in range(ses + 1):
                    model.update_fc(SalGenArgs.class_per_task * (i + 1))
            else:
                model.update_fc(SalGenArgs.class_per_task * (ses + 1))


    model_data = torch.load(model_path, map_location=device, weights_only=False)

    #print(model_data.keys())
    #print(model_data['state_dict'].keys())

    #print(model.state_dict().keys())
    if algorithm == "RPSnet":
        model.load_state_dict(model_data['state_dict'])
        model.eval()
    elif algorithm == "DGR" and scholar is not None:
        scholar.load_state_dict(model_data['state'])
        model = scholar.solver
    else:
        model.load_state_dict(model_data)
        model.eval()

    return model


def create_saliency_map(model, ses, desired_classes, imgs_per_class):
    salDataloader = sdl.SalDataloader(SalGenArgs)
    if not validate:
        sal_imgs, sal_labels, classes, MEAN, STD = salDataloader.load_data(desired_classes, imgs_per_class)
    else:
        sal_imgs, sal_labels, classes, MEAN, STD = salDataloader.load_validation_data()

    # Reshape MNIST data for RPSnet
    #if SalGenArgs.algorithm == "RPSnet" and SalGenArgs.dataset == "mnist":
    #    sal_imgs = sal_imgs.detach().numpy().reshape(-1, 784)
    #    sal_imgs = torch.from_numpy(sal_imgs)

    sal_imgs, sal_labels = sal_imgs.to(device), sal_labels.to(device)

    # Add path argument for RPSnet
    if SalGenArgs.algorithm == "RPSnet":
        infer_path = generate_path(ses, SalGenArgs.dataset, SalGenArgs.args)
        predicted = generate_predictions(SalGenArgs.algorithm, model, ses, sal_imgs, sal_labels, infer_path=infer_path)
    else:
        predicted = generate_predictions(SalGenArgs.algorithm, model, ses, sal_imgs, sal_labels)

    if SalGenArgs.algorithm == "foster" or SalGenArgs.algorithm == "memo" or SalGenArgs.algorithm == "der":
        saliency = Saliency(lambda x: model(x)["logits"])
    else: saliency = Saliency(model)

    compare_grads = {}
    nrows, ncols = (2, 10) if SalGenArgs.dataset == "cifar100" else (2, 2 * imgs_per_class)
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 5))
    for ind in range(len(desired_classes) * imgs_per_class):
        compare_grads[ind] = {"grad": None, "original": None, "pred": None}
        image = sal_imgs[ind].unsqueeze(0)
        image.requires_grad = True

        # Add additional arguments for RPSnet
        if SalGenArgs.algorithm == "RPSnet":
            grads = saliency.attribute(image, target=predicted[ind], abs=False,
                                       additional_forward_args=(infer_path, -1))
        else:
            grads = saliency.attribute(image, target=predicted[ind], abs=False)

        if SalGenArgs.dataset == "mnist":
            # Reshape MNIST data from RPSnet
            if SalGenArgs.algorithm == "RPSnet":
                grads = grads.reshape(28, 28)
            else:
                grads = grads.squeeze().cpu().detach()
            squeeze_grads = torch.unsqueeze(grads, 0)
            # Save gradient for comparison
            compare_grads[ind]["grad"] = grads
            grads = np.transpose(squeeze_grads.numpy(), (1, 2, 0))
        else:
            # Save gradient for comparison
            compare_grads[ind]["grad"] = grads
            grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

        truthStr = 'Truth: ' + str(classes[sal_labels[ind]])
        predStr = 'Pred: ' + str(classes[predicted[ind]])
        print(truthStr + '\n' + predStr)

        # Reshape MNIST data from RPSnet
        if SalGenArgs.algorithm == "RPSnet" and SalGenArgs.dataset == "mnist":
            original_image = sal_imgs[ind].cpu().reshape(28, 28).unsqueeze(0)
        else:
            original_image = sal_imgs[ind].cpu()

        # Denormalization for RGB datasets
        if SalGenArgs.dataset != "mnist":
            original_image = original_image * STD[:, None, None] + MEAN[:, None, None]

        # Save image for comparison
        compare_grads[ind]["original"] = original_image
        original_image = np.transpose(original_image.detach().numpy(), (1, 2, 0))

        methods = ["original_image", "blended_heat_map"]
        signs = ["all", "absolute_value"]
        titles = ["Original Image", "Saliency Map"]
        colorbars = [False, True]

        # Check if image was misclassified
        if predicted[ind] != sal_labels[ind]:
            compare_grads[ind]["pred"] = False
            cmap = "Reds"
        else:
            compare_grads[ind]["pred"] = True
            cmap = "Blues"

        # Select row and column for saliency image
        if SalGenArgs.dataset == "cifar100" and ind > 4:
            row, col = (1, ind - 5)
        elif SalGenArgs.dataset != "cifar100" and ind > imgs_per_class - 1:
            row, col = (1, ind - imgs_per_class)
        else:
            row, col = (0, ind)

        # Generate original images and saliency images
        for i in range(2):
            # print(f"Ind: {ind}\nRow: {row}\nCol: {col}\n")
            plt_fig_axis = (fig, ax[row][(2 * col) + i])
            _ = viz.visualize_image_attr(grads, original_image,
                                         method=methods[i],
                                         sign=signs[i],
                                         fig_size=(4, 4),
                                         plt_fig_axis=plt_fig_axis,
                                         cmap=cmap,
                                         show_colorbar=colorbars[i],
                                         title=titles[i],
                                         use_pyplot=False)
            if i == 0:
                if SalGenArgs.dataset == "mnist":
                    ax[row][2 * col + i].images[0].set_cmap('gray')
                ax[row][2 * col + i].set_xlabel(truthStr)
            else:
                ax[row][2 * col + i].images[-1].colorbar.set_label(predStr)

    fig.tight_layout()
    if SalGenArgs.algorithm == "DGR" and SalGenArgs.distill:
        fig_save_path = f"SaliencyMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}/distill"
    else:
        fig_save_path = f"SaliencyMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}"
    fig.savefig(f"{fig_save_path}/Sess{ses}SalMap.png")
    plt.close()
    torch.save(compare_grads, f"{fig_save_path}/compare_dict_sess{ses}.pt")
    # fig.show()


def main(algorithm=None, dataset=None, start_sess=0):
    if not algorithm or not dataset:
        algorithm = input("Which algorithm are you using? ")
        dataset = input("Which dataset are you using? ")

    SalGenArgs.algorithm = algorithm
    SalGenArgs.dataset = dataset

    if SalGenArgs.dataset == "cifar100":
        num_sess = 10
        imgs_per_class = 1
        SalGenArgs.class_per_task = 10
        SalGenArgs.num_class = 100
    else:
        num_sess = 5
        imgs_per_class = 5
        SalGenArgs.class_per_task = 2
        SalGenArgs.num_class = 10

    model = None
    for ses in range(start_sess, num_sess):
        if SalGenArgs.algorithm == "iTAML":
            SalGenArgs.args = iTAMLArgs
            SalGenArgs.args.dataset = SalGenArgs.dataset
            SalGenArgs.args.num_class = SalGenArgs.num_class
        else:
            if SalGenArgs.dataset == "mnist":
                SalGenArgs.args = MnistArgs
            elif SalGenArgs.dataset == "cifar10" or SalGenArgs.dataset == "svhn":
                SalGenArgs.args = Cifar10Args
            elif SalGenArgs.dataset == "cifar100":
                SalGenArgs.args = Cifar100Args

        model = load_model(SalGenArgs.algorithm, SalGenArgs.dataset, ses, args=SalGenArgs.args)

        print(f'Session {ses}')
        print('#################################################################################')
        create_saliency_map(model, ses, SalGenArgs.desired_classes, imgs_per_class)
        #create_saliency_map(model, ses, SalGenArgs.dataset, SalGenArgs.desired_classes, 100)

        # create_saliency_map(model, ses, SalGenArgs.dataset, range(2), 5)
        print('\n\n')


if __name__ == '__main__':

    SalGenArgs.algorithm = "der"
    SalGenArgs.dataset = "svhn"
    SalGenArgs.distill = False
    validate = False

    params = ((0, [0, 1]),
              (1, [2, 3]),
              (2, [4, 5]),
              (3, [6, 7]),
              (4, [8, 9]))

    c100Params = ((0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                  (1, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                  (2, [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
                  (3, [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]),
                  (4, [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]),
                  (5, [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]),
                  (6, [60, 61, 62, 63, 64, 65, 66, 67, 68, 69]),
                  (7, [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]),
                  (8, [80, 81, 82, 83, 84, 85, 86, 87, 88, 89]),
                  (9, [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]))

    end_sess = 10 if SalGenArgs.dataset == "cifar100" else 5

    #'''
    #for k in range(end_sess):
    for k in range(1):

        if SalGenArgs.dataset == "cifar100":
            start_sess, SalGenArgs.desired_classes = c100Params[k]
            # start_sess, SalGenArgs.desired_classes = params[k]
        else:
            start_sess, SalGenArgs.desired_classes = params[k]

        main(SalGenArgs.algorithm, SalGenArgs.dataset, start_sess)

        if validate:
            if SalGenArgs.distill:
                save_path = f"DGRValidation/distill"
            else:
                save_path = f"DGRValidation/"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            print(save_path)

            imgs_per_sess = 2
            for j in range(imgs_per_sess):
                if SalGenArgs.distill:
                    image_path = f"DGRValidation/distill"
                else:
                    image_path = f"DGRValidation"
                image_paths = [f'{image_path}/Sess{i}SalMap.png' for i in range(start_sess, end_sess)]
                crop_and_combine_images(image_paths,
                                        f"{save_path}/Class{SalGenArgs.desired_classes[j]}Cropped.png",
                                        False,  # j+1)#(j*5)+1)
                                        (j * 5) + 4 - j)
        else:
            if SalGenArgs.algorithm == "DGR" and SalGenArgs.distill:
                save_path = f"CroppedMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}/distill"
            else:
                save_path = f"CroppedMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            imgs_per_sess = 10 if SalGenArgs.dataset == "cifar100" else 2
            for j in range(imgs_per_sess):
                if SalGenArgs.algorithm == "DGR" and SalGenArgs.distill:
                    image_path = f"SaliencyMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}/distill"
                else:
                    image_path = f"SaliencyMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}"
                image_paths = [f'{image_path}/Sess{i}SalMap.png' for i in range(start_sess, end_sess)]
                if SalGenArgs.dataset == "cifar100":
                    crop_and_combine_images(image_paths,
                                            f"{save_path}/Class{SalGenArgs.desired_classes[j]}Cropped.png",
                                            False, j + 1)
                else:
                    crop_and_combine_images(image_paths,
                                            f"{save_path}/Class{SalGenArgs.desired_classes[j]}Cropped.png",
                                            False,  # j+1)#(j*5)+1)
                                            (j * 5) + 4 - j)
    #'''

    #'''
    if SalGenArgs.algorithm == "DGR" and SalGenArgs.distill:
        crop_path = f"CroppedMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}/distill"
        combined_path = f"CroppedMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}/distill/All{SalGenArgs.dataset.capitalize()}Cropped.png"
    else:
        crop_path = f"CroppedMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}"
        combined_path = f"CroppedMaps/{SalGenArgs.algorithm}/All{SalGenArgs.dataset.capitalize()}Cropped.png"
    if SalGenArgs.dataset == "cifar100":
        cropped_paths = [f'{crop_path}/Class{i}Cropped.png' for i in range(0, 100, 10)]
    else:
        cropped_paths = [f'{crop_path}/Class{i}Cropped.png' for i in range(10)]
    if SalGenArgs.dataset == "cifar100":
        combine_cropped_100(cropped_paths, combined_path)
    else:
        combine_cropped(cropped_paths, combined_path)
    #'''
