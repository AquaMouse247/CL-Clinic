# iTAML Imports
from basic_net import *


# RPSnet Imports
class iTAMLArgs:
    checkpoint = "results/cifar100/meta_mnist_T5_47"
    savepoint = "models/" + "/".join(checkpoint.split("/")[1:])
    data_path = "../Datasets/MNIST/"
    num_class = 10
    class_per_task = 2
    num_task = 5
    test_samples_per_class = 1000
    dataset = "mnist"
    optimizer = 'sgd'

    epochs = 20
    lr = 0.1
    train_batch = 256
    test_batch = 256
    workers = 16
    sess = 0
    schedule = [5, 10, 15]
    gamma = 0.5
    random_classes = False
    validation = 0
    memory = 2000
    mu = 1
    beta = 0.5
    r = 1

'''
def meta_test(self, model, memory, inc_dataset):

    # switch to evaluate mode
    model.eval()

    meta_models = []
    base_model = copy.deepcopy(model)
    class_acc = {}
    meta_task_test_list = {}
    for task_idx in range(self.args.sess + 1):

        memory_data, memory_target = memory
        memory_data = np.array(memory_data, dtype="int32")
        memory_target = np.array(memory_target, dtype="int32")

        mem_idx = np.where((memory_target >= task_idx * self.args.class_per_task) & (
                    memory_target < (task_idx + 1) * self.args.class_per_task))[0]
        meta_memory_data = memory_data[mem_idx]
        meta_memory_target = memory_target[mem_idx]
        meta_model = copy.deepcopy(base_model)

        meta_loader = inc_dataset.get_custom_loader_idx(meta_memory_data, mode="train", batch_size=64)

        meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=0.0, amsgrad=False)

        meta_model.train()

        ai = self.args.class_per_task * task_idx
        bi = self.args.class_per_task * (task_idx + 1)
        bb = self.args.class_per_task * (self.args.sess + 1)
        print("Training meta tasks:\t", task_idx)

        ## Instead of retraining on the original data, it retrains
        ## based on the results of the already trained model.
        # META training
        if (self.args.sess != 0):
            for ep in range(1):
                bar = Bar('Processing', max=len(meta_loader))
                for batch_idx, (inputs, targets) in enumerate(meta_loader):
                    targets_one_hot = torch.FloatTensor(inputs.shape[0], (task_idx + 1) * self.args.class_per_task)
                    targets_one_hot.zero_()
                    targets_one_hot.scatter_(1, targets[:, None], 1)
                    target_set = np.unique(targets)

                    if self.use_cuda:
                        inputs, targets_one_hot, targets = inputs.cuda(), targets_one_hot.cuda(), targets.cuda()
                    inputs, targets_one_hot, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(
                        targets_one_hot), torch.autograd.Variable(targets)

                    _, outputs = meta_model(inputs)
                    class_pre_ce = outputs.clone()
                    class_pre_ce = class_pre_ce[:, ai:bi]
                    class_tar_ce = targets_one_hot.clone()

                    loss = F.binary_cross_entropy_with_logits(class_pre_ce, class_tar_ce[:, ai:bi])

                    meta_optimizer.zero_grad()
                    loss.backward()
                    meta_optimizer.step()
                    bar.suffix = '({batch}/{size})  Total: {total:} | Loss: {loss:.4f}'.format(
                        batch=batch_idx + 1,
                        size=len(meta_loader),
                        total=bar.elapsed_td,
                        loss=loss)
                    bar.next()
                bar.finish()

        # META testing with given knowledge on task
        meta_model.eval()
        for cl in range(self.args.class_per_task):
            class_idx = cl + self.args.class_per_task * task_idx
            loader = inc_dataset.get_custom_loader_class([class_idx], mode="test", batch_size=10)

            for batch_idx, (inputs, targets) in enumerate(loader):
                targets_task = targets - self.args.class_per_task * task_idx

                if self.use_cuda:
                    inputs, targets_task = inputs.cuda(), targets_task.cuda()
                inputs, targets_task = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_task)

                _, outputs = meta_model(inputs)

                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets_task.cuda()
                inputs, targets_task = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_task)

                pred = torch.argmax(outputs[:, ai:bi], 1, keepdim=False)
                pred = pred.view(1, -1)
                correct = pred.eq(targets_task.view(1, -1).expand_as(pred)).view(-1)

                correct_k = float(torch.sum(correct).detach().cpu().numpy())

                for i, p in enumerate(pred.view(-1)):
                    key = int(p.detach().cpu().numpy())
                    key = key + self.args.class_per_task * task_idx
                    if (correct[i] == 1):
                        if (key in class_acc.keys()):
                            class_acc[key] += 1
                        else:
                            class_acc[key] = 1

        #           META testing - no knowledge on task
        meta_model.eval()
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            _, outputs = meta_model(inputs)
            outputs_base, _ = self.model(inputs)
            task_ids = outputs

            task_ids = task_ids.detach().cpu()
            outputs = outputs.detach().cpu()
            outputs = outputs.detach().cpu()
            outputs_base = outputs_base.detach().cpu()

            bs = inputs.size()[0]
            for i, t in enumerate(list(range(bs))):
                j = batch_idx * self.args.test_batch + i
                output_base_max = []
                for si in range(self.args.sess + 1):
                    sj = outputs_base[i][si * self.args.class_per_task:(si + 1) * self.args.class_per_task]
                    sq = torch.max(sj)
                    output_base_max.append(sq)

                task_argmax = np.argsort(outputs[i][ai:bi])[-5:]
                task_max = outputs[i][ai:bi][task_argmax]

                if (j not in meta_task_test_list.keys()):
                    meta_task_test_list[j] = [[task_argmax, task_max, output_base_max, targets[i]]]
                else:
                    meta_task_test_list[j].append([task_argmax, task_max, output_base_max, targets[i]])
        del meta_model

    acc_task = {}
    for i in range(self.args.sess + 1):
        acc_task[i] = 0
        for j in range(self.args.class_per_task):
            try:
                acc_task[i] += class_acc[i * self.args.class_per_task + j] / self.args.sample_per_task_testing[
                    i] * 100
            except:
                pass
    print("\n".join([str(acc_task[k]).format(".4f") for k in acc_task.keys()]))
    print(class_acc)

    with open(self.args.savepoint + "/meta_task_test_list_" + str(task_idx) + ".pickle", 'wb') as handle:
        pickle.dump(meta_task_test_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return acc_task

def get_memory(self, memory, for_memory, seed=1):
    random.seed(seed)
    memory_per_task = self.args.memory // ((self.args.sess + 1) * self.args.class_per_task)
    self._data_memory, self._targets_memory = np.array([]), np.array([])
    mu = 1

    # update old memory
    if (memory is not None):
        data_memory, targets_memory = memory
        data_memory = np.array(data_memory, dtype="int32")
        targets_memory = np.array(targets_memory, dtype="int32")
        for class_idx in range(self.args.class_per_task * (self.args.sess)):
            idx = np.where(targets_memory == class_idx)[0][:memory_per_task]
            self._data_memory = np.concatenate([self._data_memory, np.tile(data_memory[idx], (mu,))])
            self._targets_memory = np.concatenate([self._targets_memory, np.tile(targets_memory[idx], (mu,))])

    # add new classes to the memory
    new_indices, new_targets = for_memory

    new_indices = np.array(new_indices, dtype="int32")
    new_targets = np.array(new_targets, dtype="int32")
    for class_idx in range(self.args.class_per_task * (self.args.sess),
                           self.args.class_per_task * (1 + self.args.sess)):
        idx = np.where(new_targets == class_idx)[0][:memory_per_task]
        self._data_memory = np.concatenate([self._data_memory, np.tile(new_indices[idx], (mu,))])
        self._targets_memory = np.concatenate([self._targets_memory, np.tile(new_targets[idx], (mu,))])

    print(len(self._data_memory))
    return list(self._data_memory.astype("int32")), list(self._targets_memory.astype("int32"))
'''

# RPSnet Imports
from rps_net import RPS_net_mlp, RPS_net_cifar, generate_path


class MnistArgs:
    epochs = 10
    checkpoint = "results/mnist/RPS_net_mnist"
    savepoint = "results/mnist/pathnet_mnist"
    dataset = "MNIST"
    num_class = 10
    class_per_task = 2
    M = 8
    L = 9
    N = 1
    lr = 0.001
    train_batch = 128
    test_batch = 128
    workers = 16
    resume = False
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    schedule = [6, 8, 16]
    gamma = 0.5
    rigidness_coff = 2.5
    jump = 1


class Cifar10Args:
    epochs = 10
    #    epochs = 2
    checkpoint = "results/cifar10/RPS_net_cifar10"
    savepoint = "results/cifar10/pathnet_cifar10"
    dataset = "CIFAR10"
    num_class = 10
    class_per_task = 2
    M = 8
    L = 9
    N = 1
    lr = 0.001
    train_batch = 128
    test_batch = 128
    workers = 16
    resume = False
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    schedule = [6, 8, 16]
    gamma = 0.5
    rigidness_coff = 2.5
    jump = 1


class Cifar100Args:
    checkpoint = "results/cifar100/RPS_CIFAR_M8_J1"
    labels_data = "prepare/cifar100_10.pkl"
    savepoint = ""

    num_class = 100
    class_per_task = 10
    M = 8
    jump = 2
    rigidness_coff = 2.5
    dataset = "CIFAR"

    epochs = 100
    epochs = 1
    L = 9
    N = 1
    lr = 0.001
    train_batch = 128
    test_batch = 128
    workers = 16
    resume = False
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    schedule = [20, 40, 60, 80]
    gamma = 0.5


# DGRv2 Imports
from Saliency.DGR.models import CNN, WGAN
from Saliency.DGR.dgr import Scholar
#import copy, math, random

class SalGenArgs:
    algorithm = "RPSnet"
    dataset = "mnist"
    args = None
    desired_classes = [0, 1]
    class_per_task = 2
    num_class = 10
    distill = False

# PyCIL Imports
from Saliency.PyCIL.models import foster
from Saliency.PyCIL.utils import factory
from Saliency.PyCIL.utils.inc_net import FOSTERNet, AdaptiveNet, DERNet
fosterArgs = {'config': './exps/fostertest.json', 'prefix': 'cil', 'dataset': 'cifar100',
              'memory_size': 2000, 'memory_per_class': 20, 'fixed_memory': True,
              'shuffle': False, 'init_cls': 10, 'increment': 10, 'model_name': 'foster',
              'convnet_type': 'resnet32', 'device': ['0'], 'seed': [1993], 'beta1': 0.96,
              'beta2': 0.97, 'oofc': 'ft', 'is_teacher_wa': False, 'is_student_wa': False,
              'lambda_okd': 1, 'wa_value': 1, 'init_epochs': 1, 'init_lr': 0.1,
              'init_weight_decay': 0.0005, 'boosting_epochs': 1, 'compression_epochs': 1,
              'lr': 0.1, 'batch_size': 128, 'weight_decay': 0.0005, 'num_workers': 8, 'T': 2}

memoArgs = {"prefix": "benchmark", "dataset": "cifar100", "memory_size": 2000,
            "memory_per_class":20, "fixed_memory": False, "shuffle": False, "init_cls": 10,
            "increment": 10, "model_name": "memo", "convnet_type": "memo_resnet32",
            "train_base": True, "train_adaptive": False, "debug": False, "skip": False,
            "device": ["0"], "seed":[1993], "scheduler": "steplr", "init_epoch": 200,
            "t_max": None, "init_lr" : 0.1, "init_weight_decay" : 5e-4, "init_lr_decay" : 0.1,
            "init_milestones" : [60,120,170], "milestones" : [80,120,150], "epochs": 170,
            "lrate" : 0.1, "batch_size" : 128, "weight_decay" : 2e-4, "lrate_decay" : 0.1,
            "alpha_aux" : 1.0}

derArgs = {"prefix": "reproduce", "dataset": "cifar10", "memory_size": 2000,
           "memory_per_class": 20, "fixed_memory": False, "shuffle": False,"init_cls": 2,
           "increment": 2, "model_name": "der", "convnet_type": "resnet32",
           "device": ["0"], "seed": [1993]}

