import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import scipy.io
import matplotlib.pyplot as plt


def jsd(algorithm, dataset, printpar=False, plot=False):

    if dataset == "cifar100":
        paths = [['_ses_0_XA_dcts.mat', '_ses_9_XA_dcts.mat'],
                 ['_ses_0_XB_dcts.mat', '_ses_9_XB_dcts.mat'],
                 ['_ses_0_XA_dcts.mat', '_ses_0_XB_dcts.mat'],
                 ['_ses_9_XA_dcts.mat', '_ses_9_XB_dcts.mat']]

        titles = ["XA_0 vs XA_9 (Hyp = Low)", "XB_0 vs XB_9 (Hyp = Low)",
                  "XA_0 vs XB_0 (Hyp = High)", "XA_9 vs XB_9 (Hyp = High)"]
    else:
        paths = [['_ses_0_XA_dcts.mat','_ses_4_XA_dcts.mat'],
                 ['_ses_0_XB_dcts.mat','_ses_4_XB_dcts.mat'],
                 ['_ses_0_XA_dcts.mat','_ses_0_XB_dcts.mat'],
                 ['_ses_4_XA_dcts.mat','_ses_4_XB_dcts.mat']]

        titles = ["XA_0 vs XA_4 (Hyp = Low)", "XB_0 vs XB_4 (Hyp = Low)",
                  "XA_0 vs XB_0 (Hyp = High)", "XA_4 vs XB_4 (Hyp = High)"]

    distances = []

    for i in range(len(paths)):
        class1_data = scipy.io.loadmat(f'matlab/{algorithm}/{dataset}/{algorithm}{paths[i][0]}')
        fns1 = list(class1_data.keys())
        data1 = class1_data[fns1[-1]].flatten()

        class2_data = scipy.io.loadmat(f'matlab/{algorithm}/{dataset}/{algorithm}{paths[i][1]}')
        fns2 = list(class2_data.keys())
        data2 = class2_data[fns2[-1]].flatten()


        # Generate x points for PDF
        combined_data = np.concatenate((data1, data2))
        xmin = np.min(combined_data) - 3 * np.std(combined_data)
        xmax = np.max(combined_data) + 3 * np.std(combined_data)
        num_points = 500
        x_common = np.linspace(xmin, xmax, num_points)

        # Generate PDFs
        kde1 = gaussian_kde(data1)
        pdf1 = kde1(x_common)
        kde2 = gaussian_kde(data2)
        pdf2 = kde2(x_common)

        js_distance = jensenshannon(pdf1, pdf2)
        # The Jensen-Shannon Divergence is the square of the distance
        js_divergence = js_distance**2

        if printpar:
            if i == 0:
                print("Jensen-Shannon Distances (Metric)")
                print("----------------------------------")
            print(f"{titles[i]}: {js_distance:.4f}")
            #print("Jensen-Shannon Divergence:", js_divergence)

        if plot:
            # 5. Optional: Plot the estimated PDFs
            plt.figure(figsize=(10, 6))
            plt.plot(x_common, pdf1, label='PDF 1')
            plt.plot(x_common, pdf2, label='PDF 2')
            plt.xlabel('Value')
            plt.ylabel('Probability Density')
            plt.title('Estimated Probability Density Functions')
            plt.legend()
            plt.grid(True)
            plt.show()

        distances.append(js_distance)

    return distances[0], distances[1], distances[2], distances[3]


def create_jsd_table(dataset):
    from tabulate import tabulate

    if dataset == "cifar100":
        headers = ["Algorithm", "XA_0 vs XA_9", "XA_0 vs XB_0", "XA_9 vs XB_9"]
    else:
        headers = ["Algorithm", "XA_0 vs XA_4", "XA_0 vs XB_0", "XA_4 vs XB_4"]

    algs = ["iTAML", "RPSnet", "dgr", "foster", "memo", "der"]
    if dataset != "mnist":
        algs.remove("dgr")
    jsdTable = []

    for alg in algs:
        dist1, dist2, dist3, dist4 = jsd(alg, dataset)
        entry = [alg, dist1.round(4), dist3.round(4), dist4.round(4)]
        jsdTable.append(entry)


    print(tabulate(jsdTable, headers=headers, tablefmt="fancy_grid"))

if __name__ == '__main__':
    algorithm = "iTAML"
    dataset = "mnist"
    #jsd(algorithm, dataset)
    #create_jsd_table(dataset)
    jsd(algorithm, dataset, printpar=True, plot=True)




















# 3. Handle potential negative values
#pdf1[pdf1 < 0] = 0
#pdf2[pdf2 < 0] = 0