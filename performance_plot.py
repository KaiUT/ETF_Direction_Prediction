
import matplotlib.pyplot as plt

# function
def plot_accuracy(data_x, data_y, xlabels, file_path):
    r"""

    """
    plt.figure(figsize=(10,4.5))
    plt.bar(data_x, data_y, align='center', alpha=0.5)
    plt.axhline(y=0.5, linewidth=2, color='grey', linestyle=':')
    plt.xticks(data_x,xlabels)
    plt.xlabel('Models')
    plt.ylim([0.3, 0.6])
    plt.ylabel('Accuracy')
    plt.savefig(file_path)
    plt.close()




x = range(19)[1:]
xlabels = [str(i) for i in x]

# XLE
XLE_accuracy = [0.5458, 0.5737, 0.5378, 0.4580, 0.5538, 0.5020, 0.4462,
                0.5498, 0.4422, 0.4860, 0.4821, 0.5618, 0.4582, 0.5339,
                0.4781, 0.4462, 0.5378, 0.4502]

# XLU
XLU_accuracy = [0.5900, 0.5817, 0.5936, 0.4064, 0.5936, 0.5976, 0.3904,
                0.5857, 0.4821, 0.4582, 0.4661, 0.5857, 0.4064, 0.4223,
                0.5339, 0.3904, 0.4024, 0.4263]

# XLK
XLK_accuracy = [0.5418, 0.5450, 0.4980, 0.4502, 0.4502, 0.4382, 0.4382,
                0.4462, 0.5378, 0.5339, 0.5100, 0.5219, 0.4502, 0.4741,
                0.5418, 0.4382, 0.4661, 0.4701]

# XLB
XLB_accuracy = [0.5020, 0.5060, 0.5100, 0.5139, 0.4821, 0.5139, 0.5339, 0.4701,
                0.4781, 0.5020, 0.5060, 0.4741, 0.5139, 0.4900, 0.5179, 0.5339,
                0.4940, 0.5418]

# XLP
XLP_accuracy = [0.5299, 0.5378, 0.5179, 0.5179, 0.5020, 0.5100, 0.5418, 0.5179,
                0.5179, 0.5020, 0.5020, 0.5179, 0.5179, 0.5020, 0.5259, 0.5418,
                0.5020, 0.5458]

# XLY
XLY_accuracy = [0.5458, 0.5458, 0.5458, 0.5458, 0.4781, 0.5458, 0.5498, 0.4462,
                0.4622, 0.5458, 0.5378, 0.5458, 0.5458, 0.5299, 0.4103, 0.5498,
                0.5339, 0.5259]

# XLI
XLI_accuracy = [0.5538, 0.5458, 0.5139, 0.4502, 0.5259, 0.5538, 0.4382, 0.5299,
                0.4861, 0.4542, 0.4502, 0.5538, 0.4502, 0.4940, 0.5219, 0.4382,
                0.4980, 0.5538]

# XLV
XLV_accuracy = [0.5179, 0.5139, 0.4940, 0.5060, 0.5418, 0.5100, 0.5060, 0.5418,
                0.4900, 0.4980, 0.4940, 0.5378, 0.5060, 0.5100, 0.5060, 0.5060,
                0.5100, 0.4940]

# SPY
SPY_accuracy = [0.5378, 0.5339, 0.5498, 0.5498, 0.4622, 0.5498, 0.5538, 0.4582,
                0.5418, 0.4622, 0.4622, 0.5657, 0.5498, 0.4940, 0.5299, 0.5538,
                0.4900, 0.4462]


# plot
path = '/Users/kailiu/GitProjects/ETF_Direction_Prediction/results/'
plot_accuracy(x, XLE_accuracy, xlabels, path+'XLE_accuracy.pdf')
plot_accuracy(x, XLU_accuracy, xlabels, path+'XLU_accuracy.pdf')
plot_accuracy(x, XLK_accuracy, xlabels, path+'XLK_accuracy.pdf')
plot_accuracy(x, XLB_accuracy, xlabels, path+'XLB_accuracy.pdf')
plot_accuracy(x, XLP_accuracy, xlabels, path+'XLP_accuracy.pdf')
plot_accuracy(x, XLY_accuracy, xlabels, path+'XLY_accuracy.pdf')
plot_accuracy(x, XLI_accuracy, xlabels, path+'XLI_accuracy.pdf')
plot_accuracy(x, XLV_accuracy, xlabels, path+'XLV_accuracy.pdf')
plot_accuracy(x, SPY_accuracy, xlabels, path+'SPY_accuracy.pdf')
