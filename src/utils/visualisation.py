import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colour-blind-friendly palette, in the project's standard order.
# Index it directly (OKABE_ITO[i]) to colour series by position, or call
# set_okabe_ito_cycle() to make it the default matplotlib colour cycle.
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']


def set_okabe_ito_cycle(ax=None):
    """Use the Okabe-Ito palette as the matplotlib colour cycle.

    With no argument, sets it globally via rcParams for all later plots;
    pass an Axes to set it only for that Axes.
    """
    cycler = plt.cycler(color=OKABE_ITO)
    if ax is None:
        plt.rcParams['axes.prop_cycle'] = cycler
    else:
        ax.set_prop_cycle(cycler)


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.xlabel('Time Steps (Days)')
    plt.ylabel('Density Features')
    plt.show()