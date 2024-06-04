

'''
Submodules:
    data: data wrangling
    plot: plot formatting
    nn: neural network training - but also contains other (commented) code for, for example, getting data loaders

Access submodules:
    data: root (eg aku_utils.na(df))
    plot: aku_utils.plot (eg aku_utils.plot.bar(...))
    nn: same as plot
'''

from aku_utils.data import *

from aku_utils import plot

from aku_utils import nn