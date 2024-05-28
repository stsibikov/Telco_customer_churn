


'''
Submodules:
    data: data wrangling
    plot: plot formatting

Plot functions should have ready (groupy'd) data passed into its functions,
while data shouldn't (unless the function does smt specific)

Access submodules:
    data: root (eg aku_utils.na(df))
    plot: aku_utils.plot (eg aku_utils.plot.bar(...))
'''

from aku_utils.data import *

from aku_utils import plot