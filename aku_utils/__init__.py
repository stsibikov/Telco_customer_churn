


'''
Submodules:
    data: data wrangling
    plot: plot formatting

Plot functions should have ready (groupy'd) data passed into its functions,
while data shouldn't (unless the function does smt specific)

Access submodules:
    data: root (eg aku_utils.na(df))
    plot: aku_utils.plot (eg aku_utils.plot.fmt_bar(...))
'''

from aku_utils.data import (
    probs_replace,
    group_occrs,
    na,
    type_breakdown,
    inspect_mean
)

from aku_utils import plot