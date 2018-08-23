##
from soup import *

##
label_path = 'examples/9515_examples_v4//manual_labelling/ben.csv'
real_path = 'examples/9515_examples_v4//key.pd'
manual = pd.read_csv(label_path, header=1).dropna(axis=0, how='all')
real = pd.read_pickle(real_path)
##
assert np.all(real.ex_idx == manual['image ID'])
##

is_ok = manual['would not have used this (x)'].isnull()
#is_ok = [True] * len(manual)

r_real = real.r

r_man = manual['colocalization score (1-4)']
#r_man = np.round(r_man)

yn =  manual['yes or no (y/n)']

r_real = r_real[is_ok]
r_man = r_man[is_ok]
yn = yn[is_ok]

# !!!!! **** squared
#r_real = r_real**2

##
gb = r_real.groupby(r_man)
comparison = gb.mean()
comparison_err = gb.std() / np.sqrt(gb.count())
comparison_err = None
pl.errorbar(comparison.index, comparison.values, yerr=comparison_err, marker='o', linewidth=0, color='gray', markersize=10)
pl.xlabel('Manual score', fontsize='x-large')
pl.ylabel('Automatic score', fontsize='x-large')
pretty()

## experimental: using labelling to find a global threshold
grp = r_real.groupby(yn.values)
vecs = [grp.get_group(g).values for g in grp.groups]
labs = [g for g in grp.groups]

pl.boxplot(vecs)
pl.gca().set_xticklabels(labs)
ax = pl.gca()
ax.set_xticklabels(['User Rejected', 'User Accepted'], fontsize='x-large')
ax.set_ylabel('Automatic Colocalization Score', fontsize='x-large')
pretty()

##
