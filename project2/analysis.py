from statsmodels.stats.proportion import proportion_confint
from statsmodels.sandbox.stats.multicomp import multipletests
import pickle
from soup import *
from scipy.stats import f_oneway
##
# read in data and compute some stats

result_dir = 'outputs/results_v4'
result_files = ['9515_results.pkl','52416_results.pkl','52516_results.pkl','91615_results.pkl','52616_results.pkl','92215_results.pkl']
result_files = [os.path.join(result_dir,rf) for rf in result_files]
results = {}

for rf in result_files:
    if not os.path.exists(rf):
        continue
    with open(rf, 'rb') as f:
        r = pickle.load(f)
        results.update(r)

allrows = []
for k,v in results.items():
    rs = v['rs']

    origin = os.path.split(k)[0]

    # manual condition finding
    fname = os.path.splitext(os.path.split(k)[-1])[0]
    fname = fname.lower()
    if 't' in fname and 'cont' not in fname:
        cond = 't' + fname[fname.index('t')+1]
        if cond=='t.':
            print(fname)
    elif '_' in fname:
        if 'KLM3126' in fname:
            prefix = 'b'
        else:
            prefix = ''
        cond = fname[fname.index('_')+1]
        cond = prefix + cond

    if cond=='t1':
        cond = '1'
    if cond=='t2':
        cond = '3'

    rows = np.zeros([len(rs),3], dtype=np.object)
    rows[:,0] = cond
    rows[:,1] = rs
    rows[:,2] = origin
    allrows.append(rows)
rows = pd.DataFrame(np.concatenate(allrows), columns=['cond','r','origin'])
rows.loc[:,'r'] = rows.loc[:,'r'].astype(float)
rows = rows.sort_values('cond')
rows = rows[rows.cond!='6']

# !!!!!!!!!!!!!!!!!!!!!!! squared
#rows.loc[:,'r'] = rows.loc[:,'r']**2


fig,axs = pl.subplots(1,2)
# continuous
grp = rows.groupby(['cond','origin']).r.mean()
err = grp.std(level=0) / np.sqrt(grp.count(level=0))
mean = grp.mean(level=0)
idx = np.arange(len(mean))
axs[0].errorbar(idx, mean.values, label='mean', yerr=err, marker='o', linewidth=2, color='k', ecolor='gray')
axs[0].set_xticks(idx)
axs[0].set_xticklabels(np.asarray(mean.index))
# quantized
threshs = np.arange(0.6,0.95,0.1)
cols = pl.cm.viridis(np.linspace(0,.9,len(threshs)))
for thresh,col in zip(threshs,cols):
    print(thresh)
    print(rows.groupby(['origin','cond']).r.apply(lambda x: np.mean(x>thresh)).round(2))
    print()

    # pooling:
    #mean = rows.groupby('cond').r.apply(lambda x: np.mean(x>thresh))
    #midx = mean.index
    #mean = mean.values
    #err = rows.groupby('cond').r.apply(lambda x: proportion_confint(np.sum(x>thresh), len(x)))
    #err = np.asarray([e for e in err.values]).T
    #err -= mean
    #err = np.abs(err)

    # biological rep grouping:
    gb = rows.groupby(['cond','origin']).r.apply(lambda x: np.mean(x>thresh))
    mean = gb.mean(level=0)
    midx = mean.index
    mean = mean.values
    err = gb.std(level=0) / np.sqrt(gb.count(level=0))
    err = err.values

    axs[1].errorbar(idx, mean, label=thresh, color=col, yerr=err, ecolor='k', marker='o')

axs[1].set_xticks(idx)
axs[1].set_xticklabels(np.asarray(midx))
axs[0].set_title('Continuous values')
axs[1].set_title('Thresholded')
#[a.legend(fontsize='x-small', ncol=2) for a in axs]
axs[1].legend(fontsize='x-small', ncol=2, loc='upper right')
[a.set_xlabel('Condition') for a in axs]
axs[0].set_ylabel('Colocalization score')
axs[1].set_ylabel('Fraction of cells passing colocalization threshold')
[pretty(ax=ax) for ax in axs]

##
thresh = .8

gr = rows.groupby(['cond','origin']).r.apply(lambda x: np.mean(x>thresh))

base = '1'
a = gr[base].values
bs = ['2','3','4','5','t3','t4']

pvals = []
for b_selection in bs:
    b = gr[b_selection].values
    ar = f_oneway(a,b)
    pvals.append(ar.pvalue)
reject,pvals_,_,_ = multipletests(pvals, method='bonferroni')

dd = {True:'DIFFERENT', False:'NOT DIFFERENT'}
for b,p in zip(bs,pvals):
    print('ANOVA {} to {}: {} (p={:0.2f})'.format(base,b,dd[p<.05],p))
##
