##
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import ttest_ind

data = pd.read_csv('klm2.csv')
labs = data.iloc[:,0].values
labs = [l[:l.index('.lif')] for l in labs]
trial_id,cond = zip(*[l.split('_') for l in labs])
data = data.iloc[:,1].values

data = pd.Series(data)
cond = pd.Series(cond)
trial_id = pd.Series(trial_id)

# mergers
cond[cond=='T1'] = '1'
cond[cond=='T2'] = '3'
cond[cond=='T3'] = '6'
cond[cond=='T4'] = '7'

##

MODE = 'thresh' # thresh or continuous

over = data > .8

fig,axs = pl.subplots(1,2,sharey=True)
pad = .08
xposs = []
xp = 0

utid = np.unique(trial_id).tolist()
tcols = pl.cm.tab20c(np.linspace(0,1,len(utid)))

for i,c in enumerate(sorted(cond.unique())):
    trs = trial_id[cond==c]
    w = .2
    xps = []
    trialmeans = []
    for ti,t in enumerate(sorted(trs.unique())):

        if MODE == 'thresh':
            grp = over[(cond==c) & (trial_id==t)].values
            mean = grp.mean()
        elif MODE == 'continuous':
            grp = data[(cond==c) & (trial_id==t)].values
            mean = np.median(grp)

        trialmeans.append(mean)
        ci_lo, ci_hi = proportion_confint(grp.sum(), len(grp))
        yerr = np.array([[mean-ci_lo, mean+ci_hi]])

        col = tcols[utid.index(t)]
        
        xps.append(xp)
        axs[0].bar(xp, mean, yerr=yerr, width=w, color=col, ecolor='grey')
        axs[0].text(xp, mean+.05, f'{len(grp)}', fontsize='small', color=col, ha='center')
        xp += w+pad
    xp += 1
    xposs.append(np.mean(xps))
    
    trialmeans = np.array(trialmeans)
    #axs[1].bar(i+1, trialmeans.mean(), yerr=trialmeans.std(), color='k', width=.4)
    axs[1].plot([1+i-.2,1+i+.2], [np.median(trialmeans)]*2, lw=2, color='k')
    jx = [i+1+np.random.normal(0,.05) for _ in range(len(trialmeans))]
    axs[1].scatter(jx, trialmeans, marker='o', color='grey', zorder=10, s=8)
axs[0].set_xticks(xposs)
#axs[0].set_xticklabels([str(int(i+1)) for i in range(len(cond.unique()))])
axs[0].set_xticklabels(sorted(cond.unique()))

#stats
alldat = {}
for c in cond.unique():
    trs = trial_id[cond==c]
    trialmeans = []
    for ti,t in enumerate(sorted(trs.unique())):
        grp = over[(cond==c) & (trial_id==t)].values
        mean = grp.mean()
        trialmeans.append(mean)
    alldat[c] = trialmeans

out = []

lidx = 0.5
nk = len(alldat.keys())
ncomp = nk*(nk-1) / 2
for k in sorted(alldat.keys()):
    for kk in sorted(alldat.keys()):
        ki = int(k)
        kki = int(kk)
        if ki >= kki:
            continue
        p = ttest_ind(alldat[k], alldat[kk]).pvalue
        if p<.05/ncomp:
            col = 'red'
        else:
            col = 'grey'
        axs[1].plot([ki, kki], [lidx]*2, lw=1, color=col)
        lidx += .01
        

## sandbox

for i,c in enumerate(sorted(cond.unique())):
    di = data[cond==c]
    mean = np.mean(di)
    pl.violinplot(data[cond==c], positions=[i+1])
    pl.plot([i+1-.2, i+1+.2], [mean]*2, lw=2, color='k')

pl.xlim([0,8])

##
