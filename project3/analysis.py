##

result_dirs = [d for d in os.listdir('outputs') if os.path.isdir(os.path.join('outputs',d))]
result_file_paths = [os.path.join('outputs',d,'all_results.csv') for d in result_dirs]

##
tables = [pd.read_csv(f) for f in result_file_paths]

all_results = pd.concat(tables)

all_results.reset_index(inplace=True, drop=True)
all_results = all_results.iloc[:,1:]

##

all_results.to_csv('aggregate_results.csv')

##

ag = pd.read_csv('aggregate_results.csv')
ag.loc[:,'cond'] = ag.cond.str.replace('_','')

means1 = ag.groupby(['cond','filename']).value.mean()

# or

ag.loc[:,'thresholded'] = (ag.value > 0.05).values
means1 = ag.groupby(['cond','filename']).thresholded.mean()

# then

means2 = means1.mean(level=0)

##

(means1).plot(kind='bar')
(means2).plot(kind='bar')

##
