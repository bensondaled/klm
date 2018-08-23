# Main parameters
input_folder = '/Volumes/KLM2/tdp43_examples' # this folder should have the original tifs *and* the mask files
analysis = 'colocalization' # foci / colocalization / membrane

# Parameters for each type of analysis
foci_kw = dict(channel=1) # 0=red, 1=green, 2=blue
colocalization_kw = dict(channel_a=0, channel_b=1)
membrane_kw = dict(channel=1) # 0=red, 1=green, 2=blue

# Run the analysis
from cell_routines import Analysis
all_analysis_kw = dict(foci=foci_kw, 
            colocalization=colocalization_kw, 
            membrane=membrane_kw)
analysis_kw = all_analysis_kw[analysis]
anal = Analysis(input_folder, kind=analysis, **analysis_kw)
anal.run()
anal.save() # will save to results.csv in the input_folder
