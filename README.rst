This is the code for manual selection and labelling of cells for Mack et al.

File listing
------------

* selections.py : a basic interface for selecting cells in a set of images
* show.py : a simple script for visually displaying cells that have been selected in a dataset
* labelling.py : an interface for iterating through individual cells randomly, labelling them with arbitrary labels of user's choice
* labels_to_csv.py : a simple script to copy labels created via labelling.py into a csv file for further analysis
* colocalization_auto.py : for a set of selected cells, computes the correlation between two image (colour) channels and saves to csv file
