# SBER-FlowCytometryClustering

<!-- ABOUT THE PROJECT -->
## About The Project

This project is a Semester Project and it aims to use Unsupervised Learning algorithms and Supervised Learning algorithms to build models that separates cells from background and noise, in order to count the cells in each sample as accurately as possible.


<!-- USAGE EXAMPLES -->
## Usage

There are mainly Jupiter Notebooks available with a single helper file called `helpers.py`.

- In order to make everything works, there are few folder you need to have at the root folder level.
    * `data_gated` which contains two other folder `all_event` and `gated` which would contains .fcs files to label
    * `labeled_dataset` which would be empty until you create .csv labeled dataset with the respective Notebook called `Label.ipynb`.
    * `Notebooks` which is a folder containing all the different Notebooks and helper file.

- The folder `Notebooks` is containing the following file:
    * `Clustering-basic-models.ipynb` which is used to tuned all the different basic clustering algorithms like GMM, DBSCAN, KMeans and OPTICS.
    * `Clustering-new-models.ipynb` which is used to tuned the state-of-the-art clustering algorithms like FlowSOM and FlowGrid.
    * `Exploratory.ipynb` and `Exploratory-labeled-data.ipynb` which were used to explore the different files and distributions.
    * `Clustering-evaluation.ipynb` is the file which evaluated the high dimensional algorithms on all the different files
    * `Supervised.ipynb` which is used to create supervised models and test them. At the end if this file, you will also find how to save and load models in order to avoid retraining them.
    * `Supervised-tuning.ipynb` which is used to tune all the different models and find the best train set.

## Authors
* Michael Spierer
