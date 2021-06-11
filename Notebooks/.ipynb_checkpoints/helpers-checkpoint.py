import numpy as np
import pandas as pd
import time
from sklearn.covariance import EllipticEnvelope
from sklearn import datasets, metrics, preprocessing, tree
from scipy.stats import zscore
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq, kmeans as sci_kmean
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score, silhouette_score
from sklearn.metrics.cluster import rand_score, adjusted_rand_score ,v_measure_score
from sklearn.model_selection import train_test_split
from flowsom import *
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns;
import matplotlib.pyplot as plt
import collections

from data import *

"""
******************************************************************************
Helpers outside the class
******************************************************************************
"""

"""
 * Split the Dataframe a feature matrix X and a target feature y
"""
def split_input_output(df,target_feature):
    X = df.drop(target_feature,axis=1)
    y = df[target_feature]
    return X,y

"""
 * Split proportionaly the dataset into train/test if target class is imbalanced
"""
def proportional_train_test_split(X, y,y_label='label', test_size=0.2,train_size=0.8, random_state=0):
    #find all y==1 and y==0
    df = X.copy()
    df[y_label] = y.copy()
    X_1 = df[df[y_label]==1]
    y_1 = X_1[y_label]
    X_1 = X_1.drop([y_label],axis=1)
    X_not1 = df[df[y_label]!=1]
    y_not1 = X_not1[y_label]
    X_not1 = X_not1.drop([y_label],axis=1)
    
    #split in each sample
    X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X_1, y_1, test_size=test_size, random_state=random_state)
    X_not1_train, X_not1_test, y_not1_train, y_not1_test = train_test_split(X_not1, y_not1, test_size=test_size, random_state=random_state)
    
    #merge them together
    X_final_train = X_1_train.append(X_not1_train, ignore_index = True)
    X_final_test = X_1_test.append(X_not1_test, ignore_index = True)
    y_final_train = y_1_train.append(y_not1_train, ignore_index = True)
    y_final_test = y_1_test.append(y_not1_test, ignore_index = True)
    
    return X_final_train, X_final_test, y_final_train, y_final_test


"""
 * Load a FCS fle into a FCSDATA structure
"""
def load_file(fileName, folder):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    s = FlowCal.io.FCSData(dir_path+folder+fileName) 
    return s

"""
 * Load a FCS file into a pandas DataFrame structure
"""
def load_data(fileName, folder,columns):
    data = load_file(fileName, folder)
    data = data[:, columns]
    data = np.array(data)
    data_df = pd.DataFrame(data=data, columns=columns)
    return data_df

"""
 * Takes the df of all entries, the df of the gated ones
 * a label to assign to the gated entries and to the not gated ones.
 * and produce a df containing all the entries and their label
"""
def label(df_all,df_gated,label_gated,label_not_gated):
    #create a hash for each entry to find all entries not gated
    df_all.loc[:, "hash"] = df_all.apply(lambda x: hash(tuple(x)), axis = 1)
    df_gated.loc[:, "hash"] = df_gated.apply(lambda x: hash(tuple(x)), axis = 1)
    df_not_gated = df_all.loc[~df_all["hash"].isin(df_gated["hash"]), :]

    #label dataset
    df_gated.loc[:,"label"] = label_gated
    df_not_gated.loc[:,"label"] = label_not_gated

    #concat dataset and remove hash column
    #df_final = pd.concat([df_gated, df_not_gated], ignore_index=True)
    df_labeled = df_not_gated.append(df_gated,ignore_index=True)
    df_labeled.drop('hash', inplace=True, axis=1)
    return df_labeled

"""
 * Given a folder containing both the "all_event" and "gated" folder,
 * wanted columns, and an output folder
 * 
 * Create in the output folder a .csv labeled file for each entry of the input folder.
"""
#directory: folder where you can find all_event and gated
def create_labeled_dataset(directory,columns,folder_output='/../labeled_dataset/',label_gated=3,label_not_gated=-1):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_all = 'all_event/'
    folder_gated = 'gated/'
    for filename in os.listdir(dir_path+ directory+folder_all):
        if filename.endswith(".fcs"):
            df_all = load_data(filename,directory+folder_all,columns)
            df_gated = load_data(filename,directory+folder_gated,columns)
            df_labeled = label(df_all,df_gated,label_gated=label_gated,label_not_gated=label_not_gated)
            new_filename = filename.split(".")[0]+".csv"
            df_labeled.to_csv(dir_path+folder_output+new_filename, index=False)

"""
 * Given a .csv file and wanted columns
 *
 * Standardize the data and remove the outliers
 * 
 * return X,y the pre-processed dataset and target variable.
"""            
def preprocess(file,columns):
    #print(file)
    df_labeled = pd.read_csv(file)

    #creation of X and y
    X,y = split_input_output(df_labeled,target_feature='label')
    X = X[columns]
    
    #Index of all values strictly positive (because we will apply log() to all our data for standardization)
    na_indexes = (X > 0).all(1)

    #Standardize our data
    X = X[na_indexes]
    y = y[na_indexes]
    X = np.log(X)

    scaler = preprocessing.StandardScaler()
    X[columns] = scaler.fit_transform(X[columns])
    
    #Detect and remove outliers
    X,y = remove_outliers_with_y(X,y)
    
    return X,y


"""
 * Given a .fcs file and wanted columns
 *
 * Standardize the data and remove the outliers
 * 
 * return X the pre-processed unlabeled dataset
"""            
def preprocess_unlabeled(file,columns,folder='/../unlabeled_data/'):

    X = load_data(file,folder,columns)

   
    
    #Index of all values strictly positive (because we will apply log() to all our data for standardization)
    na_indexes = (X > 0).all(1)

    #Standardize our data
    X = X[na_indexes]
    X = np.log(X)

    scaler = preprocessing.StandardScaler()
    X[columns] = scaler.fit_transform(X[columns])
    
    #Detect and remove outliers
    X,indexes = remove_outliers(X)
    
    return X


"""
 * Given true y and prediction y
 *
 * Print all clustering evaluation metric
"""   
def run_eval(y_true,y_pred):
    print("Rand Index: %0.3f" % metrics.rand_score(y_true, y_pred))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(y_true, y_pred))
    #print("Homogeneity: %0.3f" % metrics.homogeneity_score(y_true, y_pred))
    #print("Completeness: %0.3f" % metrics.completeness_score(y_true, y_pred))
    print("V-measure: %0.3f" % metrics.v_measure_score(y_true, y_pred))
    print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(y_true, y_pred))
    

"""
 * Given wanted columns, a folder containing csv files, an output folder to save pictures,
 * a clustering technic
 *
 * REQUIREMENT: labeled csv file (since we are interested in evaluating the different models)
 *
 * Run a clustering algorithm on all files and save the output picture in the output folder
 * If verbose = True, print the number of cells of each cluster
"""               
def run_all(columns,directory='../labeled_dataset/',output_directory='/outputs/',clustering="kmeans",verbose=False):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for entry in os.scandir(dir_path+"/"+directory):
        if not entry.is_file:
            continue
        elif entry.path.endswith(".csv") and entry.is_file():
            filename= entry.name
            X,y = preprocess(directory+filename,columns)
            command = clustering.lower()
            if(command == "flowgrid"):
                save_to_csv(X,y)
                y_pred = run_FlowGrid(nbins=3,eps=0.9,isEvaluation=False)
                X = X.drop([0])
                y = y.drop([0])
            elif(command == "flowsom"):
                save_to_csv(X,y,X_name="flowsom.csv",y_name="flowsom_label.csv")
                y_pred = run_FlowSOM('flowsom.csv')
                X = X.drop([0])
                y = y.drop([0])
            elif(command == "kmeans"):
                cluster_model = clusterKMeans(X)
                y_pred = cluster_model.labels_
            elif(command =="gmm"):
                y_pred = clusterGMM(X)
            elif(command =="optics"):
                cluster_model = clusterOPTICS(X)
                y_pred = cluster_model.labels_
            elif(command =="dbscan"):
                cluster_model = clusterDBSCAN(X)
                y_pred = cluster_model.labels_
            elif(command =="true_labels"):
                y_pred = y
            else:
                print(f"Unknown clustering algorithm")
                print(f"Please use on of the following clustering methods:")
                print(f"kmeans, gmm, optics, dbscan, flowgrid, flowsom, true_labels")
                break
            print(entry.path)
            run_eval(y,y_pred)
            plt.figure()
            plt.title(entry.name + " - " + command)
            pl = sns.scatterplot(data=X, x="B530-H", y="B572-H", hue=y_pred)
            if verbose:
                print(X.shape)
                print("real clusters")
                print(collections.Counter(y).most_common)
                print("prediction clusters")
                print(collections.Counter(y_pred).most_common)
            
            new_filename = filename.split(".")[0]
            pl.figure.savefig(dir_path+output_directory+new_filename+"_"+command+".png")
   
            

"""
 * Save a df and a target variable in csv format
"""  
def save_to_csv(X,y,X_name="fc_data.csv",y_name="label_data.csv"):
    to_csv(X,X_name)
    to_csv(y,y_name)
    
def to_csv(df,name):
    df_array = np.array(df)
    np.savetxt(name,df_array, delimiter=",")


    
"""
 * Remove all outliers. An outlier of a dataset is defined as a value that is more than 'distance' standard deviations from the mean.
 *
 * return the filtered df
"""  
def remove_outliers(df,distance=3):
    z_scores = zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < distance).all(axis=1)
    return df[filtered_entries],filtered_entries

def remove_outliers_with_y(df,y,distance=3):
    df_ = df.copy()
    df_,filtered_entries = remove_outliers(df_,distance)
    return df_,y[filtered_entries]


def export_csv(data, name):
    np.savetxt(name+".csv", data, delimiter=",")
    

"""
  Return the centroids (mean point of a cluster) from a given assignment.
  Parameters:
    assignment : A clusterisation array given by one the clusterisation function.
  Return :
    A table of tuples composed of (id_of_cluster, position_of_centroid)
"""
def get_centroids(data,assignment):
    centroids = []
    for cur in np.unique(assignment):
        center = np.mean(data[assignment==cur], axis=0)
        centroids.append((cur, center))
    return centroids


"""
  Run the KMeans algorithm on the dataset
  Parameters:
    n_clusters : The number of expected clusters
  Returns :
    An array of assignment for each sample to a cluster
"""
def clusterKMeans(data,n_clusters=3, random_state=None):
    return KMeans(n_clusters=n_clusters, random_state=random_state).fit(data)


"""
  Run the DBSCAN algorithm on the dataset
  Parameters:
    eps : The maximum distance for two points to be considered neighbor
    min_samples : The minimum number of samples in the neighborhood of a point to be considered to be in the same cluster
  Returns :
    An array of assignment for each sample to a cluster
"""
def clusterDBSCAN(data,eps=0.38, min_samples=170,metric='l2', algorithm='auto'):
    return DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm).fit(data)

"""
  Run the OPTICS algorithm on the dataset
  Returns :
    An array of assignment for each sample to a cluster
"""
def clusterOPTICS(data,metric="l2",min_samples=187,eps=0.42,cluster_method='xi'):
    return OPTICS(metric=metric,max_eps=eps, min_samples=min_samples,cluster_method=cluster_method).fit(data)

"""
  Run the Gaussian Mixture algorithm on the dataset
  Parameters :
    nb_components : The number of desired clusters
  Returns :
    An array of assignment for each sample to a cluster
"""
def clusterGMM(data,n_components=4,covariance_type="tied",n_init=2,random_state=None):
    return GaussianMixture(n_components=n_components, covariance_type=covariance_type,n_init=n_init,random_state=random_state).fit_predict(data)

"""
  Run the FlowGrid algorithm on the dataset
  Parameters :
    nbins : The number of bins
    epsilon : used to determine if two bins are directly connected
  Returns :
    An array of assignment for each sample to a cluster
"""            
def run_FlowGrid(nbins=4,eps=1.1,isEvaluation=False):
    command = "python ../FlowGrid/sample_code.py --f fc_data.csv --n "+str(nbins)+" --eps "+str(eps)
    if(isEvaluation):
        command+= " --l label_data.csv > output.txt"
    os.system(command)
    y_pred = np.genfromtxt('fc_data_FlowGrid_label.csv', delimiter=',')
    return y_pred
            
"""
  Run the FlowSOM algorithm on the dataset
  Parameters :
    min_n : The minimum number of cluster of meta-clustering
    max_n : The maximum number of cluster of meta-clustering
  Returns :
    An array of assignment for each sample to a cluster
"""  
def run_FlowSOM(file,sigma=2.5,lr=0.1,batch_size=100,min_n=2,max_n=6,iter_n=6,n_features=4):
    #fsom = flowsom(file, if_fcs=False, if_drop=True, drop_col=['FSC-H','SSC-H','FSC-A','SSC-A','B572-A','B675-A','Time','label'])
    #fsom = flowsom(file, if_fcs=False, if_drop=True, drop_col=['Unnamed: 0'])
    fsom = flowsom(file, if_fcs=False, if_drop=False)
    fsom.som_mapping(50, 50, n_features, sigma=sigma, 
                 lr=lr, batch_size=batch_size)  # trains SOM with 100 iterations 3
    fsom.meta_clustering(AgglomerativeClustering, min_n=min_n, 
                     max_n=max_n, 
                     iter_n=iter_n) 
    fsom.labeling()
    y_pred = fsom.df['category']

    return y_pred



"""
*************************
 *   Old functions
*************************
"""

def concat_2Dlabel(data, d1, d2, labels):
    X = np.concatenate((np.reshape(data[:,d1], (len(data),1)),
                        np.reshape(data[:,d2], (len(data),1))), axis = 1)
    return np.concatenate((X, np.reshape(labels, (len(data), 1))), axis = 1)


def with_index(array):
    return np.c_[np.arange(array.shape[0]), array]


def data_filter(filter, data, assignments):
    new_data = []
    new_assignments = []
    for i in range(len(assignments)):
        if assignments[i] in filter:
            new_data.append(data[i])
            new_assignments.append(assignments[i])

    return (np.array(new_data), np.array(new_assignments))

