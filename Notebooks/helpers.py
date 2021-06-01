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


from data import *

"""
******************************************************************************
Helpers outside the class
******************************************************************************
"""
def split_input_output(df,target_feature):
    X = df.drop(target_feature,axis=1)
    y = df[target_feature]
    return X,y

def load_file(fileName, folder):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    s = FlowCal.io.FCSData(dir_path+folder+fileName) 
    return s

def load_data(fileName, folder,columns):
    data = load_file(fileName, folder)
    data = data[:, columns]
    data = np.array(data)
    data_df = pd.DataFrame(data=data, columns=columns)
    return data_df


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
            
            #new_filename = filename.split(".")[0]+".csv"
            #to_csv(df_labeled,dir_path+'/labeled_dataset/'+new_filename)

def preprocess(file,columns):
    print(file)
    df_labeled = pd.read_csv(file)

    #creation of X and y
    X,y = split_input_output(df_labeled,target_feature='label')
    
    X = X[columns]
    
    na_indexes = (X > 0).all(1)


    #Standardize our data
    X = np.log(X)
    X = X[na_indexes]
    y = y[na_indexes]
    
    scaler = preprocessing.StandardScaler()
    X[X.columns] = scaler.fit_transform(X[X.columns])
    
        
    #Detect and remove outliers
    X,y = remove_outliers_with_y(X,y)

    
    #Save file in csv format
    save_to_csv(X,y)
    return X,y

def run_eval(X, y_true,y_pred):
    print("Rand Index: %0.3f" % metrics.rand_score(y_true, y_pred))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(y_true, y_pred))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(y_true, y_pred))
    print("Completeness: %0.3f" % metrics.completeness_score(y_true, y_pred))
    print("V-measure: %0.3f" % metrics.v_measure_score(y_true, y_pred))
    print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(y_true, y_pred))
#    print("Silhouette Coefficient: %0.3f" % #metrics.silhouette_score(X, y_pred))
    
            
def run_FlowGrid(nbins=4,eps=1.1,isEvaluation=False):
    command = "python ../FlowGrid/sample_code.py --f fc_data.csv --n "+str(nbins)+" --eps "+str(eps)
    if(isEvaluation):
        command+= " --l label_data.csv > output.txt"
    os.system(command)
            
def run_FlowSOM(file):
    #fsom = flowsom(file, if_fcs=False, if_drop=True, drop_col=['FSC-H','SSC-H','FSC-A','SSC-A','B572-A','B675-A','Time','label'])
    #fsom = flowsom(file, if_fcs=False, if_drop=True, drop_col=['Unnamed: 0'])
    fsom = flowsom(file, if_fcs=False, if_drop=False)
    fsom.som_mapping(50, 50, 5, sigma=2.5, 
                 lr=0.1, batch_size=100)  # trains SOM with 100 iterations
    fsom.meta_clustering(AgglomerativeClustering, min_n=40, 
                     max_n=45, 
                     iter_n=3) 
    fsom.labeling()
    return fsom.df
            
def run_all(columns,directory='../labeled_dataset/',output_directory='/outputs/',clustering="kmeans"):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for entry in os.scandir(dir_path+"/"+directory):
        if not entry.is_file:
            continue
        elif entry.path.endswith(".csv") and entry.is_file():
            filename= entry.name
            X,y = preprocess(directory+filename,columns)
            print(entry.path)
            print(X.shape)
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            X_train = X.copy()
            X_test = X.copy()
            y_train = y.copy()
            y_test = y.copy()
            command = clustering.lower()
            if(command == "flowgrid"):
                save_to_csv(X,y)
                run_FlowGrid()
                y_pred = np.genfromtxt('fc_data_FlowGrid_label.csv', delimiter=',')
                X_test = X_test.drop([0])
                y_test = y_test.drop([0])
            elif(command == "flowsom"):
                save_to_csv(X,y,X_name="flowsom.csv",y_name="flowsom_label.csv")
                output_df = run_FlowSOM('flowsom.csv')
                y_pred = output_df['category']
                X_test = X_test.drop([len(y)-1])
                y_test = y_test.drop([len(y)-1])
            elif(command == "kmeans"):
                cluster_model = clusterKMeans(X_train)
                y_pred = cluster_model.predict(X_test)
            elif(command =="gmm"):
                cluster_model = clusterGMM(X_train)
                y_pred = cluster_model.predict(X_test)
            elif(command =="optics"):
                cluster_model = clusterOPTICS(X_test)
                y_pred = cluster_model.labels_
            elif(command =="dbscan"):
                cluster_model = clusterDBSCAN(X_test)
                y_pred = cluster_model.labels_
            else:
                print(f"Unknown clustering algorithm")
                break
            run_eval(X_test,y_test,y_pred)
            plt.figure()
            plt.title(entry.name + " - " + command)
            pl = sns.scatterplot(data=X_test, x="B530-H", y="B572-H", hue=y_pred)
            
            new_filename = filename.split(".")[0]
            pl.figure.savefig(dir_path+output_directory+new_filename+"_"+command+".png")
   
            
def save_all_true(columns,directory='../labeled_dataset/',output_directory='/outputs/'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for entry in os.scandir(dir_path+"/"+directory):
        if not entry.is_file:
            continue
        elif entry.path.endswith(".csv") and entry.is_file():
            filename= entry.name
            X,y = preprocess(directory+filename,columns)
            print(entry.path)
            plt.figure()
            plt.title(entry.name)
            pl = sns.scatterplot(data=X, x="B530-H", y="B572-H", hue=y)
            new_filename = filename.split(".")[0]
            pl.figure.savefig(dir_path+output_directory+new_filename+"_true.png")
    
    
#directory: folder where you can find all_event and gated
#def cluster(directory,columns,label_gated,label_not_gated):
#    dir_path = os.path.dirname(os.path.realpath(__file__))
#    folder_intput = 'labeled_dataset/'
#    folder_output = 'labeled_dataset/'
#    for filename in os.listdir(dir_path+ directory+folder_all):
#        if filename.endswith(".fcs"):
#            df_all = load_data(filename,directory+folder_all,columns)
#            df_gated = load_data(filename,directory+folder_gated,columns)
#            df_labeled = label(df_all,df_gated,label_gated=label_gated,label_not_gated=label_not_gated)
#            new_filename = filename.split(".")[0]+".csv"
#            df_labeled.to_csv(folder_output+new_filename, index=False)

def save_to_csv(X,y,X_name="fc_data.csv",y_name="label_data.csv"):
    to_csv(X,X_name)
    to_csv(y,y_name)
    
def to_csv(df,name):
    df_array = np.array(df)
    np.savetxt(name,df_array, delimiter=",")

    

#Remove all outliers. An outlier of a dataset is defined as a value that is more than 'distance' standard deviations from the mean.
def remove_outliers(df,distance=3):
    z_scores = zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < distance).all(axis=1)
    return df[filtered_entries]

def remove_outliers_with_y(df,y,distance=3):
    z_scores = zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < distance).all(axis=1)
    return df[filtered_entries],y[filtered_entries]


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
def clusterKMeans(data,n_clusters=2, random_state=0):
    return KMeans(n_clusters=n_clusters, random_state=random_state).fit(data)


"""
  Run the DBSCAN algorithm on the dataset
  Parameters:
    eps : The maximum distance for two points to be considered neighbor
    min_samples : The minimum number of samples in the neighborhood of a point to be considered to be in the same cluster
  Returns :
    An array of assignment for each sample to a cluster
"""
def clusterDBSCAN(data,eps=0.7, min_samples=188,metric='euclidean', algorithm='auto'):
    return DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm).fit(data)

"""
  Run the OPTICS algorithm on the dataset
  Returns :
    An array of assignment for each sample to a cluster
"""
def clusterOPTICS(data,metric="euclidean"):
    return OPTICS(metric=metric).fit(data)

"""
  Run the Gaussian Mixture algorithm on the dataset
  Parameters :
    nb_components : The number of desired clusters
  Returns :
    An array of assignment for each sample to a cluster
"""
def clusterGMM(data,n_components=2,covariance_type="full"):
    return GaussianMixture(n_components=n_components, covariance_type=covariance_type).fit(data)
     

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


def plot_results(data, channels, assignments):
    for i in range(len(channels)):
        for j in range(i+1, len(channels)):
            plt.figure(figsize=(15,7.5))

            plt1 = plt.subplot(1,2,1)
            dot_plot_personnalized(data, i, j, plot=plt1)
            plt.gca().set_xlabel(channels[i])
            plt.gca().set_ylabel(channels[j])
            plt.gca().set_xlim(-3,3)
            plt.gca().set_ylim(-3,3)

            plt2 = plt.subplot(1,2,2)
            dot_plot_personnalized(data, i ,j, colors=assignments)
            plt.gca().set_xlabel(channels[i])
            plt.gca().set_ylabel(channels[j])
            plt.gca().set_xlim(-3,3)
            plt.gca().set_ylim(-3,3)

            plt.show()


def plot_results_b530_b572(data, channels, assignments):
    plt.figure(figsize=(15,7.5))
    plt1 = plt.subplot(1,2,1)
    dot_plot_personnalized(data, 0, 3, plot=plt1)
    plt.gca().set_xlabel('B530-H')
    plt.gca().set_ylabel('B572-H')
#    plt.gca().set_xlim(-3,3)
#    plt.gca().set_ylim(-3,3)

    plt2 = plt.subplot(1,2,2)
    dot_plot_personnalized(data, 0 ,3, colors=assignments)
    plt.gca().set_xlabel('B530-H')
    plt.gca().set_ylabel('B572-H')
#    plt.gca().set_xlim(-3,3)
#    plt.gca().set_ylim(-3,3)
    
    plt.show()


        