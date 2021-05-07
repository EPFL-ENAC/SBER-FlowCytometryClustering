import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn import datasets, metrics, preprocessing, tree
from scipy.stats import zscore


from data import *

def standardize(x):
    """Standardize the original data set."""
    x = x - x.mean(axis=0)
    x = x / x.std(axis=0)
    return x


def standardize_log(x):
    indexes = np.arange(x.shape[0])
    indexes = indexes[(x > 0).all(axis=1)]

    x = x[(x > 0).all(axis=1)]
    x = np.log(x)
    x = x - x.mean(axis=0)
    x = x / x.std(axis=0)
    return x, indexes

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
    df_labeled = pd.read_csv(file)

    #creation of X and y
    X,y = split_input_output(df_labeled,target_feature='label')

    #Detect and remove outliers
    X,y = remove_outliers_with_y(X,y)

    #Standardize our data
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    X[X.columns] = scaler.fit_transform(X[X.columns])

    #Save file in csv format
    save_to_csv(X,y)
    return X,y
            
def run_FlowGrid(nbins=4,eps=1.1,isEvaluation=False):
    command = "python ../FlowGrid/sample_code.py --f fc_data.csv --n "+str(nbins)+" --eps "+str(eps)
    if(isEvaluation):
        command+= " --l label_data.csv > output.txt"
    os.system(command)
    
def cluster(columns):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_intput = 'labeled_dataset/'
    for entry in os.scandir(dir_path+directory):
        if not entry.is_file:
            continue
        elif entry.path.endswith(".csv") and entry.is_file():
            print(entry.path)
            
#directory: folder where you can find all_event and gated
def cluster(directory,columns,label_gated,label_not_gated):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_intput = 'labeled_dataset/'
    folder_output = 'labeled_dataset/'
    for filename in os.listdir(dir_path+ directory+folder_all):
        if filename.endswith(".fcs"):
            df_all = load_data(filename,directory+folder_all,columns)
            df_gated = load_data(filename,directory+folder_gated,columns)
            df_labeled = label(df_all,df_gated,label_gated=label_gated,label_not_gated=label_not_gated)
            new_filename = filename.split(".")[0]+".csv"
            df_labeled.to_csv(folder_output+new_filename, index=False)
            
            #new_filename = filename.split(".")[0]+".csv"
            #to_csv(df_labeled,dir_path+'/labeled_dataset/'+new_filename)

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
    dot_plot_personnalized(data, 2, 3, plot=plt1)
    plt.gca().set_xlabel('B530-H')
    plt.gca().set_ylabel('B572-H')
#    plt.gca().set_xlim(-3,3)
#    plt.gca().set_ylim(-3,3)

    plt2 = plt.subplot(1,2,2)
    dot_plot_personnalized(data, 2 ,3, colors=assignments)
    plt.gca().set_xlabel('B530-H')
    plt.gca().set_ylabel('B572-H')
#    plt.gca().set_xlim(-3,3)
#    plt.gca().set_ylim(-3,3)
    
    plt.show()


        