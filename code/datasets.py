import os
import numpy as np
import scipy.ndimage as nd
import scipy.io as sio
import utils

def closest_file(fid, extension=".mat"):
    # Fix occasional issues with extensions (e.g. X.mp4.mat)
    basename = os.path.basename(fid)
    dirname = os.path.dirname(fid)
    dirfiles = os.listdir(dirname)
    
    if basename in dirfiles:
        return fid
    else:
        basename = basename.split(".")[0]
        files = [f for f in dirfiles if basename in f]
        if extension is not None:
            files = [f for f in files if extension in f]
        if len(files) > 0:
            return dirname+"/"+files[0]
        else:
            print("Error: can't find file")


def remove_exts(name, exts):
    for ext in exts:
        name = name.replace(ext, "")
    return name


class Dataset:
    name = ""
    n_classes = None
    n_features = None
    activity = None

    def __init__(self, name="", base_dir="", activity=None):
        self.name = name
        self.base_dir = os.path.expanduser(base_dir)

        #Find the number of splits
        split_folders = os.listdir(self.base_dir+"splits/{}/".format(self.name))
        self.splits = np.sort([s for s in split_folders if "Split" in s])
        self.n_splits = len(self.splits)

    def feature_path(self, features):
        return os.path.expanduser(self.base_dir+"features/{}/{}/".format(self.name, features))

    def get_files(self, dir_features, split=None):
        if "Split_1" in os.listdir(dir_features):
            files_features = np.sort(os.listdir(dir_features+"/{}/".format(split)))
        else:
            files_features = np.sort(os.listdir(dir_features))
            
        files_features = [f for f in files_features if f.find(".mat")>=0]
        return files_features

    def fid2idx(self, files_features, extensions=[".mov", ".mat", ".avi", "rgb-"]):
        return {remove_exts(files_features[i], extensions):i for i in range(len(files_features))}

    def load_split(self, features, split, feature_type="X", sample_rate=1):
        # Setup directory and filenames
        dir_features = self.feature_path(features)

        # Get splits for this partion of data
        if self.activity==None:
            file_train = open(self.base_dir+"splits/{}/{}/train.txt".format(self.name, split)).readlines()
            file_test = open( self.base_dir+"splits/{}/{}/test.txt".format(self.name, split)).readlines()
        else:
            file_train = open(self.base_dir+"splits/{}/{}/{}/train.txt".format(self.name, self.activity, split)).readlines()
            file_test = open( self.base_dir+"splits/{}/{}/{}/test.txt".format(self.name, self.activity, split)).readlines()         
        file_train = [f.strip() for f in file_train]
        file_test = [f.strip() for f in file_test]     

        # Remove extension
        if  "." in file_train[0]:
            file_train = [".".join(f.split(".")[:-1]) for f in file_train]
            file_test = [".".join(f.split(".")[:-1]) for f in file_test]

        self.trials_train = file_train
        self.trials_test = file_test

        # Get all features
        files_features = self.get_files(dir_features, split)

        X_all, Y_all = [], []
        for f in files_features:        
            if "Split_" in os.listdir(dir_features)[-1]:
                data_tmp = sio.loadmat( closest_file("{}{}/{}".format(dir_features,split, f)) )
            else:
                data_tmp = sio.loadmat( closest_file("{}/{}".format(dir_features, f)) )
            X_all += [ data_tmp[feature_type].astype(np.float32) ]
            Y_all += [ np.squeeze(data_tmp["Y"]) ]

        # Make sure axes are correct (TxF not FxT for F=feat, T=time)
        if X_all[0].shape[0]!=Y_all[0].shape[0]:
            X_all = [x.T for x in X_all]
        self.n_features = X_all[0].shape[1]
        self.n_classes = len(np.unique(np.hstack(Y_all)))

        # Make sure labels are sequential
        if self.n_classes != np.hstack(Y_all).max()+1:
            Y_all = utils.remap_labels(Y_all)
            print("Reordered class labels")

        # Subsample the data
        if sample_rate > 1:
            X_all, Y_all = utils.subsample(X_all, Y_all, sample_rate, dim=0)

        # ------------Train/test Splits---------------------------
        # Split data/labels into train/test splits
        fid2idx = self.fid2idx(files_features)
        X_train = [X_all[fid2idx[f]] for f in file_train if f in fid2idx]
        X_test = [X_all[fid2idx[f]] for f in file_test if f in fid2idx]

        y_train = [Y_all[fid2idx[f]] for f in file_train if f in fid2idx]
        y_test = [Y_all[fid2idx[f]] for f in file_test if f in fid2idx]

        if len(X_train)==0:
            print("Error loading data")

        return X_train, y_train, X_test, y_test
        