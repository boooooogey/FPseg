from poissonL0segmentation import poisseg, poissegbreakpoints
import numpy as np
import math
import os, glob
from sklearn.mixture import BayesianGaussianMixture
import pyBigWig

def _read_from_bigwig(path,chr,from,to,bin_size):
    with pyBigWig.open(path) as bigwig:
        vals = bigwig.values(chr,from,to)
    if bin_size == 1:
        return vals
    else:
        return vals.reshape(-1,bin_size).mean(axis=1)

def _read_data(bw_files,chr,from,to,bin_size):
    # starts with 0 index and index "to" is discarded
    mat = np.empty((0,to-from))
    for bw_file in bw_files:
        mat = np.vstack([mat,_read_from_bigwig(bw_file,chr,from,to,bin_size)])
    return mat

def _check_dict_keys(dict1,dict2,keys):
    for i in keys:
        if dict1[i] != dict2[i]:
            raise Exception("Some of the chromosomes in different BigWig files have different number of indices.")

def _check_chromosomes(bw_files):
    chr_names = set()
    for i, bw_file in enumerate(bw_files):
        with pyBigWig.open(bw_file) as bw:
            chr_dict_curr = bw.chroms()
        if i == 0:
            chr_dict = chr_dict_curr
            chr_names = set(chr_dict_curr.keys())
        else:
            chr_names = chr_names.intersection(set(chr_dict_curr.keys()))
            _check_dict_keys(chr_dict,chr_dict_curr,chr_names)
            chr_dict = {i:chr_dict[i] for i in chr_names}
    return chr_dict

def _reduce_data_mean(data,start_indices,end_indices):
    n = len(end_indices)
    reduced = np.empty(n,data.shape[0])
    for i,s,e in zip(range(n),start_indices,end_indices):
        reduced[i] = np.mean(data[,s:e+np.uint64(1)])
    return reduced

def _reduce_data_mean_length(data,start_indices,end_indices):
    n = len(end_indices)
    reduced = np.empty(n,data.shape[0]+1)
    for i,s,e in zip(range(n),start_indices,end_indices):
        reduced[i] = np.r_[np.mean(data[,s:e+np.uint64(1)]),e-s+1)]
    return reduced

def _reduce_data_neighbors(data,start_indices,end_indices):
    n = len(end_indices)
    p = int(data.shape[0] + 1)
    reduced = np.empty(n,p*3)
    
    prev = np.zeros(p)
    s, e = start_indices[0], end_indices[0]
    curr = np.r_[np.mean(data[,s:e+np.uint64(1)]),e-s+1)]

    for i,s,e in zip(range(n-1),start_indices[:n-1],end_indices[:n-1]):
        ns, ne = start_indices[i+1], end_indices[i+1]

        next = np.r_[np.mean(data[,ns:ne+np.uint64(1)]),ne-ns+1)]

        reduced[i,:p] = prev
        reduced[i,p:2*p] = curr
        reduced[i,2*p:3*p] = next

        curr = next
        prev = curr

    reduced[n-1,:p] = prev
    reduced[n-1,p:2*p] = curr
    reduced[n-1,2*p:3*p] = np.zeros(p)

    return reduced

def _find_all_breakpoints(data, each_lambda):
    n = int(data.shape[1]*0.01)
    start_points = np.emtpy(0)
    end_points = np.empty(0)
    #vals = np.empty(0)
    for row, lamb in zip(data,each_lambda):
        ss, es, _ = poissegbreakpoints(row,lamb)
        start_points = np.unique(np.hstack([np.empty(0),ss]))
        end_points = np.unique(np.hstack([np.empty(0),es]))
        #vals = np.unique(np.hstack([np.empty(0),vals]))
    return start_points, end_points#, vals

class FPseg:
    def __init__(self, *, path=None, bin_size=200, batch_size=10000000, reduction_type = "neighbors", 
            track_lambda = None, sample_ratio = 0.2, n_components=25, covariance_type='full', tol=1e-3,
            reg_covar=1e-6, max_iter=1000, n_init=1, init_params='kmeans',
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=None,mean_precision_prior=None, 
            mean_prior=None,degrees_of_freedom_prior=None,covariance_prior=None,
            random_state=None, verbose=0,verbose_interval=10):
        #Set path to BigWig files
        self.path = path
        self.bin_size = bin_size
        self.batch_size = batch_size
        self.track_lambda = track_lambda
        self.index = 0
        self.sample_ratio = sample_ratio
        self.reduction_type = reduction_type
        self.cont_iteration_ = True
        self.labels = None
        if random_state is not None:
            self.rng = np.random.default_rng(random_state)
        #Initialize Bayesian Gaussian mixture model
        self.bgmm = BayesianGaussianMixture(n_components=n_components, covariance_type=covariance_tyep, tol=tol,
            reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
            weight_concentration_prior_type=weight_concentration_prior_type,
            weight_concentration_prior=weight_concentration_prior,mean_precision_prior=mean_precision_prior, 
            mean_prior=mean_prior,degrees_of_freedom_prior=degrees_of_freedom_prior,covariance_prior=covariance_prior,
            random_state=random_state, warm_start=True, verbose=verbose,verbose_interval=verbose_interval)

    def _check_path(self):
        self.bw_files = sorted(glob.glob(os.path.join(self.path,"*.bw")))
        if len(self.bw_files) == 0:
            raise Exception("There are no BigWig files in the directory.") 
        self.chr_dict = _check_chromosomes(self.bw_files)
        self.chr_list = sorted(list(self.chr_dict.keys()),key=lambda x: int(x[3:]) if x[3:].isnumeric() else ord(x[3:]))
        self.chr = 0
        self.num_tracks = len(self.bw_files)

    def _check_parameters(self):
        if self.path is not None:
            self._check_path()
        if self.track_lambda is None:
            self.track_lambda = _pick_lambdas()
        else:
            if len(self.track_lambda) != len(self.chr_dict.keys()) and len(self.track_lambda) != 1:
                raise ValueError("Expected length of 'track_lambda' to be "
                                f"1 or equal to the number of tracks ({self.num_tracks}) "
                                f"but got {len(self.track_lambda)}")
        if isinstance(self.bin_size,float) and not self.bin_size.is_integer():
            raise ValueError("Expected a positive integer for 'bin_size' "
                            f"got {bin_size}.")
        else:
            self.bin_size = int(self.bin_size)
        if isinstance(self.batch_size,float) and not self.batch_size.is_integer():
            raise ValueError("Expected a positive integer for 'batch_size' "
                            f"got {batch_size}.")
        else:
            self.batch_size = int(self.batch_size)
        if self.bin_size <= 0:
            raise ValueError("Expected a positive value for 'bin_size' "
                            f"got {bin_size}.")
        if self.batch_size <= 0:
            raise ValueError("Expected a positive value for 'batch_size' "
                            f"got {batch_size}.")
        if self.reduction_type not in ["mean","mean_length","neighbors"]:
            raise ValueError("Invalid value for 'reduction_type': %s"
                             "'reduction_type' should be in "
                             "['mean', 'mean_length', 'neighbors']"
                             % self.reduction_type)
        if not(self.sample_ratio > 0 and self.sample_ratio <= 1):
            raise ValueError("Expected a value between 0 and 1 "
                             "for 'sample_ratio' got %.3f." 
                             % self.sample_ratio)

    def _read_batch(self,data=None):
        if data == None:
            curr_chr = self.chr_list[self.chr]
            if self.index + self.batch_size < self.chr_dict[curr_chr]:
                data = _read_data(self.bw_files,curr_chr,self.index,self.index+self.batch_size,self.bin_size)
            else:
                data = _read_data(self.bw_files,curr_chr,self.index,self.chr_dict[curr_chr],self.bin_size)
                self.chr += 1
                if self.chr => len(self.chr_list):
                    self.cont_iteration_ = False
        self.start_points, self.end_points = _find_all_breakpoints(data, self.track_lambda)
        if self.reduction_type == 'mean':
            self.batch = _reduce_data_mean(data, self.start_points, self.end_points):
        if self.reduction_type == 'mean_length':
            self.batch = _reduce_data_mean_length(data, self.start_points, self.end_points):
        if self.reduction_type == 'neighbors':
            self.batch = _reduce_data_neighbors(data, self.start_points, self.end_points):

    def fit(self,data=None):
        if data is None:
            self.cont_iteration_ = True
            while self.cont_iteration:
                self._read_batch()
                #subsample
                ii = self.rng.choice(range(self.batch.shape[0]),math.ceil(self.batch.shape[0]*self.sample_ratio))
                self.bgmm.fit(self.batch[ii])
        else:
            self._read_batch(data)
            self.bgmm.fit(self.batch)
        return self

    def predict(self,data=None):
        if data is None:
            pass
            self.labels = None
        else:
            self._read_batch(data)
            self.labels = self.bgmm.predict(self.batch)
        return self.labels

    def fit_predict(self,data=None):
        self.fit(data)
        return self.predict(data)
