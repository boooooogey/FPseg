import numpy as np
import glob, os
import pyBigWig
from .l0approximator.poissonfunctions import l0poissonapproximateCondensed, l0poissonbreakpoint
from sklearn.mixture import BayesianGaussianMixture
from distinctipy import distinctipy

def _read_bed(path):
    '''
    Read the content of a bed file
        path: the path to a bed file
    '''
    with open(path,'r') as file:
        data = file.readlines()
    data = [i.split() for i in data]
    chrs = [i[0] for i in data]
    s = [int(i[1]) for i in data]
    e = [int(i[2]) for i in data]
    return chrs, s, e

def _read_from_bigwig(vec, path, chr, start, end):
    '''
    Read from a bigwig file
        vec: vec is an array that is overwritten with the values from bigwig.
        path: path of the bigwig file
        chr: chromosome number.
        start: starting index in the chromosome number.
        end: ending index in the chromosome number.
    '''
    with pyBigWig.open(path) as bigwig:
        vec[:(end-start)] = np.array(bigwig.values(chr,start,end))
    np.nan_to_num(vec,copy=False,nan=0)

def mean_nan_to_zero(arr):
    mask = np.isnan(arr)
    if np.all(mask):
        return 0
    else:
        return np.nanmean(arr)*np.sum(~np.isnan(arr))/len(arr)

def _read_from_bigwig_breakpoints(vec, path, chr, start, end, bps):
    '''
    Read from a bigwig file
        vec: vec is an array that is overwritten with the values from bigwig.
        path: path of the bigwig file
        chr: chromosome number.
        start: starting index in the chromosome number.
        end: ending index in the chromosome number.
    '''
    with pyBigWig.open(path) as bigwig:
        if len(bps) == 0:
            vec[0] = mean_nan_to_zero(bigwig.values(chr,start,end))
        else:
            vec[0] = mean_nan_to_zero(bigwig.values(chr,start,bps[0]))
            for i in range(len(bps)-1):
                arr = bigwig.values(chr,bps[i],bps[i+1])
                vec[i+1] = mean_nan_to_zero(bigwig.values(chr,bps[i],bps[i+1]))
            vec[len(bps)] = mean_nan_to_zero(bigwig.values(chr,bps[-1],end))

def _read_from_bigwig_binned(vec, path, chr, start, end, bin_size):
    '''
    Read from a bigwig file and return binned values
        vec: vec is an array that is overwritten with the values from bigwig.
        path: path of the bigwig file
        chr: chromosome number.
        start: starting index in the chromosome number.
        end: ending index in the chromosome number.
        bin_size: size of the bins.
    '''
    with pyBigWig.open(path) as bigwig:
        vals = np.array(bigwig.values(chr,start,end))
    np.nan_to_num(vals,copy=False,nan=0)
    if len(vals) % bin_size == 0:
        vec[:] = vals.reshape(-1,bin_size).mean(axis=1)
    else:
        vec[:-1] = vals[:int(bin_size * (len(vec)-1))].reshape(-1,bin_size).mean(axis=1)
        vec[-1] = vals[int(bin_size * (len(vec)-1)):].mean()
        #vec[:] = np.pad(vals, (0, bin_size - len(vals) % bin_size)).reshape(-1,bin_size).mean(axis=1) 

def _read_from_bigwig_sampled(vec, path, chr, start, end, bin_size):
    with pyBigWig.open(path) as bigwig:
        vals = np.array(bigwig.values(chr,start,end))
    vec[:] = vals[::bin_size]
    np.nan_to_num(vec, copy=False, nan=0)

def _read_dict(path):
    with open(path, 'r') as file:
        data = file.readlines()
    data = [i.strip().split() for i in data]
    data = {i[0]:int(i[1]) for i in data}
    return data

def _read_arr(path):
    with open(path, 'r') as file:
        data = file.readlines()
    data = [float(i) for i in data]
    return data

def _fold_10_cv(data,lamb):
    k = 10
    ii = list(range(len(data)))
    np.random.shuffle(ii)
    fold_size = len(data)//k
    fold_val = np.empty(10)
    for i in range(k):
        tmp = np.copy(data)
        if i != 9:
            iitmp = ii[i*fold_size:(i+1)*fold_size]
        else:
            iitmp = ii[i*fold_size:]
        target = tmp[iitmp]
        tmp[iitmp] = np.mean(np.delete(tmp,iitmp))
        tmpseg = FPseg.poisseg(tmp,lamb)
        fold_val[i] = poisson(target, tmpseg[iitmp])
        if np.isinf(fold_val[i]):
            embed()
    return np.mean(fold_val)

def pick_lambda(data, lambdas):
    pass

def _bed_to_np(bed, chr, start, end):
    bed = [i.strip().split() for i in bed]
    bed = [[*i[:-2], np.array(i[-2].split(','), dtype=int), np.array(i[-1].split(','), dtype=int)] for i in bed]
    bed = filter(lambda x: x[0] == chr, bed)
    labels = np.empty(end-start, dtype=int) 
    for i in bed:
        ends = i[-1] + i[-2]
        ii = np.where(np.logical_and(ends > start, i[-1] < end))
        startii = i[-1][ii]-start
        numii = i[-2][ii]
        for k in range(len(startii)):
            if startii[k] < 0:
                s = 0
            else:
                s = startii[k]
            if startii[k]+numii[k] > len(labels):
                e = len(labels)
            else:
                e = startii[k]+numii[k]
            
            labels[s:e] = int(i[3])
    return labels

def extend_labels(labelchrs, label, interval_starts, interval_ends):
    for l, s, e in zip(label, interval_starts, interval_ends):
        labelchrs[s:e] = l

def hextodecimal(hexc):
    return f"{int(hexc[1:3],16)},{int(hexc[3:5],16)},{int(hexc[5:7],16)}"

def convert_to_bed(labels,chr):
    unique_labels = np.unique(labels)
    colors = distinctipy.get_colors(len(unique_labels))
    colors = [distinctipy.get_hex(i) for i in colors]
    data = {i:[(0,0)] for i in unique_labels}
    write = False
    currs = 0
    currl = labels[0]
    for i,l in enumerate(labels):
        if l != currl or i == len(labels) - 1:
            data[currl].append((i-currs,currs))
            currl = l
            currs = i
    bbed = ""
    comma = ","
    for k,i in enumerate(data):
        color = hextodecimal(colors[k])
        data[i].append((1,len(labels)-1))
        bbed = bbed + chr + "\t0\t" + str(len(labels)) + "\t" + str(i) + "\t1000\t.\t0\t" + str(len(labels)) + "\t" + color + "\t" + str(len(data[i])) + "\t" + comma.join([str(i[0]) for i in data[i]]) + "\t" + comma.join([str(i[1]) for i in data[i]]) + "\n"
    return bbed

def write_text(text, path):
    with open(path, "w") as file:
        file.write(text)

def copy_binned(outvec, invec, bin_size):
    if len(invec) % bin_size == 0:
        outvec[:] = invec.reshape(-1, bin_size).mean(axis=1)
    else:
        outvec[:-1] = invec[:int(bin_size * (len(outvec)-1))].reshape(-1, bin_size).mean(axis=1)
        outvec[-1] = invec[int(bin_size * (len(outvec)-1)):].mean()

class GenomeReader:
    '''
    Reader for a genome. It takes a list of bw files. It reads a bed file of indexes for training regions and returns the signal values for those regions.
        bwpath: path to the directory where the bw files are restored.
        bedpath: path to the bed files that keeps training indexes.
        binsize: size of the bins
        chrsizes: dictionary of chromosome sizes.
    '''
    def __init__(self, bwpath, binsize, lambdapath, chrsizepath, include):
        self.bwpath = bwpath
        self.binsize = binsize
        self.bwfiles = sorted(glob.glob(os.path.join(self.bwpath,"*.bw")))
        self.nbwpath = len(self.bwfiles)
        self.lambdas = _read_arr(lambdapath)
        self.chrsizes = _read_dict(chrsizepath)
        self.include = include

    def nbw(self):
        return len(self.bwfiles)

    def get_binned_training_mat(self, i):
        '''
        Read the ith training chunk as a matrix with binning.
        '''
        out = np.empty((self.nbwpath, self.get_chunk_size(i)))
        for j, f in enumerate(self.bwfiles):
            _read_from_bigwig_binned(out[j], f, self.chrs[i], self.starts[i], self.ends[i], self.binsize)
        return out, self.starts[i], self.ends[i], self.chrs[i]

    def get_binned_training_vec(self, vec, i, j):
        '''
        Read the ith training chunk from jth bw file as a matrix with binning.
        '''
        _read_from_bigwig_binned(vec[:], self.bwfiles[j], self.chrs[i], self.starts[i], self.ends[i], self.binsize)
        return self.start[i], self.ends[i], self.chrs[i], self.bwfiles[j]

    def get_exact_training_mat(self, i):
        '''
        Read the ith training chunk as a matrix without binning.
        '''
        out = np.empty((self.nbwpath, self.ends[i] - self.starts[i]))
        for j, f in enumerate(self.bwfiles):
            _read_from_bigwig(out[j], f, self.chrs[i], self.starts[i], self.ends[i])
        return out, self.start[i], self.ends[i], self.chrs[i]

    def get_exact_training_vec(self, vec, i, j):
        '''
        Read the ith training chunk from jth bw file as a matrix without binning.
        '''
        _read_from_bigwig(vec[:], self.bwfiles[j], self.chrs[i], self.starts[i], self.ends[i])
        return self.start[i], self.ends[i], self.chrs[i], self.bwfiles[j]

    def get_bin(self, chr, index, j, offset):
        out = np.empty(self.binsize)
        _read_from_bigwig(out, self.bwfiles[j], chr, offset + index * self.binsize, offset + (index+1) * self.binsize)
        return out

    def read_l0_approx(self, chr, start, end):
        init_size = int(200)
        if end > self.chrsizes[chr]:
            end = self.chrsizes[chr]
        binvec = np.empty(self.binsize)
        length = end - start
        binned_length = int(np.ceil((end - start)/self.binsize))
        mat = np.empty((len(self.bwfiles), length))
        for i, f in enumerate(self.bwfiles):
            _read_from_bigwig(mat[i], f, chr, start, end)
        vecbinned = np.empty(binned_length)
        breakpoints = np.empty(init_size, dtype=int)
        iibreakpoints = 0

        for j, f in enumerate(self.bwfiles):
            copy_binned(vecbinned, mat[j], self.binsize)
            bps = l0poissonapproximateCondensed(vecbinned, self.lambdas[j])[1][:-1]
            for k in range(len(bps)):
                s, e = bps[k]*self.binsize, (bps[k]+1)*self.binsize #start+bps[k]*self.binsize, start+(bps[k]+1)*self.binsize
                if e > self.chrsizes[chr]:
                    e = self.chrsizes[chr]
                if init_size <= iibreakpoints + k:
                    tmp = np.empty(int(init_size * 2), dtype=int)
                    tmp[:init_size] = breakpoints
                    init_size = int(init_size*2)
                    breakpoints = tmp

                breakpoints[iibreakpoints+k] = bps[k]*self.binsize+l0poissonbreakpoint(mat[j,s:e]) #start+bps[k]*self.binsize+l0poissonbreakpoint(mat[j,s:e])
            iibreakpoints += len(bps)
        
        breakpoints = np.unique(breakpoints[:iibreakpoints])
        if self.include == "lengths":
            out = np.empty((len(breakpoints)+1, len(self.bwfiles)+1))
            mat = mat.T
            if len(breakpoints) == 0:
                out[0,:-1] = mat.mean(axis=0)
                out[0,-1] = end - start
            else:
                out[0,:-1] = mat[0:breakpoints[0]].mean(axis=0)
                out[0,-1] = breakpoints[0]
                for i in range(1,len(breakpoints)):
                    out[i,:-1] = mat[breakpoints[i-1]:breakpoints[i]].mean(axis=0)
                    out[i,-1] = breakpoints[i]-breakpoints[i-1]
                out[-1,:-1] = mat[breakpoints[-1]:].mean(axis=0)
                out[-1,-1] = end - breakpoints[-1] - start
        elif self.include == "neighbors":
            out = np.empty((len(breakpoints)+1, 3*(len(self.bwfiles)+1)))
            k = len(self.bwfiles)+1
            mat = mat.T
            if len(breakpoints) == 0:
                out[0,k:2*k-1] = mat.mean(axis=0)
                out[0,2*k-1] = end - start
                out[0,:k] = 0
                out[0,2*k:] = 0 
            else:
                out[0,k:2*k-1] = mat[0:breakpoints[0]].mean(axis=0)
                out[0,2*k-1] = breakpoints[0]
                for i in range(1,len(breakpoints)):
                    out[i,k:2*k-1] = mat[breakpoints[i-1]:breakpoints[i]].mean(axis=0)
                    out[i,2*k-1] = breakpoints[i]-breakpoints[i-1]
                out[-1,k:2*k-1] = mat[breakpoints[-1]:].mean(axis=0)
                out[-1,2*k-1] = end - breakpoints[-1] - start
                out[1:, :k] = out[:-1,k:2*k]
                out[0, :k] = 0
                out[:-1,2*k:] = out[1:,k:2*k]
                out[-1, 2*k:] = 0 
        else:
            out = np.empty((len(breakpoints)+1, len(self.bwfiles)))
            mat = mat.T
            if len(breakpoints) == 0:
                out[0] = mat.mean(axis=0)
            else:
                out[0] = mat[0:breakpoints[0]].mean(axis=0)
                for i in range(1,len(breakpoints)):
                    out[i] = mat[breakpoints[i-1]:breakpoints[i]].mean(axis=0)
                out[-1] = mat[breakpoints[-1]:].mean(axis=0)


        return out, np.r_[start, start + breakpoints], np.r_[start + breakpoints, end]
                    
class Segmentor:
    def __init__(self, bwpath, binsize, lambdapath, chrsizepath, bedpath, outputpath, include = None, chunksize = 10000000, nsample=2000, n_components=25, random_state=42, max_iter=2000, covariance_type="full", init_params="kmeans", weight_concentration_prior_type="dirichlet_process"):
        self.reader = GenomeReader(bwpath, binsize, lambdapath, chrsizepath, include)
        self.nsample = nsample
        self.chunksize = chunksize
        self.bedpath = bedpath
        self.chrs, self.starts, self.ends = _read_bed(self.bedpath)
        self.nchunk = len(self.chrs)
        self.chrsizes = _read_dict(chrsizepath)
        self.outputpath = outputpath
        self.rng = np.random.default_rng(seed=42)
        self.gmm = BayesianGaussianMixture(n_components=n_components, random_state=random_state, max_iter=max_iter, covariance_type=covariance_type, init_params=init_params, weight_concentration_prior_type=weight_concentration_prior_type)

    def train_gmm(self):
        traindata = np.empty((self.nchunk*self.nsample, self.reader.nbw()))
        for i, c, s, e in zip(range(self.nchunk), self.chrs, self.starts, self.ends):
            tmp = self.reader.read_l0_approx(c,s,e)[0]
            traindata[(i*self.nsample):((i+1)*self.nsample)] = tmp[self.rng.choice(range(tmp.shape[0]), self.nsample, replace=False)]
        self.gmm.fit(traindata)

    def label_data(self, verbose=False):
        bed = ""
        for i in self.chrsizes.keys():
            labelchrs = np.empty(self.chrsizes[i], dtype=int)
            ii = 0
            cont = True
            while cont:
                s, e = ii, ii+self.chunksize
                ii += self.chunksize
                if e > self.chrsizes[i]:
                    e = self.chrsizes[i]
                    cont = False
                if verbose: print(f"Labeling chromosome {i}: {s}-{e}")
                if verbose: print(f"\tReading L0 approximation")
                data, interval_starts, interval_ends = self.reader.read_l0_approx(i, s, e)
                if verbose: print(f"\tCarrying out predictions")
                labels = self.gmm.predict(data)
                if verbose: print(f"\tExpanding labels")
                extend_labels(labelchrs, labels, interval_starts, interval_ends)
            bed += convert_to_bed(labelchrs, i)
        if verbose: print(f"\tWriting labels")
        write_text(bed, self.outputpath)

