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

def fold10cv(data,lamb):
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
    with open(text, "w") as file:
        file.write(path)

class GenomeReader:
    '''
    Reader for a genome. It takes a list of bw files. It reads a bed file of indexes for training regions and returns the signal values for those regions.
        bwpath: path to the directory where the bw files are restored.
        bedpath: path to the bed files that keeps training indexes.
        binsize: size of the bins
        chrsizes: dictionary of chromosome sizes.
    '''
    def __init__(self, bwpath, bedpath, binsize, lambdapath, chrsizepath):
        self.bwpath = bwpath
        self.bedpath = bedpath
        self.binsize = binsize
        self.bwfiles = sorted(glob.glob(os.path.join(self.bwpath,"*.bw")))
        self.nbwpath = len(self.bwfiles)
        self.chrs, self.starts, self.ends = _read_bed(self.bedpath)
        self.nchunks = len(self.chrs)
        self.lambdas = _read_arr(lambdapath)
        self.chrsizes = _read_dict(chrsizepath)

    def get_chunk_size(self, i):
        return int(np.ceil((self.ends[i] - self.starts[i])/self.binsize))

    def get_chunk_number(self):
        return self.nchunks

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

    def get_l0_approx_training(self):
        binvec = np.empty(self.binsize)
        init_size = int(200)
        out = np.empty((self.nbwpath,0))

        for i in range(self.nchunks):
            length = self.ends[i] - self.starts[i]
            binned_length = int(np.ceil((self.ends[i] - self.starts[i])/self.binsize))
            vec = np.empty(length)
            vecbinned = np.empty(binned_length)
            breakpoints = np.empty(init_size, dtype=int)
            iibreakpoints = 0
            
            print("approximating")
            for j, f in enumerate(self.bwfiles):
                print(f)
                _read_from_bigwig_sampled(vecbinned, f, self.chrs[i], self.starts[i], self.ends[i], self.binsize)
                #_read_from_bigwig_binned(vecbinned, f, self.chrs[i], self.starts[i], self.ends[i], self.binsize)
                bps = l0poissonapproximateCondensed(vecbinned, self.lambdas[j])[1][:-1]
                for k in range(len(bps)):
                    s, e = self.starts[i]+bps[k]*self.binsize, self.starts[i]+(bps[k]+1)*self.binsize
                    _read_from_bigwig(binvec, f, self.chrs[i], s, e)
                    if init_size <= iibreakpoints + k:
                        tmp = np.empty(int(init_size * 2), dtype=int)
                        tmp[:init_size] = breakpoints
                        init_size = int(init_size*2)
                        breakpoints = tmp

                    breakpoints[iibreakpoints+k] = self.starts[i]+bps[k]*self.binsize+l0poissonbreakpoint(binvec)
                iibreakpoints += len(bps)
            
            breakpoints = np.unique(breakpoints[:iibreakpoints])
            chunk = np.empty((len(self.bwfiles), len(breakpoints)+1))
            print("refining")
            for j, f in enumerate(self.bwfiles):
                print(f)
                _read_from_bigwig_breakpoints(chunk[j], f, self.chrs[i], self.starts[i], self.ends[i], breakpoints)
            out = np.hstack([out,chunk])

        return out

    def read_l0_approx(self, chr, start, end):
        binvec = np.empty(self.binsize)
        init_size = int(200)
        length = end - start
        binned_length = int(np.ceil((end - start)/self.binsize))
        vec = np.empty(length)
        vecbinned = np.empty(binned_length)
        breakpoints = np.empty(init_size, dtype=int)
        iibreakpoints = 0

        for j, f in enumerate(self.bwfiles):
            _read_from_bigwig_sampled(vecbinned, f, chr, start, end, self.binsize)
            #_read_from_bigwig_binned(vecbinned, f, chr, start, end, self.binsize)
            bps = l0poissonapproximateCondensed(vecbinned, self.lambdas[j])[1][:-1]
            for k in range(len(bps)):
                s, e = start+bps[k]*self.binsize, start+(bps[k]+1)*self.binsize
                if e > self.chrsizes[chr]:
                    e = self.chrsizes[chr]
                _read_from_bigwig(binvec, f, chr, s, e)
                if init_size <= iibreakpoints + k:
                    tmp = np.empty(int(init_size * 2), dtype=int)
                    tmp[:init_size] = breakpoints
                    init_size = int(init_size*2)
                    breakpoints = tmp

                breakpoints[iibreakpoints+k] = start+bps[k]*self.binsize+l0poissonbreakpoint(binvec)
            iibreakpoints += len(bps)
        
        breakpoints = np.unique(breakpoints[:iibreakpoints])
        out = np.empty((len(self.bwfiles), len(breakpoints)+1))
        for j, f in enumerate(self.bwfiles):
            _read_from_bigwig_breakpoints(out[j], f, chr, start, end, breakpoints)
        return out, np.r_[start, breakpoints], np.r_[breakpoints, end]
                    
class Segmentor:
    def __init__(self, bwpath, bedpath, binsize, lambdapath, chrsizepath, outputpath, chunksize = 10000000, nsample=2000, n_components=25, random_state=42, max_iter=2000, covariance_type="full", init_params="kmeans", weight_concentration_prior_type="dirichlet_process"):
        self.reader = GenomeReader(bwpath, bedpath, binsize, lambdapath, chrsizepath)
        self.nchunk = self.reader.get_chunk_number()
        self.nsample = nsample
        self.chunksize = chunksize
        self.chrsizes = _read_dict(chrsizepath)
        self.outputpath = outputpath
        self.rng = np.random.default_rng(seed=42)
        self.gmm = BayesianGaussianMixture(n_components=n_components, random_state=random_state, max_iter=max_iter, covariance_type=covariance_type, init_params=init_params, weight_concentration_prior_type=weight_concentration_prior_type)

    def train_gmm(self):
        traindata = self.reader.get_l0_approx_training().T
        self.gmm.fit(traindata[self.rng.choice(range(traindata.shape[0]), self.nsample, replace=False)])

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
                data = data.T
                if verbose: print(f"\tCarrying out predictions")
                labels = self.gmm.predict(data)
                if verbose: print(f"\tExpanding labels")
                extend_labels(labelchrs, labels, interval_starts, interval_ends)
            bed += convert_to_bed(labelchrs, i)
        if verbose: print(f"\tWriting labels")
        write_text(bed, self.outputpath)

