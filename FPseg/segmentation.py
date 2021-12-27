import numpy as np
import pyBigWig
from .l0approximator.poissonfunctions import l0poissonapproximate, l0poissonapproximateCondensed, l0poissonbreakpoint 
import ctypes
import multiprocessing as mp
from contextlib import closing
from sklearn.preprocessing import StandardScaler
import pickle

def _to_numpy_array(mp_arr, N):
    '''
    Get a numpy array using memory of a multiprocessing array.
    mp_arr is a multiprocessing array.
    N is the dimensions of the numpy array if
        N = 1 then an one dimensional array is returned.
        N > 1 then a two dimensional array of shape N by M is returned. The length of the multiprocessing array must be divisible by N.
        N is a tuple then an array of shape N is returned.
    '''
    if N == 1:
        return np.frombuffer(mp_arr.get_obj())
    else:   
        arr = np.frombuffer(mp_arr.get_obj())
        try:
            return arr.reshape(*N)
        except:
            return arr.reshape(N, -1)

def _read_from_bigwig_parallel(i, path, chr, start, end, N):
    '''
    Reads from a Bigwig file into a row of the shared matrix.
    i is row number.
    path is the path to the BigWig file.
    chr is the chromosome name.
    start is the starting index.
    end is the ending index.
    N is the first dimension of the shared matrix. Corresponds to the number of BigWig files.
    '''
    mat = _to_numpy_array(shared_arr, N+1)
    with pyBigWig.open(path) as bigwig:
        mat[i, :(end-start)] = np.array(bigwig.values(chr,start,end))
    np.nan_to_num(mat[i],copy=False,nan=0)

def _readbw_parallel(mat, bwfiles, chrom, start, end, number_of_cores):
    '''
    Reads BigWig files into rows of the matrix.
    mat is a multiprocessing array. mat.shape = (N, end-start < )
    bwfiles is a list of BigWig files (paths).
    chrom is the chromosome.
    start is the starting index.
    end is the ending index.
    '''
    def init(shared_arr_):
        global shared_arr
        shared_arr = shared_arr_ # must be inherited, not passed as an argument
    
    N = len(bwfiles)
    M = end - start
    ncores = min(N, mp.cpu_count(), number_of_cores)

    # write to arr from different processes
    with closing(mp.Pool(ncores, initializer=init, initargs=(mat,))) as p:
        try:
            n = len(start)
            p.starmap_async(_read_from_bigwig_parallel, [(i, bwfiles[i], chrom, start[i], end[i], N) for i in range(N)])
        except:
            p.starmap_async(_read_from_bigwig_parallel, [(i, bwfiles[i], chrom, start, end, N) for i in range(N)])
    p.join()

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

def _mean_nan_to_zero(arr):
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
            vec[0] = _mean_nan_to_zero(bigwig.values(chr,start,end))
        else:
            vec[0] = _mean_nan_to_zero(bigwig.values(chr,start,bps[0]))
            for i in range(len(bps)-1):
                arr = bigwig.values(chr,bps[i],bps[i+1])
                vec[i+1] = _mean_nan_to_zero(bigwig.values(chr,bps[i],bps[i+1]))
            vec[len(bps)] = _mean_nan_to_zero(bigwig.values(chr,bps[-1],end))

def _copy_binned(outvec, invec, bin_size, size):
    '''
    Copy invec to outvec by binning.
    size is the number of cells occupied in invec.
    '''
    invec = invec[:size]
    if len(invec) % bin_size == 0:
        outvec[:] = invec.reshape(-1, bin_size).mean(axis=1)
    else:
        outvec[:-1] = invec[:int(bin_size * (len(outvec)-1))].reshape(-1, bin_size).mean(axis=1)
        outvec[-1] = invec[int(bin_size * (len(outvec)-1)):].mean()

def _return_break_points_from_a_single_bw(chrom_size, binsize, l, N, M, size, i):
    '''
    Calculate break points from each row 
    '''
    mat = _to_numpy_array(shared_arr, N+1)

    if binsize != 1:
        binned_length = int(np.ceil(size/binsize))
        vecbinned = np.empty(binned_length)
        _copy_binned(vecbinned, mat[i], binsize, size)
        bps = l0poissonapproximateCondensed(vecbinned, l)[1][:-1]
        for k in range(len(bps)):
            s, e = bps[k] * binsize, (bps[k]+1)*binsize
            if e > chrom_size:
                e = chrom_size
            bps[k] = bps[k] * binsize + l0poissonbreakpoint(mat[i, s:e])
    else:
        bps = l0poissonapproximateCondensed(mat[i], l)[1][:-1]
    
    mat[N,bps] = 1

def _return_break_points_reduced_parallel(mat, chrom, bin_size, lambdas, chrsizes, N, M, size, number_of_cores):
    '''
    Calculates the all breakpoints.
    mat is the matrix whose rows are epigenetic files.
    chrom is the chromosome
    bin_size is the size of the bins.
    lambdas is the list of the hyper-parameters for the L0 segmentation.
    N is the number of epigenetic tracks
    M is the second dimension of the shared matrix
    size is the length of the each track
    '''
    
    chrom_size = chrsizes[chrom]
    
    def init(shared_arr_):
        global shared_arr
        shared_arr = shared_arr_ # must be inherited, not passed as an argument
        
    results = _to_numpy_array(mat, N+1)
    results[N, :] = 0
    results[N, 0] = 1
    
    ncores = min(N, mp.cpu_count(), number_of_cores)

    with closing(mp.Pool(ncores, initializer=init, initargs=(mat,))) as p:
        p.starmap_async(_return_break_points_from_a_single_bw, [(chrom_size, bin_size, lambdas[i], N, M, size, i) for i in range(N)])
    p.join()
    return np.where(results[N,:] == 1)[0]

def _fill_reduced_parallel(array_ii, selection_ii, bps, N, M, offset, j):
    '''
    fill the shared matrix with selected segments from L0 segmentation.
    array_ii is the indeces of selected segments for the process j
    selection_ii is the selected segments for the training.
    bps is the starting indices for each segment.
    N is the number of epigenetic tracks.
    M is the second dimension of the shared matrix.
    offset is the starting index for the function to start filling the shared matrix.
    j is the process number.
    '''
    mat = _to_numpy_array(shared_arr, N+1)
    reduced = _to_numpy_array(shared_reduce, 3*(N+1))
    for i in array_ii:
        selection_i = selection_ii[i]
        
        if selection_i + 1 >= len(bps):
            s, e = bps[selection_i], M
        else:
            s, e = bps[selection_i], bps[selection_i+1]
        tmp = mat[0:N,s:e].mean(axis=1)
        length = e - s
        
        reduced[N+1:2*(N+1)-1, offset+i] = tmp
        reduced[2*(N+1)-1, offset+i] = length
        
        if s == 0:
            reduced[0:N+1, offset + i] = 0
        elif i-1 >= 0 and selection_ii[i-1] + 1 == selection_i:
            reduced[2*(N+1):3*(N+1)-1, offset + i-1] = tmp
            reduced[3*(N+1)-1, offset + i-1] = length
        else:
            sprev, eprev = bps[selection_i-1], bps[selection_i]
            reduced[0:N, offset + i] = mat[0:N,sprev:eprev].mean(axis=1)
            reduced[N, offset + i] = eprev - sprev
            
        if e == M:
            reduced[2*(N+1):, i] = 0
        elif i+1 < len(selection_ii) and selection_ii[i+1] == selection_i + 1:
            reduced[0:N, offset + i+1] = tmp
            reduced[N, offset + i+1] = length
        else:
            if selection_i+2 >= len(bps):
                snext, enext = bps[selection_i+1], M
            else:
                snext, enext = bps[selection_i+1], bps[selection_i+2]
            reduced[2*(N+1):3*(N+1)-1, offset + i] = mat[0:N,snext:enext].mean(axis=1)
            reduced[3*(N+1)-1, offset + i] = enext - snext

def _return_reduced_data_parallel(mat, reduced, bps, N, M, offset, selected_ii, number_of_cores):
    '''
    Fills reduced with mat using segments. Returns number of samples selected.
    mat is the data matrix.
    reduced is the smaller data matrix summarized over selected segments.
    bps is the starting indeces of the segments.
    N is the number of epigenetic segments.
    M is the second dimension of the mat.
    offset is the starting index to right reduced. (column)
    selected_ii is the selected segments.
    '''
    
    def init(shared_arr_, shared_reduced_):
        global shared_arr, shared_reduce
        shared_arr = shared_arr_ # must be inherited, not passed as an argument
        shared_reduce = shared_reduced_
        
    n_sample = len(selected_ii)
    if n_sample > len(bps):
        n_sample = len(bps)

    n_cpus = mp.cpu_count()
    n_cpus = min(n_cpus, number_of_cores)

    n_size = int(np.ceil(n_sample / n_cpus))
    array_ii = [range(i* n_size, min((i+1)*n_size, n_sample)) for i in range(n_cpus)]

    with closing(mp.Pool(n_cpus, initializer=init, initargs=(mat,reduced,))) as p:
        p.starmap_async(_fill_reduced_parallel, [(array_ii[i], selected_ii, bps, N, M, offset, i) for i in range(n_cpus)])
    p.join()
    return n_sample

def _bed_to_np(bed, chr, start, end):
    bed = [i.strip().split() for i in bed]
    bed = [[*i[:-2], np.array(i[-2].split(','), dtype=int), np.array(i[-1].split(','), dtype=int)] for i in bed]
    bed = filter(lambda x: x[0] == chr, bed)
    labels = [''] * (end-start)#np.empty(end-start, dtype=int) 
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
            
            labels[s:e] = i[3]
    return labels

def _return_target_ii(end, chunk_size, start = 0):
    n = int(np.ceil((end-start) / chunk_size))
    tmp = np.concatenate([np.arange(start, end, chunk_size), [end]])
    return tmp[:n], tmp[1:]

def _prepare_for_classifier(mat, N, number_of_sample_found):
    scaler = StandardScaler(copy=False)
    training = _to_numpy_array(mat, N)[:,:number_of_sample_found].T
    np.log1p(training, out = training)
    scaler.fit_transform(training)

def find_the_best_region(bw_path, chrom, window, method = "coverage"):
    bw = pyBigWig.open(bw_path)
    nwindow = int(np.ceil(bw.chroms()[chrom] / window))
    max_val = -1
    max_curr = -1
    for i in range(nwindow):
        tmp = bw.stats(chrom, i*window, min(bw.chroms()[chrom], (i+1)*window), type = method)[0]
        #print(tmp)
        if tmp is None: tmp = 0
        if max_val < tmp:
            max_val = tmp
            max_curr = i
    return max_curr*window, min(bw.chroms()[chrom], (max_curr+1)*window)

def poisson_error(ypred, y):
    offset = 1
    #return -(1-w)@(y*np.log(ypred+offcset) - ypred + np.log(y+offset)/2 + y * np.log(y+offset) - y )
    return np.sum(y*np.log(y+offset)-y*np.log(ypred+offset)-y+ypred)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def validate_lambda(vec, lamb, w = 10, show = False):
    smoothed = moving_average(vec, w)
    if show:
        figure, axis = plt.subplots(2,2,figsize=(10,10))
        axis[0,0].plot(range(len(vec)), vec, 'o')
        axis[0,1].plot(range(len(smoothed)), smoothed, 'o')
    segmented = l0poissonapproximate(vec, lamb)
    if show:
        axis[1,0].plot(range(len(vec)), vec, 'o')
        axis[1,0].plot(range(len(segmented)), segmented, linewidth=3)
        axis[1,1].plot(range(len(smoothed)), smoothed, 'o')
        axis[1,1].plot(range(len(segmented)), segmented, linewidth=3)
    return poisson_error(smoothed, segmented) #poisson_error(segmented, smoothed)

def _find_lambda(vec, lambs, w = 10, show = False):
    obj = np.empty_like(lambs)
    for k,l in enumerate(lambs):
        obj[k] = validate_lambda(vec, l, w=w, show=show) 
    return obj

def _pick_lambda(reader, chrom, lambda_window, binsize, search_end = 3, search_start = 0, num = 100, second_window = 50, verbose = True, path = None): #search_end and search_start are logarithmic

    starts = np.empty(reader.number_of_bigwig_files, dtype=int)
    ends = np.empty(reader.number_of_bigwig_files, dtype=int)
    for l in range(reader.number_of_bigwig_files):
        starts[l], ends[l] = find_the_best_region(reader.bigwig_file_list[l], chrom, lambda_window)
        if verbose: print(f"For {reader.bigwig_file_list[l]} using {chrom}:{starts[l]}-{ends[l]} for lambda selection.")

    matrix_shared = mp.Array(ctypes.c_double, (reader.number_of_bigwig_files+1) * lambda_window)
    reader.read(matrix_shared, chrom, starts, ends)
    data = _to_numpy_array(matrix_shared, reader.number_of_bigwig_files+1)

    if binsize != 1:
        binned_length = int(np.ceil(lambda_window/binsize))
        vecbinned = np.empty(binned_length)

    search_range = np.logspace(search_start, search_end, num=num)
    hyperparameter_list = np.empty(reader.number_of_bigwig_files)
    for i in range(data.shape[0]-1):
        if verbose: print(f"Picking lambda for {reader.bigwig_file_list[i]}")
        if binsize == 1:
            obj = _find_lambda(data[i], search_range)
        else:
            _copy_binned(vecbinned, data[i], binsize, lambda_window)
            obj = _find_lambda(vecbinned, search_range)
        search_range_2 = np.linspace(max(0, search_range[np.argmin(obj)]-second_window), search_range[np.argmin(obj)]+second_window, num = num)
        if binsize == 1:
            obj = _find_lambda(data[i], search_range_2)
        else:
            _copy_binned(vecbinned, data[i], binsize, lambda_window)
            obj = _find_lambda(vecbinned, search_range_2)
        hyperparameter_list[i] = search_range_2[np.argmin(obj)]
        if verbose: print(f"Lambda: {hyperparameter_list[i]}")
    if path is not None:
        with open(path, "w") as file:
            for i in range(len(hyperparameter_list)):
                file.write(reader.bigwig_file_list[i] + ": " + str(hyperparameter_list[i]) + "\n")
    return hyperparameter_list

class GenomeReader:
    '''
    Reader for a genome. It takes a list of bw files. It reads a bed file of indexes for training regions and returns the signal values for those regions.
        bwpath: path to the directory where the bw files are restored.
        bedpath: path to the bed files that keeps training indexes.
        binsize: size of the bins
        chrsizes: dictionary of chromosome sizes.
    '''
    def __init__(self, bigwig_file_list):

        self.bigwig_file_list = bigwig_file_list
        self._number_of_bigwig_files = None 
        self._number_of_bigwig_files = len(self.bigwig_file_list)

    @property 
    def data_dim(self):
        return 3*(self._number_of_bigwig_files+1)

    @property
    def number_of_bigwig_files(self):
        return self._number_of_bigwig_files

    @number_of_bigwig_files.setter
    def number_of_bigwig_files(self, value):
        self._number_of_bigwig_files = value

    def read(self, mat_shared, chromosome, start, end, number_of_cores = 4):
        _readbw_parallel(mat_shared, self.bigwig_file_list, chromosome, start, end, number_of_cores)

class Segmentor:
    def __init__(self, reader, classifier, lambda_list, number_of_components, chromosome_size_dictionary, bin_size = 200, chunk_size = 10000000, number_of_sample = 2000, number_of_cores = 4):

        self.reader = reader

        self.classifier = classifier

        self.lambda_list = lambda_list

        self.number_of_components = number_of_components

        self.chromosome_size_dictionary = chromosome_size_dictionary

        self.bin_size = bin_size

        self.chunk_size = chunk_size
        self.M = chunk_size

        self.number_of_sample = number_of_sample

        self.number_of_epigenetic_tracks = self.reader.number_of_bigwig_files
        self.N = self.reader.number_of_bigwig_files
        self.number_of_cores = number_of_cores


    def fit(self, regions, verbose = False):
        number_of_iterations = len(regions)
        matrix_shared = mp.Array(ctypes.c_double, (self.N+1) * self.M)
        reduced_shared = mp.Array(ctypes.c_double, 3*(self.N+1) * self.number_of_sample * number_of_iterations)
        number_of_sample_found = 0

        if verbose: print("Training...")
        for i, (chromosome, start, end) in enumerate(regions):
            if verbose: print(f"Collecting sample from {chromosome}:{start}-{end}...")
            self.reader.read(matrix_shared, chromosome, start, end, self.number_of_cores)
            bps = _return_break_points_reduced_parallel(matrix_shared, chromosome, self.bin_size, self.lambda_list, self.chromosome_size_dictionary, self.N, self.M, end - start, self.number_of_cores)
            selected_ii = np.sort(np.random.choice(len(bps), self.number_of_sample, replace=False))
            number_of_sample_found += _return_reduced_data_parallel(matrix_shared, reduced_shared, bps, self.N, self.M, i * self.number_of_sample, selected_ii, self.number_of_cores)

        if verbose: print(f"Training the classifier...")
        training = _to_numpy_array(reduced_shared, 3*(self.N+1))[:,:number_of_sample_found].T
        _prepare_for_classifier(reduced_shared, 3*(self.N+1), number_of_sample_found)
        training = _to_numpy_array(reduced_shared, 3*(self.N+1))[:,:number_of_sample_found].T
        self.classifier.fit(training)
        if verbose: print(f"Training is done.")
        classifier_file = "classifier.pkl"
        if verbose: print(f"Saving the classifier into {classifier_file}.")
        with open(classifier_file, 'wb') as file:
            pickle.dump(self.classifier, file) 

    def predict(self, output_file, target = None, verbose = False):
        matrix_shared = mp.Array(ctypes.c_double, (self.N+1) * self.M)
        reduced_shared = mp.Array(ctypes.c_double, 3*(self.N+1) * self.M)
        label = np.empty(self.M)
        with open(output_file, "w") as file:
            file.write("chromosome\tstart\tend\tlabel\n")
        if verbose: print("Annotating genome")
        if target is None:
            for chromosome in sorted(self.chromosome_size_dictionary.keys()):
                start_indices, end_indices = _return_target_ii(self.chromosome_size_dictionary[chromosome], self.chunk_size)
                for k in range(len(start_indices)):
                    start, end = start_indices[k], end_indices[k]
                    if verbose: print(f"Annotating {chromosome}:{start}-{end}...")
                    self.reader.read(matrix_shared, chromosome, start, end, self.number_of_cores)
                    bps = _return_break_points_reduced_parallel(matrix_shared, chromosome, self.bin_size, self.lambda_list, self.chromosome_size_dictionary, self.N, self.M, end - start, self.number_of_cores)
                    number_of_sample_found = _return_reduced_data_parallel(matrix_shared, reduced_shared, bps, self.N, self.M, 0, np.arange(len(bps)), self.number_of_cores)
                    _prepare_for_classifier(reduced_shared, 3*(self.N+1), number_of_sample_found)
                    annotation = _to_numpy_array(reduced_shared, 3*(self.N+1))[:,:number_of_sample_found].T
                    label[:number_of_sample_found] = self.classifier.predict(annotation)
                    if verbose: print(f"Saving annotations...")
                    with open(output_file, "a") as file:
                        for i in range(number_of_sample_found-1):
                            file.write(f"{chromosome}\t{bps[i]+start}\t{bps[i+1]+start}\t{int(label[i])}\n")
                        file.write(f"{chromosome}\t{bps[len(bps)-1]+start}\t{end}\t{int(label[len(bps)-1])}\n")
        else:
            target = sorted(sorted(target, key = lambda x: x[1]), key = lambda x: x[0])
            for chromosome, start_target, end_target in target:
                start_indices, end_indices = _return_target_ii(end_target, self.chunk_size, start = start_target)
                for k in range(len(start_indices)): 
                    start, end = start_indices[k], end_indices[k]
                    if verbose: print(f"Annotating {chromosome}:{start}-{end}...")
                    self.reader.read(matrix_shared, chromosome, start, end, self.number_of_cores)
                    bps = _return_break_points_reduced_parallel(matrix_shared, chromosome, self.bin_size, self.lambda_list, self.chromosome_size_dictionary, self.N, self.M, end - start, self.number_of_cores)
                    number_of_sample_found = _return_reduced_data_parallel(matrix_shared, reduced_shared, bps, self.N, self.M, 0, np.arange(len(bps)), self.number_of_cores)
                    _prepare_for_classifier(reduced_shared, 3*(self.N+1), number_of_sample_found)
                    annotation = _to_numpy_array(reduced_shared, 3*(self.N+1))[:,:number_of_sample_found].T
                    label[:len(bps)] = self.classifier.predict(annotation)
                    if verbose: print(f"Saving annotations...")
                    with open(output_file, "a") as file:
                        for i in range(number_of_sample_found-1):
                            file.write(f"{chromosome}\t{bps[i]+start}\t{bps[i+1]+start}\t{int(label[i])}\n")
                        file.write(f"{chromosome}\t{bps[len(bps)-1]+start}\t{end}\t{int(label[len(bps)-1])}\n")
        if verbose: print(f"Done...")
