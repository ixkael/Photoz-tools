#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
import sys
import time

# 	Input 1 (required) : file name for inputs (likelihoods)
# 	Input 2 (required) : number of Gibbs samples 
# 	Input 3 (optional) : file name for input initialisation

# Parse inputs
assert len(sys.argv) <= 3, "The number of inputs should be <= 3"
infilename = sys.argv[1]
nsamples = int(sys.argv[2])
if len(sys.argv) > 3:
    fname_ini = sys.argv[3]
burnin_fraction = 0.3

comm = MPI.COMM_WORLD
MPI_size = comm.Get_size()
MPI_rank = comm.Get_rank()

if MPI_rank == 0:
    print('Input parameters:', sys.argv)
    print('Running on %d cores' % MPI_size)

def dirichlet(rsize, alphas):
    gammabs = np.array([np.random.gamma(alpha+1, size=rsize) for alpha in alphas])
    fbs = gammabs / gammabs.sum(axis=0)
    return fbs.T

pdfints_npfile = infilename+'_'+str(MPI_rank+1)+'.npy'
pdfints = np.load(pdfints_npfile)
sh = pdfints.shape
nobj = sh[0]
print('Read file', pdfints_npfile, 'and found', nobj, 'objects')
nbins = np.prod(sh[1:])
pdfints = pdfints.reshape((nobj, nbins))


if MPI_rank == 0:
    if len(sys.argv) > 3:
        print('Initialised sampler with file', fname_ini)
        hbs = np.load(fname_ini).reshape((nbins,))
        hbs /= hbs.sum()
    else:# random initialisation
        nbs = np.random.rand(nbins)
        hbs = dirichlet(1, nbs)
else:
    hbs = None

comm.Barrier()

hbs = comm.bcast(hbs, root=0)
if MPI_rank == 0:
    print('Broadcasted hbs')

if MPI_rank == 0:
    fbs = np.zeros( (nsamples, nbins) )
    tstart = time.time()

comm.Barrier()
ibins = np.repeat(np.arange(1, nbins), nobj).reshape((nbins-1, nobj)).T.ravel()
for kk in range(1, nsamples):

    prods = (pdfints * hbs)
    cumsumweights = np.add.accumulate(prods, axis=1).T #cumsumweights = prods.cumsum(axis=1).T 
    cumsumweights /= cumsumweights[-1,:]
    pos = np.random.uniform(0.0, 1.0, size=nobj)
    cond = np.logical_and(pos > cumsumweights[:-1,:], pos <= cumsumweights[1:,:])
    res = np.zeros(nobj, dtype=int)
    res[pos <= cumsumweights[0,:]] = 0
    locs = np.any(cond, axis=0)
    res[locs] = ibins[cond.T.ravel()]
    ind_inrange = np.logical_and(res > 0, res < nbins)
    nbs = np.bincount(res[ind_inrange], minlength=nbins)

    nbs_all = np.zeros_like(nbs)
    comm.Allreduce(nbs, nbs_all, op=MPI.SUM)

    if MPI_rank == 0:

        if kk % 10 == 0:
            #print kk
            tend = time.time()
            fname = infilename + '_post.npy'
            ss = int(burnin_fraction*kk)
            sh2 = tuple([kk-ss]+list(sh[1:]))
            print('Saving', kk-ss, 'samples to', fname, '(%.2f' % (float(tend-tstart)/kk), 'sec per sample)')
            np.save(fname, fbs[ss:kk, :].reshape(sh2))

        hbs = dirichlet(1, nbs_all) #### PLUS ONE HERE OR NOT?? ???? ??? 

    hbs = comm.bcast(hbs, root=0)

    if MPI_rank == 0:
        fbs[kk,:] = hbs
