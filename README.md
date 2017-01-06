# Photoz-tools

This is a set of useful python notebooks related to photometric surveys and redshift estimation from noisy flux measurements.

Content:
- *Photoz galaxy survey mock and N(z) inference.ipynb*: notebook to generate a photometric survey mock, with realistic fluxes, redshifts and underlying galaxy types. Also recovers the underlying distributions via the hierarchical model/sampling of Leistedt, Mortlock and Peiris (2016).
- *bayeshist.py*: MPI implementation of the hierarchical model and Gibbs sampler of Leistedt, Mortlock and Peiris (2016) for inferring histograms of underlying distributions with binned likelihoods.
- *filters* and *seds* contain copies of the CWW galaxy SED templates and the SDSS photometric filters.

Contributors:
- Boris Leistedt (NYU)
- Daniel Mortlock (Imperial College)
- Hiranya Peiris (UCL)
- *add your name here*

Related papers:
- *Hierarchical Bayesian inference of galaxy redshift distributions from photometric surveys* by Leistedt, Mortlock and Peiris. [arxiv:1602.05960](http://arxiv.org/abs/1602.05960).

This code is released under MIT License. **Please cite the relevant papers if you use this code (ask the contributors if you're not sure).** Feel free to contribute via pull requests!
