# fastMCS

This toolbox implements the fast updating implementation of the Model Confidence Set (MCS) presented in "A fast algorithm for finding the confidence set of large model collections". This improves on the original elimination implementation of Hansen *et. al.* (2011).

## Requirements and Installation

The toolbox is implemented entirely using `numpy 1.23.5`, which is the only external dependency. The requirements file specifies allow any compatible version to be used. The toolbox has **not** been tested with `numpy 2.0`, which may not be compatible. Additional (standard) packages required are: `os, sys, time, pickle, zlib`.

At the moment the `fastMCS` toolbox is still in development, and does not yet have a distributable package (this is on the to-do list!). The functionality is all contained in the `fastMCS.py` module, and can be obtained simply by installing `numpy` and placing a copy of the file in the relevant directory.

## Functionality

This section details the basic functionality of the toolbox. The github repo also provides a demonstration example, both as a python script (`fastMCS_demo.py`) and as a jupyter notebook (`fastMCS_demo.ipynb`). The `mcs` class contains all the functionality required  and is imported as follows.

```python
from fastMCS import mcs
```

### Initialisation

An empty MCS analysis task is initialised as follows:

```python
initSeed = 10
mcsEst = mcs(seed = initSeed, verbose = False)
```

The two options that can be picked at initialisation are
 - The seed used by the random number generator, which controls the bootstrap replications used in the analysis. This can be provided manually to ensure replicability of results. If the seed is not provided, the class uses the current time (provided by `time.time()`) as the default.
 - The verbosity of the output. By default, this is set to `True` and the toolbox provides talkbacks for the outcome of each command. This can be manually turned off by setting the option to `False` (as is the case in the example here), to avoid producing console outputs. This can be useful if a large number of MCS analyses need to be run as part of a wider automated process, and one wishes to avoid crowding out the console.

### Adding losses and running an MCS analysis

To run an MCS analysis on an $N \times M$ loss matrix $L$, where $N$ is the number of observations and $M$ is the number of models, using a previously initialised MCS object `mcsEst`, the syntax is as follows:

```python
# Adding a loss matrix 'losses'
mcsEst.addLosses(losses)

B = 2000    # set number of bootstrap replications
b = 5       # set block size
mcsEst.run(B, b, bootstrap = 'stationary', algorithm = '2-pass')
```

The first two parameters control the bootstrap settings:
- `B` is the number of bootstrap replications, this is 1000 by default.
- `b` is the average size of a bootstrap block, this is 10 by default.

Two options are available for the choice of bootstrap:
- `'stationary'`: uses the Politis & Romano (1994) stationary bootstrap. This is the default option.
- `'block'`: uses a fixed-width block bootstrap.

Finally, the `fastMCS` module provides three options for the MCS analysis itself:
- `'elimination'`: uses the **Note:** This is provided for comparability with the fast updating algorithm, users should not be used for large scale ($M>500$) applications due to the high memory requirement involved.
- `'2-pass'`: uses the 2-pass fast updating algorithm, with the first pass producing the model rankings and the second producing the p-values. This provides equivalent output to `'elimination'` and is the default option.
- `'1-pass'`: uses the 1-pass fast updating algorithm, where both the model rankings and p-values are updated in a single pass, using a heuristic to obtain the latter. The model rankings will correspond to `'2-pass'` but the p-values might differ slightly. This option is imposed when updating an existing `mcs` object with additional losses.

### Getting the MCS

The MCS analysis carried out using `mcsEst.run()` calculates the model rankings (and thus the elimination sequence) and the corresponding p-values for the entire collection of $M$ models, but does not directly return the actual MCS, as this depends on the significant level $\alpha$ that the user wishes to pick. The partition of the original collection into the MCS and the set of eliminated models can be obtained using:

```python
# Get 90% MCS  
inclMods, exclMods = mcsEst.getMCS(alpha = 0.1)
```

### Saving and loading

Once the analysis is complete, the state of the `mcs` object can be save for later retrieval.

```python
# Saving to 'savePath/filename.pkl'
mcsEst.save(savePath, filename)
```

Similarly, a saved analysis can be recovered by initialising an empty `mcs` object and using the `load` method:

```python
# Initialise an mcs object
mcsEstRecovered = mcs()

# load from 'savePath/filename.pkl'
mcsEstRecovered.load(savePath, filename)
```

### Performance statistics

The `mcsEst` object records the performance of each analysis in its  `mcsEst.stats` attribute. Because a given object can be updated with new losses an indefinite amount of times, this attribute consists of a list of python `dicts`, each one corresponding to a run. For instance `mcsEst.stats[0]` will contain the statistic for the initial run, `mcsEst.stats[1]` those of the first update, `mcsEst.stats[2]` those of the second update, etc.

Each run `dict` contains the following fields:
- `'run'`: The ID of the run
- `'models processed'`: number of models processed in run
- `'method'`: algorithm used to process the MCS
- `'time'`: time taken in seconds
- `'memory use'`: total memory used in bytes
- `'memory alloc'`: detailed allocation of memory over variables

## Reference:

- Hansen, P.R., Lunde, A. and Nason, J.M., 2011. The model confidence set. *Econometrica*, 79(2), pp.453-497.
- Politis, D.N. and Romano, J.P., 1994. The stationary bootstrap. *Journal of the American Statistical association*, 89(428), pp.1303-1313.
