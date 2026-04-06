#!/usr/bin/env python
"""
.. module:: MontePython
   :synopsis: Main module
.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>

Monte Python, a Monte Carlo Markov Chain code (with Class!)
"""
import sys
import warnings

# B.H. I added:
import matplotlib
matplotlib.use('Agg')
# B.H. I added think it actually works
import os
execfile('/usr/share/Modules/init/python.py')
module('unload','mpi/3.2')
module('unload','mpi/intel')
module('load','mpi/3.1.1')
module('list')
from mpi4py import MPI
quit()
#os.environ["OMP_NUM_THREADS"] = "4"

#from run import run


# -----------------MAIN-CALL---------------------------------------------
if __name__ == '__main__':
    # MPI is tested for, and if a different than one number of cores is found,
    # it runs mpi_run instead of a simple run.
    MPI_ASKED = False
    try:
        from mpi4py import MPI
        NPROCS = MPI.COMM_WORLD.Get_size()
        if NPROCS > 1:
            print 1
            MPI_ASKED = True
    # If the ImportError is raised, it only means that the Python wrapper for
    # MPI is not installed - the code should simply proceed with a non-parallel
    # execution.
    except ImportError:
        print 2
        pass

    if MPI_ASKED:
        # This import has to be there in case MPI is not installed
        from run import mpi_run
        sys.exit(mpi_run())
        print 3
    else:
        sys.exit(run())
        print 4
