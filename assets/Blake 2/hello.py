from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()
#print nproc
#print myrank
if ( myrank == 0 ):
    for i in range(1 , nproc ):
        a = np.arange( 10 , dtype ='i')
        comm.Send([a , 10 , MPI.INT], dest =i , tag =7 )
        print 'la'
else :
    a = np.zeros(10 , dtype ='i')
    comm.Recv([a , 10 , MPI.INT], source =0 , tag =7 )
    print a**10
    print ("I'm process {0} and received : {1}\n".format(myrank , a ) )
    print 'lalala'
print ("Hello , World !")
