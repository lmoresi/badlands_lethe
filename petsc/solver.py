import numpy as np
import sys, petsc4py
from petsc4py import PETSc
petsc4py.init(sys.argv)

def solve(A, b, ksp=None, pc=None, comm=None, **kwargs):
    """
    Solve the system of equations Ax = b for x.

        ARGUMENTS
            A       : left hand side PETSc matrix object
            b       : right hand size PETSc vector object

        OPTIONAL ARGUMENTS
            pc      : Preconditioner (str or Preconditioner object)
            ksp     : Krylov space method (str or KSP object)
                      -- if None uses PETSc default: 'gmres'
            kwargs  : Optional keyword arguments to be passed to KSP
                      e.g. rtol=10e-5, atol=10e-50, divtol=10e5, mat_it=10e4

        RETURNS
            x       : PETSc vector object
                      solution to SLE, x = A^{-1} b

    """

    KSP = PETSc.KSP()
    KSP.create(comm=comm)
    if isinstance(pc, str):
        KSP.getPC().setType(pc)
    elif pc != None:
        KSP.setPC(pc)

    if ksp != None:
        KSP.setType(ksp)

    for key, value in kwargs.items():
        setattr(KSP, key, value)

    KSP.setFromOptions()

    KSP.setOperators(A.mat)
    x = A.mat.createVecRight()
    KSP.solve(b, x)
    KSP.destroy()

    return x.array


class Solver(PETSc.KSP):
    """
    Useful for solving successive linear systems using the same matrix.
    Use the `solve` function if preconditioning is not required, there is
    no performance advantage using this object without preconditioning.

    Preconditioning the left matrix can vastly reduce solving time.
    Common preconditioners are:

        PARALLEL PC TYPES
            Jacobi        : "jacobi"
            Multigrid     : "mg"

        SEQUENTIAL PC TYPES
            SOR           : "sor"
            Cholesky      : "icc"

    For a more complete list go to
    <http://www.mcs.anl.gov/petsc/documentation/linearsolvertable.html>
    """

    def __init__(self, ksp='gmres', pc=None, comm=None):
        """
        Preconditioning the left matrix is mandatory.

            ARGUMENTS
                pc        : Preconditioner (str or Preconditioner object)
                ksp       : Krylov space method (str or KSP object)
                comm      : MPI communicator object
                            -- use mpi4py.MPI.COMM_WORLD
        """

        # Construct solver
        super(Solver, self).__init__()
        self.create(comm=comm)
        self.getPC().setType(pc)
        self.setType(ksp)
        self.setFromOptions()
        # Use previous solution vector to help solve
        self.setInitialGuessNonzero(True)


    def solve(self, A, b):
        """
        Solve a system of equations Ax = b for x
        A and b must be PETSc objects.

            ARGUMENTS
                A  : PETSc sparse matrix shape (m,m)
                b  : PETSc vector (m,)
            RETURNS
                x  : NumPy array (m,)
        """

        self.pc.setUseAmat(A.mat)
        self.ksp.setOperators(A.mat)

        if len(B.shape) == 1:
            # Create resultant vector
            x = np.zeros(b.shape[0])
            self.ksp.solve(b, x)

            return x.array

        else:
            raise TypeError('Cannot determine shape of B matrix/vector')
