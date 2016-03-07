import numpy as np
import sys, petsc4py
from petsc4py import PETSc
petsc4py.init(sys.argv)

def solve(A, b, ksp=None, pc=None, comm=None):
    """
    Solve the system of equations Ax = b for x.

        ARGUMENTS
            A       : left hand side PETSc matrix object
            b       : right hand size PETSc vector object

        OPTIONAL ARGUMENTS
            pc      : Preconditioner
            ksp     : Krylov space method

        RETURNS
            x       : PETSc vector object
                      solution to SLE, x = A^{-1} b

    """

    KSP = PETSc.KSP()
    KSP.create(comm=comm)
    if type(pc) == str:
        KSP.getPC().setType(pc)
    elif pc != None:
        KSP.setPC(pc)

    if pc != None:
        KSP.setType(ksp)
    KSP.setFromOptions()

    KSP.setOperators(A.mat)
    x = A.mat.createVecRight()
    KSP.solve(b, x)
    KSP.destroy()

    return x.array


class Solver(PETSc.KSP):
    """
    Useful if you want to solve multiple SLEs using the same matrix.
    Preconditioning can vastly reduce solving time.
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

        # Construct solver
        PETSc.KSP.__init__(self)
        self.create(comm=comm)
        self.getPC().setType(pc)
        self.setType(ksp)
        self.setFromOptions()


    def solve(self, A, b):
        """
        Solve a system of equations Ax = b to find x
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
