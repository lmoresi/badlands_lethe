import numpy as np
import sys, petsc4py
from petsc4py import PETSc
petsc4py.init(sys.argv)

class Preconditioner(PETSc.PC):
    def __init__(self, A, pc, comm, external_package=None):
        PETSc.PC.__init__(self)
        self.create(comm=comm)
        self.setType(pc)
        if external_package != None:
            self.setFactorSolverPackage(external_package)

        self.setUseAmat(A)
