import numpy as np
import time

class Conduction2D:
	"""
	Builds a 2D object and adds conductivity and heat source fields to solve conduction
	over a regular-spaced mesh.
	"""
	def __init__(points_x, points_y, res_x, res_y, conductivity, heat_sources, verbose=False):
		""" Initialise points and fields """
		self.points_x = points_x
		self.points_y = points_y
		self.dx = res_x
		self.dy = res_y
		self.conductivity = conductivity
		self.k0 = conductivity
		self.heat_sources = heat_sources
		self.verbose = verbose

		return

	def boundary_conditions(self, topBC, bottomBC, leftBC=0, rightBC=0):
		""" Assign numeric value to each wall """
		self.topBC = topBC
		self.bottomBC = bottomBC
		self.leftBC = leftBC
		self.rightBC = rightBC

		return

	def boundary_types(self, topFlux, bottomFlux, leftFlux=True, rightFlux=True):
		""" Determine the type of BC on each wall (Dirichlet / Neumann) """
		self.topFlux = topFlux
		self.bottomFlux = bottomFlux
		self.leftFlux = leftFlux
		self.rightFlux = rightFlux

		return

	def nonLinearConductivity(self, PHI, a=0.5):
		""" Controls the relation of conductivity with another field """
		conductivity = np.array(self.k0) * np.array(298/PHI)**a

		return conductivity

	def solve(self, non_linear=True):
		""" Solve thermal conduction implicitely using finite differences """
		t = time.clock()
		from scipy.sparse import coo_matrix, csr_matrix
		from scipy.sparse.linalg import spsolve

		def writeMatrix(Ir, Jr, Vr):
			""" Writes nodes to a sparse tridiagonal matrix """
			global I, J, V, nPos
			I[nPos] = Ir
			J[nPos] = Jr
			V[nPos] = Vr
			nPos += 1

		nx, ny = len(self.points_x), len(self.points_y)
		Nnode = nx*ny

		global I, J, V, nPos
		I = np.zeros(Nnode*5-3, dtype=int)
		J = np.zeros(Nnode*5-3, dtype=int)
		V = np.zeros(Nnode*5-3)
		nPos = 0
		matC = np.zeros(Nnode)

		adx = 1.0 / (2*self.dx**2)
		ady = 1.0 / (2*self.dy**2)

		residual = 10.0
		nIter = 0
		while residual > 1e-6:
			nIter += 1
			nPos = 0
			index = np.range(Nnode, dtype=int)
			remove = []
			temperature_last = self.temperature

			## BCs ##
			# Top and Bottom BCs
			for pos in range(0, nx):
				jBC = pos
				if self.topFlux:
					writeMatrix(jBC, jBC, (self.conductivity[-1,pos]+self.conductivity[-2,pos])*-ady)
					writeMatrix(jBC, jBC+nx, (self.conductivity[-1,pos]+self.conductivity[-2,pos])*ady)
					matC[jBC] = (self.heatProduction[-1,pos]+self.heatProduction[-2,pos]) / 2.0 - self.topBC/self.dy
				else:
					writeMatrix(jBC, jBC, 1.0)
					matC[jBC] = -300. #Top BC
				remove.append(jBC)

				jBC += Nnode-nx
				if self.bottomFlux:
					writeMatrix(jBC, jBC, (self.conductivity[-1,pos]+self.conductivity[-2,pos])*-ady)
					writeMatrix(jBC, jBC-nx, (self.conductivity[-1,pos]+self.conductivity[-2,pos])*ady)
					matC[jBC] = (self.heatProduction[-1,pos]+self.heatProduction[-2,pos]) / 2.0 - self.bottomBC/self.dy
				else:
					writeMatrix(jBC, jBC, 1.0)
					matC[jBC] = self.bottomBC *-1
				remove.append(jBC)

			# Left and Right BCs
			for pos in range(1, ny-1):
				iBC = pos*nx
				if self.leftFlux:
					writeMatrix(iBC, iBC, (self.conductivity[pos,0]+self.conductivity[pos,1])*adx)
					writeMatrix(iBC, iBC+1, (self.conductivity[pos,0]+self.conductivity[pos,1])*-adx)
					matC[iBC] = (self.heatProduction[pos,0]+self.heatProduction[pos,1]) / 2.0 - self.leftBC/self.dx
				else:
					writeMatrix(iBC, iBC, 1.0)
					matC[iBC] = self.leftBC *-1
				remove.append(iBC)

				iBC += nx-1
				if self.rightFlux:
					writeMatrix(iBC, iBC, (self.conductivity[pos,-1]+self.conductivity[pos,-2])*adx)
					writeMatrix(iBC, iBC-1, (self.conductivity[pos,-1]+self.conductivity[pos,-2])*-adx)
					matC[iBC] = (self.heatProduction[pos,-1]+self.heatProduction[pos,-2]) / 2.0 - self.rightBC/self.dx
				else:
					writeMatrix(iBC, iBC, 1.0)
					matC[iBC] = self.rightBC *-1
				remove.append(iBC)

			index = np.delete(index, remove)

			pos = 0
			for r in range(1, ny-1):
				for c in range(1, nx-1):
					Di = [self.conductivity[r,c-1], self.conductivity[r,c], self.conductivity[r,c+1]]
					Dj = [self.conductivity[r-1,c], self.conductivity[r,c], self.conductivity[r+1,c]]

					writeMatrix(index[pos], index[pos]-nx, (Dj[0]+Dj[1])*ady)
					writeMatrix(index[pos], index[pos]-1, (Di[0]+Di[1])*adx)
					writeMatrix(index[pos], index[pos], (Di[0]+2*Di[1]+Di[2])*-adx + (Dj[0]+2*Dj[1]+Dj[2])*-ady)
					writeMatrix(index[pos], index[pos]+1, (Di[2]+Di[1])*adx)
					writeMatrix(index[pos], index[pos]+nx, (Dj[2]+Dj[1])*ady)

					matC[index[pos]] = self.heatProduction[r,c]
					pos += 1

			A = coo_matrix((V, (I, J))).tocsr()
			b = coo_matrix((np.array(matC)*-1), shape=(Nnode,1)).T
			temperature = spsolve(A, b.tocsr())

			self.temperature = np.reshape(temperature, (ny, nx))
			if non_linear:
				self.conductivity = nonLinearConductivity(temperature)

			residual = np.absolute(np.array(temperature)-np.array(temperature_last)).max()

			if self.verbose:
				print "[Iteration %i] residual - %.04f, walltime - %0.2f" % (nIter, residual, time.clock()-t)

		if non_linear:
			return self.temperature, self.conductivity
		else:
			return self.temperature

