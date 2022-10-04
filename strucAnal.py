'''
THIS SCRIPT PERFORMS BASIC STRUCTURAL PERFORMANCE ANALYSIS ON A GIVEN
STRUCTURE 
'''

import numpy as np
import numpy.matlib
import cv2
from scipy.sparse import csr_matrix
from pypardiso import spsolve

# img_path = '/YOUR/PATH/IMAGE.png'
# img = cv2.imread(img_path, 0) # read image as grayscale. Set second parameter to 1 if rgb is required

class TOStruc:
    """
    This is a parent class which is meant to feed into both TPMS and Truss 
    generation classes (below). It will flesh out the main structure of the 
    lattice generation classes, with the methods to be completed with speci-
    fics in the child classes. 
    
    :PATH : where the .png files are stored
    :type : string

    :Loading: Loading type
                1 - Point Load Downwards
                2 - 
    :type :  int

    :BC : Boundary Condition
                1 - Half-MBB
                2 - 
    
    """
    def __init__(self, PATH, Loading=1, BC=1):

        input = cv2.imread(PATH, 0)
        self.struc = 1-input/255
        
        if Loading == 1:
            self.Loading = Loading
            print('Loading: point load at the end')
        if BC == 1:
            self.BC = BC
            print('BC: Half-MBB')
        

    def compliance(self):
        _, c = self.FEA(self.Loading, self.BC)
        return c

    def strainEnergy(self):
        ce, _ = self.FEA(self.Loading, self.BC)
        return ce

    def vf(self):
        xPhys = self.struc
        nely, nelx = xPhys.shape
        vf = np.sum(np.sum(xPhys))/(nely*nelx)
        return vf
        

    def FEA(self, Loading, BC):
        xPhys = self.struc
        nely, nelx = xPhys.shape
        # MATERIAL PROPERTIES
        E0 = 1
        Emin = 1e-9
        nu = 0.3

        # Approx. Settings
        penal = 3
        
        # PREPARE FINITE ELEMENT ANALYSIS
        A11 = np.array([[12, 3, -6, -3],[3, 12, 3, 0],[-6, 3, 12, -3],[-3, 0, -3, 12]])
        A12 = np.array([[-6, -3, 0, 3],[-3, -6, -3, -6],[0, -3, -6, 3],[3, -6, 3, -6]])
        B11 = np.array([[-4, 3, -2, 9],[3, -4, -9, 4],[-2, -9, -4, -3],[9, 4, -3, -4]])
        B12 = np.array([[2, -3, 4, -9],[-3, 2, 9, -2],[4, 9, 2, 3],[-9, -2, 3, 2]])
        Atop = np.concatenate((A11, A12),axis = 1) 
        Abottom = np.concatenate((A12.T, A11), axis = 1)
        A = np.concatenate((Atop,Abottom), axis = 0)
        Btop = np.concatenate((B11, B12), axis = 1)
        Bbottom = np.concatenate((B12.T, B11), axis = 1)
        B = np.concatenate((Btop, Bbottom), axis = 0)

        KE = 1/(1-nu**2)/24 *(A + nu*B)
        nodenrs = np.reshape(np.arange(1,((nelx+1)*(nely+1)+1)), (1+nelx,1+nely))
        nodenrs = nodenrs.T
        edofVec = np.ravel(nodenrs[0:nely,0:nelx], order='F') *2 + 1
        edofVec = edofVec.reshape((nelx*nely,1))
        edofMat = np.matlib.repmat(edofVec,1,8) + np.matlib.repmat(np.concatenate(([0, 1], 2*nely+np.array([2,3,0,1]), [-2, -1])),nelx*nely,1)

        iK = np.reshape(np.kron(edofMat, np.ones((8,1))).T, (64*nelx*nely,1),order='F')
        jK = np.reshape(np.kron(edofMat, np.ones((1,8))).T, (64*nelx*nely,1),order='F')

        if Loading == 1:
            # DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
            F = np.zeros((2*(nely+1)*(nelx+1),1))
            F[1,0] = -1

        if BC == 1:
            U = np.zeros((2*(nely+1)*(nelx+1),1))
            fixeddofs = np.union1d(np.arange(1,2*(nely+1),2),2*(nelx+1)*(nely+1))
            alldofs = np.arange(1,2*(nely+1)*(nelx+1)+1)
            freedofs = np.setdiff1d(alldofs, fixeddofs)


        # FE ANALYSIS
        sK = np.reshape(KE.ravel(order='F')[np.newaxis].T @ (Emin+xPhys.ravel(order = 'F')[np.newaxis]**penal*(E0-Emin)),(64*nelx*nely,1),order='F')
        K = csr_matrix( (np.squeeze(sK), (np.squeeze(iK.astype(int))-1,np.squeeze(jK.astype(int))-1)))
        K = (K + K.T) / 2
        U[freedofs-1,0]=spsolve(K[freedofs-1,:][:,freedofs-1],F[freedofs-1,0])   #### BOTTLENECK

        #OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
        ce =  np.reshape((np.sum( U[edofMat-1,0]@KE*U[edofMat-1,0] , axis = 1)),(nely, nelx),order='F') # strain energy distribution
        c = np.sum(np.sum( (Emin+xPhys**penal*(E0-Emin))*ce )) # compliance

        return ce, c
