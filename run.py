# execute the functions for pre-, post-processing and structural analysis
import os
from strucAnal import TOStruc
import postProcess
import numpy as np
import time
import cv2

folder = './processed/'
# files = os.listdir(folder)
files = [file for file in os.listdir(folder) if file.endswith('.png')]
output = open(folder+'iterationsAR_MMA.txt', 'a')

# test = np.load(folder+'52_26_0.42_GAN.png.npy')

for file in files:

    PATH = folder+file
    itemsList = file.split('_')[:-1]
    fileNAME = file.split('.')[0:-1]
    vf = float(itemsList[2])
    nelxOri = int(itemsList[0])
    nelyOri = int(itemsList[1])

    xPhysInput = TOStruc(PATH,0)
    resxPhys = cv2.resize(xPhysInput.struc, dsize=(nelxOri, nelyOri), interpolation = cv2.INTER_NEAREST)
    nely = xPhysInput.nely
    nelx = xPhysInput.nelx

    penal = 3  # consistent with setting in data generation
    rmin = 2   # consistent with setting in data generation

    start_time = time.time()
    xPrint, loop = postProcess.main(resxPhys, nelxOri, nelyOri,vf,penal,rmin, 2, 1)
    duration = time.time()-start_time

    #### determine no. of unprintable elements
    # BinaryedxPhys = TOStruc(PATH,2)
    # xPrint, _ = BinaryedxPhys.AMfilter()
    # xPrintBinary = np.where(xPrint<0.1, 0, 1)

    # initial = np.sum(np.sum(BinaryedxPhys.struc))
    # printable = np.sum(np.sum(xPrintBinary))
    # nUnprintable = initial - printable 

    output.write("%s \t  %s s \t %s  \n" %(file, duration, loop))
    np.save(folder+file+'.npy', xPrint)

##### TESTS
# testPath = 'test1.png'
# test1 = TOStruc(testPath)
# test1Thres = TOStruc(testPath, 1)

# print(test1.compliance())
# print(test1Thres.compliance())
