# TEST FILE FOR strucAnal.py
import os
from strucAnal import TOStruc
import numpy as np

folder = './ARProcessed/'
# files = os.listdir(folder)
files = [file for file in os.listdir(folder) if file.endswith('.png')]
output = open(folder+'unprintableElements.txt', 'a')

for file in files:

    PATH = folder+file
    BinaryedxPhys = TOStruc(PATH,2)
    nely = BinaryedxPhys.nely
    nelx = BinaryedxPhys.nelx

    xPrint, _ = BinaryedxPhys.AMfilter()
    xPrintBinary = np.where(xPrint<0.1, 0, 1)

    initial = np.sum(np.sum(BinaryedxPhys.struc))
    printable = np.sum(np.sum(xPrintBinary))

    nUnprintable = initial - printable 

    # compliance = test1.compliance()
    # ce = test1.strainEnergy()
    # vf = test1.vf()

    output.write("%s \t nelx - %s nely - %s \t nUnprintable: %s \n" %(file, nelx, nely, nUnprintable))

##### TESTS
# testPath = 'test1.png'
# test1 = TOStruc(testPath)
# test1Thres = TOStruc(testPath, 1)

# print(test1.compliance())
# print(test1Thres.compliance())
