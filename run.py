# TEST FILE FOR strucAnal.py
import os
from strucAnal import TOStruc

folder = './combinedProcessed/'
# files = os.listdir(folder)
files = [file for file in os.listdir(folder) if file.endswith('.png')]
output = open(folder+'outputThres.txt', 'a')

for file in files:

    PATH = folder+file
    test1 = TOStruc(PATH,1)

    compliance = test1.compliance()
    ce = test1.strainEnergy()
    vf = test1.vf()

    output.write("%s \t %s \t %s \n" %(file,vf, compliance))

##### TESTS
# testPath = 'test1.png'
# test1 = TOStruc(testPath)
# test1Thres = TOStruc(testPath, 1)

# print(test1.compliance())
# print(test1Thres.compliance())
