# TEST FILE FOR strucAnal.py
import os
from strucAnal import TOStruc

folder = './vfProcessed/'
files = os.listdir(folder)
output = open(folder+'output.txt', 'a')

for file in files:

    PATH = folder+file
    test1 = TOStruc(PATH)

    compliance = test1.compliance()
    ce = test1.strainEnergy()
    vf = test1.vf()

    output.write("%s \t %s \t %s \n" %(file,vf, compliance))
