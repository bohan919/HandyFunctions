# TEST FILE FOR strucAnal.py

from strucAnal import TOStruc

PATH = 'test2.png'
test1 = TOStruc(PATH)

compliance = test1.compliance()
ce = test1.strainEnergy()
vf = test1.vf()

print(compliance)
print(vf)