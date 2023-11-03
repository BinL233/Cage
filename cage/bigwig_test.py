import pyBigWig

bw = pyBigWig.open('../../Data/bpNetlite_ex/ENCSR000AKO_plus.bigWig')
values = bw.values('chr2',  203781867, 203783123)
print(values)