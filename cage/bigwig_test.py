import pyBigWig

bw = pyBigWig.open('../../Data/bpNetlite_ex/hg38.fa')
values = bw.values('chr2',  203781867, 203783123)
print(values)