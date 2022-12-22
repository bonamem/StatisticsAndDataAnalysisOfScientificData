exec(open('imports.py').read())

A=[10,11,13,11,10]

B=[7,8,1,11,10,8]

meanA=np.mean(A)
varA=np.var(A,ddof=1)
stdevA=np.std(A,ddof=1)
meanB=np.mean(B)
varB=np.var(B,ddof=1)
stdevB=np.std(B,ddof=1)

print('A: %3.3f \pm %3.3f (%3.2f) , B: %3.3f\pm %3.3f (%3.2f)'%
        (meanA,stdevA,varA,meanB,stdevB,varB))

z = (meanA-meanB)/(varA/len(A) + varB/len(B))**0.5
print('Geweke z-score: %3.3f'%z)
