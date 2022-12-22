exec(open('imports.py').read())

# use the 10 measurements for chi2 and F tests
x=np.array([10,12,15,11,13,16,12,10,18,13])

print('Problem 9.5')
x1=x[0:2]
x2=x[2:10]
m1=np.mean(x1)
m2=np.mean(x2)
s21=np.var(x1,ddof=1)
s22=np.var(x2,ddof=1)
F=s22/s21
N1=2
N2=8
print('means of 0-1 and 2-10 segments: %3.3f, %3.3f, variances: %3.3f, %3.3f, S2: %3.3f %3.3f, F: %3.3f'
        %(m1,m2,s21,s22,s21*(N1-1),s22*(N2-1),F))
fCrit=f.ppf(0.95,N1-1,N2-1)
fPValue=f.sf(F,N1-1,N2-1)
print('F: %3.3f (crit value 0.90: %3.3f)'%(F,fCrit))
print('F: p-value %3.3f'%fPValue)

quit()
####################################################
# This is used for the examples in Chapter 9
x1=x[0:5]
x2=x[5:10]
mu=13
yMod=mu*np.ones(5)
sigma=2**0.5
yErr=sigma*np.ones(5)
chi21=chisquared(x1,yErr,yMod)
chi22=chisquared(x2,yErr,yMod)
chi2Crit=chi2.ppf(0.95,5)
N=5
print(x1,x2)
print('chi2: %3.3f, %3.3f (crit value 0.95: %3.3f'%(chi21,chi22,chi2Crit))
F=chi22/chi21
fCrit=f.ppf(0.95,N,N)
fPValue=f.sf(F,N,N)
print('F: %3.3f (crit value 0.95: %3.3f)'%(F,fCrit))
print('F: p-value %3.3f'%fPValue)
# now with sample means
m1=np.mean(x1)
m2=np.mean(x2)
s21=np.var(x1,ddof=1)
s22=np.var(x2,ddof=1)
F=s22/s21
# ==== z-score of first five measurements
N=5
z1=(m1-mu)/(sigma/N**0.5)
p1=norm.sf(abs(z1),loc=0,scale=1)*2
print('zscore of 5 measurements: %3.3f, pvalue: %3.3f'%(z1,p1))
# ==== t stat for the 5 measurements
tStat1=(m1-mu)/(s21/N)**0.5
tPvalue=t.sf(abs(tStat1),N-1) # one-sided prob.
tPvalue=tPvalue*2
print('t statistic: %3.3f, 2-sided p-value: %3.3f'%(tStat1,tPvalue))
# ==== t stat for the 10 measurements
s2=(s21*(N-1)+s22*(N-1))/(N+N-2)
tStatComp=(m1-m2)/s2**0.5/(1/N+1/N)**0.5
print('s2= %3.3f, t stat for comparison of two sample means: %3.3f'%(s2,tStatComp))
tPValueComp=2*t.sf(abs(tStatComp),2*(N-1))
print('p-value for the comparison: %3.3f'%tPValueComp)

