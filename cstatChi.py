exec(open('imports.py').read())

# sample data for problems for chapters 15 and 16
N=9
#problem 15.4
print('Problem 15.4')
scale=2
y=scale*np.asarray([2,4,6,2,4,6,2,4,6]) # sample dataset
print(y,len(y))
ymodel=4*scale**np.ones(N) # fixed model - no fitting

# statistics
Cstat=cstat(y,ymodel)
chi=chisquared(y,y**0.5,ymodel)

# critical value of chi2
p=0.9
chiCrit=chi2.ppf(p,N) 
# critical value of cstat
mean=4*scale #  prob15.4
j=1
Expect=funcECmin(mean,A[j],B[j],C[j],D[j],E[j],betaB[j],F[j],alphaB[j])
j=3
Variance=funcVarC(mean,A[j],B[j],C[j],D[j],E[j],F[j],G[j],H[j],alphaB[j],betaB[j],I[j])
print('for mu=%3.3f: E=%3.3f, Var=%3.3f'%(mean,Expect,Variance))
EC=Expect*N
VarC=N*Variance
print('Expected C=%3.3f \pm %3.3f'%(EC,VarC**0.5))
# For N=100, Gaussian approx for C applies.
q=1.3
CCrit=EC+q*(VarC)**0.5
print('C crit for N=%d: %3.3f'%(N,CCrit))


print('chi2=%3.3f (%3.2f), cstat=%3.3f (%3.2f)'%(chi,chiCrit,Cstat,CCrit))

######################
#problem 16.8 
print('Problem 16.8')
scale=2
y=scale*np.asarray([2,4,6,2,4,6,2,4,6]) # sample dataset

xaxis=np.ones(len(y)) # this is irrelevant since it's constant model
popt, pcov = curve_fit(constant,xaxis , y, sigma=y**0.5,absolute_sigma=True)
print('chi2 curve_fit',popt,pcov)
print('a=%3.4f +- %3.4f '%(popt[0],pcov[0][0]**0.5))

######## now cstat
ahat=np.mean(y) # this is the known solution
ymod=np.ones(len(y))*ahat
cmin=cstat(y,ymod)
print(' cmin=%3.2f for ahat=%3.2f'%(cmin,ahat))
a=np.linspace(6,10,40)
c=np.zeros(40)
for i in range(len(a)):
    c[i]=cstat(y,a[i]*np.ones(len(y)))
    #print('a=%3.2f cstat=%3.2f'%(a[i],c[i]))

