# These are the analytical function from Bonamente20
# Taken from cstat.py in ./pg1116

exec(open('imports.py').read())

# Simulate a normal distribution
fig,ax=plt.subplots(figsize=(8,6))

#def funcECmin(x, a,b,c,d,e,beta,f,alpha):
#    return (a+b*x+c*(x-d)**2.0)*np.exp(-alpha*x)+e*np.exp(-beta*x)+f
#def funcVarC(x, a,b,c,d,e,f,g,h,alpha,beta,i):
#    return  (a+b*x**2+c*(x-d)**2.0)*np.exp(-alpha*x) +  (e+f*x+g*(x-h)**2.0)*np.exp(-beta*x) +i

#iimport numpy as np
#from matplotlib import pyplot as plt
#import math


# Parameter values as reported in Bonamente20
parNamesE=['a','b','c','d','e','beta','f','alpha']
parNamesVar=['a','b','c','d','e','f','g','h','alpha','beta','i']

# These are the best-fit values for E[Ci] on full and narrow range, 
# followed by Var[Ci] for full and narrow range
A=[0.065672, -0.56709, -2.4637, -3.1971]
B=[-6.9461,-2.7336,1.5109,1.5118]
C=[-8.0124,-2.3603,-1.5109,-1.5118]
D=[0.40165,0.52816,0.60509,0.79384]
E=[0.261037,0.33133,1.4761,1.9294]
F=[1.00512,1.0174,18.358,6.1740]
G=[0,0,0.87316e-3,22.360e-3]
H=[0,0,-0.08592,-7.2981]
I=[0,0,2.02343,2.08378]
alphaB=[5.5178,3.9375,0.62652,0.750315]
betaB=[0.34817,0.48446,7.8187,4.49654]

colors=['black','red','gray','orange']
descriptor=['E[$C_i$] (range $\mu$=0.01-100)','E[$C_i$] (range $\mu$=0.1-10)','Var($C_i$) (range $\mu$=0.01-100)','Var($C_i$) (range $\mu$=0.1-10)'] 
# ==============================================

# Plot the four functions
size=100
mu1=np.logspace(-2,2,num=size)
mu2=mu=np.logspace(-1,1,num=size)
yaxis=np.zeros((4,size))

for j in range(2):
    if j==1: 
        mu=mu2
    if j==0:
        mu=mu1
    for i in range(size):
            yaxis[j][i]=funcECmin(mu[i],A[j],B[j],C[j],D[j],E[j],betaB[j],F[j],alphaB[j])
    print(yaxis[j][i])
    plt.semilogx(mu,yaxis[j],linewidth=2,color=colors[j],label=descriptor[j])
for j in range(2,4,1):
    if j==2:
        mu=mu1
    if j==3:
        mu=mu2
    for i in range(size):
            yaxis[j][i]=funcVarC(mu[i],A[j],B[j],C[j],D[j],E[j],F[j],G[j],H[j],alphaB[j],betaB[j],I[j])
    print(yaxis[j][i])
    plt.semilogx(mu,yaxis[j],linewidth=2,color=colors[j],label=descriptor[j])

for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())
# Add horizontal and vertical lines
plt.hlines(1,mu1[0],mu1[size-1],linewidth=2,linestyle='--')
plt.hlines(2,mu1[0],mu1[size-1],linewidth=2,linestyle='--')
ax.set_xlabel('Poisson mean $\mu$')
ax.set_ylabel('Appr. Expectation and Variance')
plt.vlines(0.1,0.0,2.5,linestyle='--',color='black',linewidth=1)
plt.vlines(10,0.0,2.5,linestyle='--',color='black',linewidth=1)
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))

plt.grid(which='both')
plt.xlim(0.01,100)
plt.ylim(0,2.5)
plt.legend(loc=2,prop={'size': 12})
plt.savefig('CiMeanVariance.pdf')

# ============
#mean=0.1
#  problem 15.2
print('Problem 15.2')
mean=0.2 
j=1
Expect=funcECmin(mean,A[j],B[j],C[j],D[j],E[j],betaB[j],F[j],alphaB[j])
j=3
Variance=funcVarC(mean,A[j],B[j],C[j],D[j],E[j],F[j],G[j],H[j],alphaB[j],betaB[j],I[j])
print('for mu=%3.3f: E=%3.3f, Var=%3.3f'%(mean,Expect,Variance))
N=100
EC=Expect*N
VarC=N*Variance
print('C=%3.3f \pm %3.3f'%(EC,VarC**0.5))
# For N=100, Gaussian approx for C applies.
q=1.3
print('C crit for N=%d: %3.3f'%(N,EC+q*(VarC)**0.5))
# ====================
# problem 15.3
print('Problem 15.3')
mean=3.0 # 
j=1
Expect=funcECmin(mean,A[j],B[j],C[j],D[j],E[j],betaB[j],F[j],alphaB[j])
j=3
Variance=funcVarC(mean,A[j],B[j],C[j],D[j],E[j],F[j],G[j],H[j],alphaB[j],betaB[j],I[j])
print('for mu=%3.3f: E=%3.3f, Var=%3.3f'%(mean,Expect,Variance))
N=100
EC=Expect*N
VarC=N*Variance
print('C=%3.3f \pm %3.3f'%(EC,VarC**0.5))



