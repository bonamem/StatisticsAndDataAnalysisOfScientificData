exec(open('imports.py').read())
# Numerical calculation of the critical values for Ci
# In the low--count and low--N limit where no approximation is available

def CContribwTwo(Ni,mu):
    if (Ni==0):
        return 2*mu
    if (Ni>0):
        return 2*(mu - Ni +Ni*np.log(Ni/mu)) # factor of 2 included here

# =========================================
# Simple calculations in the case of large N, where C~Normal
N=100
chiCrit=chi2.ppf(0.9,N)
mu=N
sigma=(2*N)**0.5
NormCrit=norm.ppf(0.9,N,sigma)
print('Critical value of chi2(100): %3.3f, N(100,200):%3.3f'%(chiCrit,NormCrit))

# -------------
yi=2
mui=0.3
CTest=CContribwTwo(yi,mui)
print('Ci=%3.3f for yi=%d and mui=%3.3f'%(CTest,yi,mui))

# =========================================
# Under the null hypothesis that Mu is the parent mean,
# Draw random Poisson samples from Ni~Poiss(mu) and calculate
# Critical values at various confidence levels

# This should have been done in the Bonamente20 paper, but wasn't

nProb=5
p = [0.683,0.90,0.95,0.99,0.999]

# Choose whether to have many values or just one
#size=100
#mu=np.logspace(-2,2,num=size)
size=8
mu=[0.1,0.3,0.5,1.0,3,5,10.,100]
colors=['blue','red','green','magenta','brown','orange','violet','yellow']
Ciquantile=np.zeros((size,nProb)) # quantiles of Ci, approximate
chi2quantile=np.zeros((size,nProb)) # this is done for comparison

sizeSample=100000
sampleC=np.zeros((size,sizeSample))
for i in range(size):
    # 1. generate random values
    sampleP=poisson.rvs(mu[i],size=sizeSample) # random values
    # 2. Now sample values of Ci
    for k in range(sizeSample):
        sampleC[i,k]=CContribwTwo(sampleP[k],mu[i])
    # 3. find the p quantile
    print('Ci quantiles for mu=%4.3f'%mu[i])
    for j in range(nProb):
        Ciquantile[i,j]=np.quantile(sampleC[i],p[j])
        chi2quantile[i,j]=chi2.ppf(p[j],1) # this remains the same as a function ofi mu, 
        # so it should not be re-calculated
    print('Quantiles of Ci     ',Ciquantile[i,:],'\nQuantiles of chi2(1)',chi2quantile[i,:])

# ==================================================================
# Print the unique values of C_i for mu=0.1
CiUnique=np.unique(sampleC[0,:])
print('Unique value of Ci for mu=0.1',CiUnique)

# Check that these correspond to n=0,1,2,3 counts
A=np.zeros(4)
muTest=0.1
for i in range(4):
    A[i]=CContribwTwo(i,muTest)
print(A)

# ===================================================================
# Plot a few specific cases
PLOT=1
bins=np.linspace(0.,10,1000)

fig,ax=plt.subplots(figsize=(8,6))

for i in range(size*PLOT):
    n,bins,_=ax.hist(sampleC[i],bins=bins,cumulative=True,histtype='step',log=False,density=True,linewidth=2,alpha=0.5,color=colors[i],label='mean $\mu_i$=%2.2f'%mu[i])
    # overplot the chi2(1) distribution using the bins in the histogram plot
    binsCenter=(bins[1:] + bins[:-1])/2 # arithmetic mean
#    binsCenter=np.sqrt(bins[1:] * bins[:-1]) # geometric mean

ax.plot(binsCenter,chi2.cdf(binsCenter,1),linewidth=2,color='black',linestyle='--',label='$\chi^2(1)$')
ax.set_xlabel('Values of $C_i$')
ax.set_ylabel('Sample Cumulative Distribution')
ax.set_yticks(np.linspace(0,1,11,dtype=float))
ax.set_xticks(np.arange(0,11,step=1.0))
ax.set_ylim(0,1)
ax.set_yscale('linear')
ax.set_xscale('log')
ax.set_xlim(0.07,10)

# formatting log axis with linear labels
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())


plt.grid(which='both')
plt.legend(loc=4,prop={'size': 12})
plt.savefig('CiCritical.pdf')
#plt.show()


