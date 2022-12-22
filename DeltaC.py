exec(open('imports.py').read())
# Numerical calculation of Delta C
# In the low--count and low--N limit where no approximation is available
# For the constant model

# Choose a number of measurements and a parent mean
N=10    # number of measurements in teh dataset
mu=0.1  # parent Poisson mean

# Choose a sample size for the simulation
nSize=10000
sampleP=np.zeros(nSize)
sampleCtrue=np.zeros(nSize)
sampleCmin=np.zeros(nSize)
DeltaC=np.zeros(nSize)
Ctrue=np.zeros(nSize)
Cmin=np.zeros(nSize)

# Simulate the distribution of Delta C
for i in range(nSize):
    sampleP=poisson.rvs(mu,size=N) # random values
    a=np.average(sampleP) # this is the best--fit parameter a
    for j in range(N):
#        Calculate the contributions to Ctrue and Cmin
        sampleCtrue[j]=CContribwTwo(sampleP[j],mu)
        sampleCmin[j]=CContribwTwo(sampleP[j],a)
    # Now calculate Cmin and C=C_true(mu)
    Ctrue[i]=np.sum(sampleCtrue)
    Cmin[i]=np.sum(sampleCmin)
    DeltaC[i]=Ctrue[i]-Cmin[i]
    #print(sampleP,end='')
    #print('a=%3.3f, mu=%3.3f '%(a,mu),end='')
    print('%3.3f-%3.3f=%3.3f'%(Ctrue[i],Cmin[i],DeltaC[i]))

fig,ax=plt.subplots(figsize=(8,6))

bins=np.linspace(0.0,10,1000)
n,bins,_=ax.hist(DeltaC,bins=bins,cumulative=True,histtype='step',log=False,density=True,linewidth=2,alpha=1.0,color='black',label='$\Delta C$ for N=%d, $\mu$=%2.2f'%(N,mu))

binsCenter=(bins[1:] + bins[:-1])/2 # arithmetic mean

ax.set_ylim(0,1)
ax.set_yscale('linear')
ax.set_xscale('log')
ax.set_xlim(0.5,10)
plt.grid(which='both')
# formatting log axis with linear labels
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())

# also overplot horizontal lines for notable critical values
hlines=[0.683,0.90,0.95]
for  i in range(3):
    plt.hlines(hlines[i],0.1,10,color='blue',linewidth=2,linestyle='--')


# overplot the chi^2(1) distribution, for comparison
ax.plot(binsCenter,chi2.cdf(binsCenter,1),linewidth=2,color='red',linestyle='--',label='$\chi^2(1)$')

plt.legend(loc=4,prop={'size': 12})
ax.set_xlabel('Values of $\Delta C$')
ax.set_ylabel('Sample Cumulative Distribution')

#plt.show()
plt.savefig('DeltaC.pdf')

# ===============================
# add calculation of binomial probabilities to obtain datasets

n=[0,1,2,3]
for i in range(len(n)):
    print('Binomial prob for n=%d: %4.3f'%(n[i],binom.pmf(n[i],N,mu)))



