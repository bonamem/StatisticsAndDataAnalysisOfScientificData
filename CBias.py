exec(open('imports.py').read())
# Numerical calculation of the bias using chi^2 as a function of Poisson mean
# when fitting data with a constant model


#mu=[0.01,0.03,0.05,0.1,0.3,0.5,1,3,5,10,30,50,100]
# WARNING: can't use datasets with 0 counts, since the estimated variance would be 0.

mu=np.logspace(1,3,30) # Values of parent Poisson means to try
nSim=1000 # Number of simulated datasets for a given value of mu
N=100   # Number of datapoints in the dataset

sampleP=np.zeros(nSim)
a=np.zeros(nSim)    # for best-fit a
aB=np.zeros(nSim)   # for best-fit a from chi^2 fit

aBMean=np.zeros(len(mu))
aBStd=np.zeros(len(mu))
aMean=np.zeros(len(mu))
aStd=np.zeros(len(mu))
Bias=np.zeros(len(mu))

for i in range(len(mu)):
    for j in range(nSim):
        sampleP=poisson.rvs(mu[i],size=N) # random values

        a[j]=np.average(sampleP)                    # this is the sample mean
        aB[j]=np.average(sampleP,weights=sampleP**(-2.0))# this is the weighted mean
    aMean[i]=np.mean(a)
    aStd[i]=np.std(a)
    aBMean[i]=np.nanmean(aB)
    aBStd[i]=np.nanstd(aB)
    Bias[i]=100*(aBMean[i]-mu[i])/mu[i]
    print('mu=%3.3f: a=%3.3f+-%3.3f, aB=%3.3f+-%3.3f, fract. bias: %3.3f'%(mu[i],aMean[i],aStd[i],aBMean[i],aBStd[i],Bias[i]))

# ============================================
# Now summarize the simulation into a plot
fig,ax=plt.subplots(figsize=(8,6))

ax.errorbar(mu,aMean,yerr=aStd,label='Poisson fit (sample average)')
ax.errorbar(mu,aBMean,yerr=aBStd,label='$\chi^2$ fit (weighted mean)')

ax.set_ylim(10,300)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(10,300)
ax.set_xlabel('Parent Poisson mean')
ax.set_ylabel('Measured Value')

# formatting log axis with linear labels
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())

plt.grid(which='both')
plt.legend(loc=4,prop={'size': 12})
plt.savefig('CBias.eps')


# ============================================
# Plot the bias
fig,ax=plt.subplots(figsize=(8,6))
ax.plot(mu,Bias,color='black',linewidth=2)
ax.set_ylim(-30,6)
ax.set_yscale('linear')
ax.set_xscale('log')
ax.set_xlim(10,1000)
ax.set_xlabel('Parent Poisson mean')
ax.set_ylabel('Bias (Percentage of Poisson mean)')
ax.hlines(0,10,1000,linestyle='--',linewidth=2,color='blue')
# formatting log axis with linear labels
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())

plt.grid(which='both')

plt.savefig('CBias2.pdf')



