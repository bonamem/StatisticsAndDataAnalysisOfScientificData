exec(open('imports.py').read())

# Do an MCMC on Hubble's data

dataTypes=[('name','S8'),('v','f'),('N','I'),('m','f')]
data = np.genfromtxt('hubbleHumasonData.dat',skip_header=1,dtype=dataTypes)
print(data)

v=data['v']
m=data['m']
logv=np.log10(v)
n=len(m) # number of datapoints
# Set up priors - uniform
aMin=0.2
aMax=0.9
bMin=0.15
bMax=0.25

# Set up proposals - uniform
Deltaa=0.2
Deltab=0.04
# Set up the chain
NMCMC=1000
aMCMC=np.zeros(NMCMC)
bMCMC=np.zeros(NMCMC)
logvMCurrent=np.zeros(n)
logvMPrevious=np.zeros(n)

fout=open('hubbleMCMC.txt','w')
# Start at an arbitrary initial value
aMCMC[0]=0.5*(aMax+aMin)
bMCMC[0]=0.5*(bMax+bMin)
logvMPrevious=linear(m,aMCMC[0],bMCMC[0])	# current model
fout.write('%3.5f  %3.5f\n'%(aMCMC[0],bMCMC[0]))
# variances of the data were not provided - estimate via model sample variance
# fit without errors
slope, intercept, r_value, p_value, std_err = stats.linregress(m,logv)
# best-fit model
ymodel=linear(m,intercept,slope)
# sample model variance
dof=2 # number of free parameters
mVar=modelVariance(logv,ymodel,dof)
print('Model variance is %f (sqrt: %f)'%(mVar,mVar**0.5))
logvErr=np.ones(len(m))*(mVar**0.5)
chi2Previous=chisquared(logv,logvErr,logvMPrevious)

# Now run the MCMC
NAccept=0
for i in range(1,NMCMC):	# start at i=1
	# Draw random values for candidates in the neighborhood of previous parameter
	aPrime=uniform.rvs(loc=aMCMC[i-1]-Deltaa/2,scale=Deltaa,size=1)
	bPrime=uniform.rvs(loc=bMCMC[i-1]-Deltab/2,scale=Deltab,size=1)
	# calculate the likelihood, aka chi^2
	logvMCurrent=linear(m,aPrime,bPrime)
	chi2Current=chisquared(logv,logvErr,logvMCurrent)
	# Calculate the acceptance probability
	alpha=min(1,np.exp((chi2Previous-chi2Current)/2))
	print('a=%3.3f, b=%3.3f, chi2Current=%3.4f, chi2Previous=%3.4f, alpha=%3.4f'
		%(aPrime,bPrime,chi2Previous,chi2Current,alpha))
	u=uniform.rvs(size=1)
	# Decide whether to accept or reject
	if (alpha>=u):	# accept
		aMCMC[i]=aPrime
		bMCMC[i]=bPrime
		chi2Previous=chi2Current# this becomes the previous chi2
		print('Accepted')
		NAccept=NAccept+1
	if (alpha<u):
		aMCMC[i]=aMCMC[i-1]
		bMCMC[i]=bMCMC[i-1]
		print('Rejected')
	fout.write('%3.5f  %3.5f\n'%(aMCMC[i],bMCMC[i]))
fout.close()
print('Acceptance rate of MCMC = %d of %d (%3.4f)'%(NAccept,NMCMC,NAccept/NMCMC))
# Plot the parameter distribution
fig,ax=plt.subplots(1,2,figsize=(8,6))
ax[0].hist(aMCMC,bins=40,linewidth=2,histtype='step',color='black')
ax[0].set_xlabel('Parameter a')
ax[0].xaxis.set_major_locator(MultipleLocator(0.2))
ax[0].xaxis.set_minor_locator(MultipleLocator(0.05))
ax[0].yaxis.set_major_locator(MultipleLocator(NMCMC*0.01))
ax[0].yaxis.set_minor_locator(MultipleLocator(NMCMC*0.005))
ax[1].hist(bMCMC,bins=40,linewidth=2,histtype='step',color='black')
ax[1].set_xlabel('Parameter b')
ax[1].xaxis.set_major_locator(MultipleLocator(0.02))
ax[1].xaxis.set_minor_locator(MultipleLocator(0.005))
ax[1].yaxis.set_major_locator(MultipleLocator(NMCMC*0.01))
ax[1].yaxis.set_minor_locator(MultipleLocator(NMCMC*0.005))
ax[0].set_ylabel('Number of occurrence')
ax[0].set_ylim((0,0.09*NMCMC))
ax[1].set_ylim((0,0.09*NMCMC))
ax[0].grid(which='both')
ax[1].grid(which='both')
for axs in ax.flat:
    axs.label_outer()
plt.savefig('hubbleMCMCHist.pdf')

fig,ax=plt.subplots(2,1,figsize=(8,6))
xaxis=np.arange(NMCMC)
ax[0].scatter(xaxis,aMCMC,s=30,color='none',edgecolor='black',marker='o')
ax[1].scatter(xaxis,bMCMC,s=30,color='none',edgecolor='black',marker='o')
ax[0].grid(which='both')
ax[1].grid(which='both')
ax[0].set_ylabel('Parameter a')
ax[1].set_xlabel('Iteration Number')
ax[1].set_ylabel('Parameter b')
for axs in ax.flat:
	axs.label_outer()
ax[0].set_xlim(0,NMCMC)
ax[1].set_xlim(0,NMCMC)
ax[0].yaxis.set_major_locator(MultipleLocator(0.2))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.05))
ax[1].yaxis.set_major_locator(MultipleLocator(0.02))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.005))
ax[1].xaxis.set_major_locator(MultipleLocator(NMCMC*0.100))
ax[0].xaxis.set_major_locator(MultipleLocator(NMCMC*0.100))
plt.savefig('hubbleMCMCTime.pdf')
