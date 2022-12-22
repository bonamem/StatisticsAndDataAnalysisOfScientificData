exec(open('imports.py').read())

# Python code to do statistics from Hubble data
# Bonamente 2020

dataTypes=[('name','S8'),('v','f'),('N','I'),('m','f')]

data = np.genfromtxt('hubbleHumasonData.dat',skip_header=1,dtype=dataTypes)
print(data)

v=data['v']
m=data['m']

logv=np.log10(v)

# Do a simple linear regression (with no errors) prob11.1

slope, intercept, r_value, p_value, std_err = stats.linregress(m,logv)
print("Problem 11.1")
print("slope: %f    intercept: %f" % (slope, intercept))
print('Problem 14.3')
print("r=%3.4f, R-squared: %f" % (r_value,r_value**2))
print("============")
# check that these formulas are the same as using variance and covariance
N=len(m)
covMatrix=np.cov(m,logv,ddof=1)
covXY=covMatrix[0,1]
varX=covMatrix[0,0]
varY=covMatrix[1,1]
EX=np.mean(m)
EY=np.mean(logv)
b=covXY/varX
a=EY-b*EX
print('var(X)=%3.3f, Cov(X,Y)=%3.3f, a=%3.3f, b=%3.3f'%(varX,covXY,a,b))
# Now also repeat the fit but X/Y, and then converting back to a and b
bXY=varY/covXY
aXY=EY-bXY*EX
print(' aXY=%3.3f, bXY=%3.3f'%(aXY,bXY))

fig,ax=plt.subplots(figsize=(8,6))

xmin=11
xmax=20
xline=[xmin,xmax]
# regression of Y on X
yline=[intercept+slope*xline[0],intercept+slope*xline[1]]
# regression of X on Y
ylineXY=[aXY+bXY*xline[0],aXY+bXY*xline[1]]

plt.plot(m,logv,linestyle='',marker='o',color='black', label='Hubble data (no errors)')
plt.plot(xline,yline,color='black',linewidth=2,label='Linear regression of Y on X')
plt.plot(xline,ylineXY,color='blue',linewidth=2,linestyle='--',label='Linear regression of X on Y')
plt.xlabel('Magnitude m')
plt.ylabel('Log velocity')
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.25))
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(1.0))
plt.grid(which='both')


# Now get the model variance and then repeat the fit - prob12.1
print('Problem 12.1')
ymodel=linear(m,intercept,slope)
dof=2# number of free parameters (notice the misnomer)
mVar=modelVariance(logv,ymodel,dof)
print('Model variance is %f (sqrt: %f)'%(mVar,mVar**0.5))

modelError=np.ones(len(m))*(mVar**0.5)
popt,pcov = curve_fit(linear,m,logv,sigma=modelError)
print("a=%4.3f +- %4.3f, b=%4.3f +- %4.3f"%(popt[0],pcov[0][0]**0.5,popt[1],pcov[1][1]**0.5))
print("Cov(a,b)=%3.3f, or r=%3.3f between a and b"%(pcov[0][1],pcov[0][1]/(pcov[0][0]*pcov[1][1])**0.5))
chi2min=chisquared(logv,modelError,ymodel)
print("chi2min using the model sample variance: %3.3f"%chi2min)
#Add the error bars from model variance to the plot
#plt.errorbar(m,logv,yerr=modelError,marker='.',linestyle='',color='blue',capsize=2,label='model errors')
plt.legend(prop={'size': 12})
plt.savefig('hubbleFit.pdf')


# For problem 17.4, do a fit with sigma_i=0.01 and then find the intrinsic scatter
print('problem 17.4')
sigma=0.01*np.ones(len(logv)) # this is the error assumed
sigmaInt=intrinsicScatter(logv,ymodel,sigma,dof)
print('intrinsic scatter: %4.4f'%sigmaInt)

## ==============================================
# jackknife  on the Hubble data
aJackj=np.zeros(N)
bJackj=np.zeros(N)
aJackjStar=np.zeros(N)
bJackjStar=np.zeros(N)
mj=np.zeros((N,N-1)) # these are the resampled datasets, N sets of size N-1
logvj=np.zeros((N,N-1))
# start with the estimates from the original data
bhat,ahat,r_value, p_value, std_err = stats.linregress(m,logv)
for j in range(N):
	mj[j,:]=np.delete(m,j)
	#print(j, m,mj[0,:])
	logvj[j,:]=np.delete(logv,j)
	# these are the theta_j estimates from Zj
	bJackj[j],aJackj[j],r_value, p_value, std_err = stats.linregress(mj[j,:],logvj[j,:])
	aJackjStar[j]=N*ahat-(N-1)*aJackj[j]
	bJackjStar[j]=N*bhat-(N-1)*bJackj[j]
meanaJackStar=np.mean(aJackjStar)
erraJackStar=np.std(aJackjStar,ddof=1)/N**0.5 # this is error on mean
meanbJackStar=np.mean(bJackjStar)
errbJackStar=np.std(bJackjStar,ddof=1)/N**0.5

print('Jackknife estimates: a=%3.3f +-%3.3f, b=%3.3f+-%3.3f'%(meanaJackStar,erraJackStar,meanbJackStar,errbJackStar))

# ==================================
# bootstrap, also Problem 20.5
print('Problem 20.5, bootstrap of Hubble data')
NBoot=10000
aBoot=np.zeros(NBoot)
bBoot=np.zeros(NBoot)
indices=np.arange(N)
print(indices,m,m[indices])
for i in range(NBoot):
	indicesChoice=choices(indices,k=N)
	bBoot[i],aBoot[i],r_value, p_value, std_err = stats.linregress(m[indicesChoice],logv[indicesChoice])
fig,ax=plt.subplots(figsize=(8,6))
binsa=np.linspace(0,2,60)
binsb=np.linspace(0.12,0.28,60)
print('bins:',binsa,binsb)
plt.hist(aBoot,bins=binsa,linewidth=2,density=True,histtype='step',color='black',label='Parameter a')
plt.hist(bBoot,bins=binsb,linewidth=2,density=True,histtype='step',color='blue',label='Parameter b')
plt.xlabel('Parameter value')
plt.ylabel('Distribution')
plt.legend(prop={'size': 12})
plt.xlim((0.1,1.0))
plt.ylim((0,5))
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(MultipleLocator(0.025))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.2))
plt.grid(which='both')

# mean, median and percentiles
xaxis=np.linspace(0,2,2000)
p=0.683
alpha=(1-p)/2
beta=(1+p)/2
aBootMean=np.mean(aBoot)
aBootMedian=np.median(aBoot)
erraBoot=np.std(aBoot)
alphaa=np.percentile(aBoot,alpha*100)
yaxis=norm.pdf(xaxis,loc=aBootMean,scale=erraBoot)
plt.plot(xaxis,yaxis,linewidth=1,linestyle='--',color='black')
betaa=np.percentile(aBoot,beta*100)
bBootMean=np.mean(bBoot)
bBootMedian=np.median(bBoot)
errbBoot=np.std(bBoot)
alphab=np.percentile(bBoot,alpha*100)
betab=np.percentile(bBoot,beta*100)
yaxis=norm.pdf(xaxis,loc=bBootMean,scale=errbBoot)
plt.plot(xaxis,yaxis,linewidth=1,linestyle='--',color='blue')
print('Bootstrap estimates (mean and std. dev.: a=%3.3f+-%3.3f, b=%3.3f+-%3.3f'
	%(aBootMean,erraBoot,bBootMean,errbBoot))
print('Bootstrap median and p=%3.3f central intervals:'%p)
print('a=%3.3f (%3.3f-%3.3f), b=%3.3f ((%3.3f-%3.3f)'
	%(aBootMedian,alphaa,betaa,bBootMedian,alphab,betab))
plt.savefig('bootstrapHubble.pdf')
