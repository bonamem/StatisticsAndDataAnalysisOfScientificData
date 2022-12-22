exec(open('imports.py').read())

# integrate 1/4 of a circle
def fCircle(x,r):
	return (r**2-x**2)**0.5

def expFInv(p,lambdaPar):
	value = - np.log(1.-p)/lambdaPar
	return value

def GaussFInv(p1,p2,mu,sigma):
	value = ((-2*np.log(1-p1))**0.5)*np.cos(2*np.pi*p2)*sigma+mu
	return value

# (A) Simple estimate of the integral, also prob20.3
print('problem 20.3')
a=0
b=1
N=100000
#N=1000
unifSample=uniform.rvs(size=N)
fSample=fCircle(unifSample,1)
fSampleMean=np.mean(fSample)
fSampleVar=np.var(fSample)
PiSim=4.0*fSampleMean
PiSimErr=4.0*fSampleVar**0.5/N**0.5 # error according to Eq. 20.6
print('(a) Monte Carlo standard integration')
print('for N=%d expected precision of %3.4f %%'%(N,100*PiSimErr/PiSim))
print('Pi for N=%d: %4.4f +-%4.4f (%4.4f, error=%3.4f or %3.3f %%)'%
		(N,PiSim,PiSimErr,np.pi,PiSim-np.pi,100*(PiSim-np.pi)/np.pi))



# (B) Dart Monte Carlo
# y(-1,1) is related to x(0,1) by y=2x-1 or x=(y+1)/2
print('Problem 20.4')
y1=2*uniform.rvs(size=N)-1 # between -1 and 1
y2=2*uniform.rvs(size=N)-1 # same
NR=0 # number of simulated data points within R
for i in range(N):
	if (y1[i]**2+y2[i]**2)<= 1:
		NR=NR+1

fig,ax=plt.subplots(figsize=(8,8))
# estimated binomial probabilities	
pHat=NR/N
qHat=1-pHat
V=2*2
A=V*NR/N
VarA = (4.0**2)*pHat*qHat/N
plt.plot(y1,y2,linestyle='',marker='x',color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim((-1,1))
plt.ylim((-1,1))
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.25))
plt.grid(which='both')

circle=plt.Circle((0,0),radius=1,color='blue',linewidth=2,linestyle='--',fill=False)
ax.add_patch(circle)

plt.savefig('dartMC.pdf')
print('(b) Dart method: A=%4.4f+-%4.4f (%d out of %d)'%(A,VarA**0.5,NR,N))
print(' for N=%d expected precision of %3.4f pct'%(N,100*VarA**0.5/np.pi))
# ============================
# ====== simulation of an exponential distribution

N=1000 		# number of samples
lambdaPar=1.0	# parameter of the exponential distribution
uRand=uniform.rvs(size=N)
uRand2=uniform.rvs(size=N)
#print('uRand',uRand)
lambdaRand=expFInv(uRand,lambdaPar)
# also simulate a Gaussian from 2 uniform
mu=5
sigma=1
gaussRand=GaussFInv(uRand,uRand2,mu,sigma)
xaxis=np.linspace(0,10,1000)
# theoretical PDF of an exponential variable
lambdaPDF=expon.pdf(xaxis,scale=1/lambdaPar)
gaussPDF=norm.pdf(xaxis,loc=mu,scale=sigma)

fig,ax=plt.subplots(figsize=(8,6))
ax.xaxis.set_major_locator(MultipleLocator(1.0))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
plt.grid(which='both')
plt.hist(lambdaRand,bins=50,linewidth=2,histtype='step',density=True,label='Simulated Exponential Sample',color='black')
plt.plot(xaxis,lambdaPDF,linewidth=2,label='Exponential distribution ($\lambda=1$)',color='blue')
plt.hist(gaussRand,bins=50,linewidth=2,histtype='step',density=True,label='Simulated Gaussian Sample',color='grey')
plt.plot(xaxis,gaussPDF,linewidth=2,label='Gaussian distribution ($\mu=5$, $\sigma=1$)',color='red')
plt.legend(prop={'size': 12})
plt.xlim((0,10))
plt.xlabel('x')
plt.ylabel('Distribution')
plt.savefig('exponMonteCarlo.pdf')




