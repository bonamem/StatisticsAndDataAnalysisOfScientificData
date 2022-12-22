exec(open('imports.py').read())

# Make a simulation of Delta chi^2 for the detection of a line
# Assume that there are 100 bins with a constant model
# and that the line lies in one specific bin, say the first.

alpha=100 # parent value of the continuum
NBins=100
NSim=1000	# number of simulations
DeltaChi2=np.zeros(NSim)
b=np.zeros(NSim)
x=np.arange(NBins)	# x-axis of data, irrelevant
for i in range(NSim):
	data = poisson.rvs(alpha,size=NBins)
	error=data**0.5
	poptC, pcovC = curve_fit(constant, x,data, sigma=error,absolute_sigma=True)	
	modelC=constant(x,poptC[0])
	chiminC=chisquared(data,error,modelC)
	poptCP, pcovCP = curve_fit(constantPlusOne, x,data, sigma=error,absolute_sigma=True,bounds=([-100,-100],[100,0]))
	modelCP=constantPlusOne(x,poptCP[0],poptCP[1])
	b[i]=poptCP[1] # best-fit of parameter "b" of line
	chiminCP=chisquared(data,error,modelCP)
	DeltaChi2[i]=chiminC-chiminCP
	print(DeltaChi2[i],b[i])

# now plot distribution of DeltaChi2
fig,ax=plt.subplots(figsize=(8,6))
xaxis=np.linspace(0,10,100)
yaxis=chi2.pdf(xaxis,1)
plt.hist(DeltaChi2,bins=100,density=True,color='black')
plt.plot(xaxis,yaxis)
plt.xlabel('$\Delta chi^2$')
plt.ylabel('Prob. Distribution')
plt.xlim(0,6)
plt.savefig('Deltachi2F.pdf')

# plot distribution of b values
fig,ax=plt.subplots(figsize=(8,6))
plt.hist(b,bins=100,density=True,color='red')
plt.xlim(-30,30)
plt.xlabel('$b$ parameter')
plt.ylabel('Prob. Distribution')
plt.savefig('Deltachi2F-b.pdf')
