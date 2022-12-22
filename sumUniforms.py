exec(open('imports.py').read())
# Make figure 4.1 (and problem 4.4)
# This is the sum of 100 uniforms, with a number of samples n=1,000 or 100,000


N=100   # number of uniform distributions
n=1000  # number of samples for each variable

sample=np.zeros((N,n))
sampleSum=np.zeros(n)
# 1. Simulate n samples from N random variables
for i in range(N):
    sample[i,:]=uniform.rvs(size=n)
#2. Sum the values of the N variables, sample by sample
    sampleSum[:]=sum(sample[:,])
#3. Plot the sample distribution of the sum
# =========================================

fig,ax=plt.subplots(figsize=(8,6))

bins=np.linspace(40,60,100)
ax.hist(sampleSum,bins=bins,cumulative=False,histtype='bar',log=False,density=True,linewidth=2,alpha=0.5,color='black',label='Sum of %d uniform variables with %d samples'%(N,n))
ax.hist(sampleSum,bins=bins,cumulative=False,histtype='step',color='black',density=True)
# the expected distribution for large N is a Gaussian  ---------------------------
meanU=0.5 # mean of U(0,1)
varU=1./12. # variance of U(0,1)
mu=N*meanU
var=N*varU
ax.plot(bins,norm.pdf(bins,loc=mu,scale=var**0.5),color='red',linewidth=2,label='Gaussian distribution')
# --------------------------------------------------------------------------------
ax.set_xlabel('y')
ax.set_ylabel('Probability of occurrence')

# -------------------------------------
# ---- overplot with 100,000 samples
n=100000  # number of samples for each variable
sampleSum=np.zeros(n)
sample=np.zeros((N,n))
for i in range(N):
    sample[i,:]=uniform.rvs(size=n)
#2. Sum the values of the N variables, sample by sample
    sampleSum[:]=sum(sample[:,])
bins=np.linspace(40,60,100)
ax.hist(sampleSum,bins=bins,cumulative=False,histtype='bar',log=False,density=True,linewidth=2,alpha=0.5,color='blue',label='Sum of %d uniform variables with %d samples'%(N,n))
ax.hist(sampleSum,bins=bins,cumulative=False,histtype='step',color='blue',density=True)



plt.legend(loc=2,prop={'size': 12})
#plt.show()



plt.savefig('sumUniforms.pdf')
