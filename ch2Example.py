exec(open('imports.py').read())

# Simulate a normal distribution
fig,ax=plt.subplots(figsize=(8,6))

# Simulate N sample from a Gaussian distribution
N=1000
mu=5
sigma=1.
sample=norm.rvs(size=N,loc=mu,scale=sigma)
bins=np.linspace(0,10,51)
ax.hist(sample,bins=bins,histtype='bar',hatch='///',edgecolor='black',color='white',cumulative=False,density=True,linewidth=2,alpha=0.75,label='Sample distribution')
# the expected distribution for large N is a Gaussian  ---------------------------
ax.plot(bins,norm.pdf(bins,loc=mu,scale=sigma),color='red',linewidth=2,label='Parent distribution')
print(bins)
# --------------------------------------------------------------------------------
# Overplot it using number instead of pdf
ax.set_xlabel('Variable X')
ax.set_ylabel('Probability of occurrence')
ax2=ax.twinx()
ax2.hist(sample,bins=bins,histtype='bar',alpha=0)
ax2.set_ylabel('Number of occurrence')
ax.legend(loc=2,prop={'size': 12})
#plt.show()
plt.savefig('ch2Example.pdf')

# -------------------------------------------
# also plot an exponential function
x=np.linspace(0,10,100)
Lambda=0.5
fig,ax=plt.subplots(figsize=(8,6))
ax.plot(x,expon.pdf(x,scale=1/Lambda),color='black',linewidth=2,label='Exponential PDF')
ax.plot(x,expon.cdf(x,scale=1/Lambda),color='black',linestyle='--',linewidth=2,label='Exponential CDF')
ax.set_xlabel('Variable X')
ax.set_ylabel('Distributions f(x) and F(x)')
ax.legend(loc=2,prop={'size': 12})
#plt.show()
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.10))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.set_ylim(0,1)
ax.set_xlim(0,10)

plt.grid()
plt.savefig('ch2Exponential.pdf')

# -------------------------------------------
# For Chapter 7, use the figure for confidence intervals

x=np.linspace(0,10,100)
Lambda=0.5
fig,ax=plt.subplots(figsize=(8,6))
ax.plot(x,expon.pdf(x,scale=1/Lambda),color='black',linewidth=2,label='Exponential PDF')
ax.plot(x,expon.cdf(x,scale=1/Lambda),color='black',linestyle='--',linewidth=2,label='Exponential CDF')
ax.set_xlabel('Variable X')
ax.set_ylabel('Distributions f(x) and F(x)')
#plt.show()
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.10))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.set_ylim(0,1)
ax.set_xlim(0,10)

# add the 90% confidence intervals
pp5=expon.ppf(0.05,scale=1/Lambda)
pp95=expon.ppf(0.95,scale=1/Lambda)

ax.vlines(pp5,0,1,color='gray')
ax.vlines(pp95,0,1,color='gray')
ax.hlines(0.05,0,10,color='red',linestyle='--')
ax.hlines(0.95,0,10,color='red',linestyle='--')
ax.axvspan(pp5,pp95,color='gray',alpha=0.5,label='90% Confidence Interval')
plt.grid()
ax.legend(loc=5,prop={'size': 12})
plt.savefig('ch7Exponential.pdf')




