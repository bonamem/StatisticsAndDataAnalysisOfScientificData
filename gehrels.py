exec(open('imports.py').read())


# Make a plot to replace Fig. 5.2, to illustrate the confidence
# intervals on a Poisson variable

xmax=10
x=np.linspace(0,xmax,xmax+1,endpoint=True,dtype=int)
n=3 # this is the example of a number of detected counts
mu1=0.83
mu2=7.8
p1 = poisson.pmf(x,mu1)
p2= poisson.pmf(x,mu2)

fig,ax=plt.subplots(figsize=(8,6))

ax.plot(x,p1,color='black',linewidth=2,linestyle=':')#,label='mean $\mu_{lo}$')#=%d'%mu1)
ax.plot(x,p1,color='black',marker='o',linestyle='')
ax.plot(x,p2,color='blue',linewidth=2,linestyle=':')#,label='mean $\mu_{up}$')#=%d'%mu2)
ax.plot(x,p2,color='blue',marker='o',linestyle='')
ax.fill_between(x[3:10],p1[3:10],color='grey',alpha=0.5,label='prob. 1-$p$')
ax.fill_between(x[0:4],p2[0:4],color='blue',alpha=0.5,label='prob. 1-$p$')

plt.xlabel('X')
plt.ylabel('Poisson Distribution')
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.10))
ax.yaxis.set_minor_locator(MultipleLocator(0.10))
ax.vlines(3,0.0,1.0,linestyle='--',color='red',linewidth=2,label='measurement')
plt.hlines(0.05,0.,10.,color='grey')
plt.hlines(0.95,0.,10.,color='grey')
plt.grid(which='both')
ax.set_xlim(0,8)
ax.set_ylim(0,1.0)

# let's add the cumulative distributions
P1 = poisson.cdf(x,mu1)
P2= poisson.cdf(x,mu2)
ax.plot(x,P1,color='black',linewidth=2,label='mean $\mu_{lo}$')
ax.plot(x,P2,color='blue',linewidth=2,label='mean $\mu_{up}$')



plt.legend(loc=1,prop={'size': 12})
plt.savefig('gehrels.pdf')
print(x)


# Check that the S parameters correspond to the p-quantiles
p=[0.9,0.95,0.99,0.841,0.977,0.999]
S=norm.ppf(p)
print(p,S)

# =======================================
# Also check numerical solution of prob7.2
print('problem 7.2 (c)')
def ULfunc(x,p):
	return np.exp(-x)*(1+x)-(1-p)
p=0.84
muup=fsolve(ULfunc,3,args=(p))
print('Upper limit',muup)

