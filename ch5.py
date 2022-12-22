exec(open('imports.py').read())

# Simulate a normal distribution
fig,ax=plt.subplots(figsize=(8,6))

# exponential distribution

x=np.linspace(0,10,100)
Lambda=0.5
y=expon.pdf(x,scale=1/Lambda)
yCDF=expon.cdf(x,scale=1/Lambda)
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.20))
ax.yaxis.set_minor_locator(MultipleLocator(0.10))


ax.plot(x,y,color='black',label='Probability Distribution f(x)',linewidth=2)
# same curve again. but labeled as quantile function
ax.plot(x,yCDF,color='red',linewidth=2,linestyle='-',label='Quantile Function $F^{-1}(p)$')
ax.plot(x,yCDF,color='black',linewidth=2,linestyle='--',label='Cumulative Distribution F(x)')

ax.set_xlim(0,10)
ax.set_ylim(0,1)
ax.set_xlabel('x')
ax.set_ylabel('Distribution')
plt.grid(which='both')
ax.legend(loc=5,prop={'size': 12})
ax2=ax.twinx()
ax2.set_ylabel('Quantile $p$')
ax2.yaxis.label.set_color('red')
ax2.tick_params(axis='y', colors='red')


ax3=ax.twiny()
ax3.set_xlim(0,10)
ax3.xaxis.label.set_color('red')
ax3.tick_params(axis='x', colors='red')
ax3.set_xlabel('Quantile Function $x=F^{-1}(p)$',color='red')

#plt.legend(loc=2,prop={'size': 12})
plt.savefig('exponential.pdf')

