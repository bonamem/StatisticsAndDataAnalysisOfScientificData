exec(open('imports.py').read())


# Gaussian distribution
fig,ax=plt.subplots(figsize=(8,6))
xmin=-10
xmax=10

x=np.linspace(xmin,xmax,10000,endpoint=True)

sigma=1.0
S=1 # np.e # Euler's number
mu = np.log(S)
loc,scale=mu,sigma

yLog=lognorm.pdf(x,s=sigma,scale=S)
yNorm=norm.pdf(x,loc,scale)

# Now also check directly that the lognormal distribution
# is the same thing as a normal distr for z=log X
xLog=[np.log(xi) for xi in x if xi>0]
#print(yLog)
yNormxLog=norm.pdf(xLog,loc,scale)





# ------------------------------------------

ax.plot(x,yNorm,color='black',linewidth=2,label='Normal Variable Y=ln X~N($\mu$,$\sigma$)')
ax.plot(x,yLog,color='blue',linewidth=2,label='Lognormal Variable X~lognorm($\mu$,$\sigma$)')

#ax.plot(xLog,yNormxLog,color='red',linestyle='--',label='Normal with log(x) at x axis')

ax.set_xlim(-2,5)
ax.set_ylim(0,0.7)
ax.set_xlabel('x')
ax.set_ylabel('Probability Distribution')
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.yaxis.set_major_locator(MultipleLocator(0.20))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
plt.grid(which='both')
#ax2=ax.twiny()
#ax2.xaxis.label.set_color('red')
#ax2.tick_params(axis='x', colors='red')
#ax2.plot(xLog,yNormxLog,color='red',label='Normal Variable')
#ax2.set_xlabel('ln X')
#ax2.set_xlim(-2,5)
ax.legend(prop={'size' : 12},loc=1)
plt.savefig('lognormal.pdf')


