exec(open('imports.py').read())

# ==============================================
# Plot the rdist for the book.

fig,ax=plt.subplots(figsize=(8,6))

#Note that GAmma(0) is undefined, so f=2 is not meaningful
N=[3, 4,5,7,12,22]  # NUMBER OF data points for r
colors=['black','grey','red','blue','lightgreen','darkgreen']
k=2 # this is simple linear regression with 2 free parameters
x = np.linspace(-1,1,100)
for i in range(len(N)):
    f=N[i]-k
    rdistVar=rdist.var(f)
    plt.plot(x,rdist.pdf(x, f),color=colors[i],linewidth=2,label='f=%d'%f)
    print('f=%d: var=%2.4f, 1/(f+1)=%3.4f'%(f,rdistVar,1/(f+1)))
# overplot the N(0,1) for reference - not useful
#plt.plot(x,norm.pdf(x),color='black',linestyle='--',linewidth=2)

plt.legend(loc=1,prop={'size': 12})
#plt.xlabel('r or r2')
plt.grid(which='both')
plt.xlabel('r')
plt.xlim((-1,1))
plt.ylim((0,1.8))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))

plt.ylabel('Probability Distribution Function')
plt.savefig('rDist.pdf')

# ===========================================
# Also plot the r^2 distribution
fig,ax=plt.subplots(figsize=(8,6))

k=2 #k=2 or m=1 is for R^2=r^2, simple linear regression
xplus = np.linspace(0,1,100)
for i in range(len(N)):
    a=(k-1)/2.
    b=(N[i]-k)/2.
    f=N[i]-k
    plt.plot(xplus,beta.pdf(xplus,a,b),color=colors[i],linewidth=2,linestyle='-',label='f=%d'%f)
plt.legend(loc=1,prop={'size': 12})
plt.ylim((0,4))
#plt.xlabel('r or r2')
plt.grid(which='both')
plt.xlabel('$r^2$')
plt.xlim((0,1))
ax.yaxis.set_major_locator(MultipleLocator(1.0))
ax.yaxis.set_minor_locator(MultipleLocator(0.2))
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
plt.ylabel('Probability Distribution Function')
plt.savefig('r2Dist.pdf')

# =======================================================
# Also plot the R^2 distribution for multiple regressions
fig,ax=plt.subplots(figsize=(8,6))
N=[4,5,8,13,23]  # NUMBER OF data points for r
k=3 #k=2 or m=1 is for R^2=r^2, simple linear regression
xplus = np.linspace(0,1,100)
for i in range(len(N)):
    a=(k-1)/2.
    b=(N[i]-k)/2.
    f=N[i]-k
    plt.plot(xplus,beta.pdf(xplus,a,b),color=colors[i],linewidth=2,linestyle='-',label='f=%d'%f)
plt.legend(loc=1,prop={'size': 12})
plt.ylim(0,4)
#plt.xlabel('r or r2')
plt.grid(which='both')
plt.xlabel('$R^2$ ($m$=2)')
plt.xlim((0,1))
ax.yaxis.set_major_locator(MultipleLocator(1.0))
ax.yaxis.set_minor_locator(MultipleLocator(0.2))
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
plt.ylabel('Probability Distribution Function')
plt.savefig('R2Dist.pdf')



# Try and calculate r for N=2 data
X=uniform.rvs(size=2)
Y=uniform.rvs(size=2)
b,a,r,p,stdrerr=linregress(X,Y)
print('b=%3.2f, r=%3.3f, p=%3.3f'%(b,r,p))
bPr,aPr,rPr,pPr,stdrerr=linregress(Y,X)
print('bPr=%3.2f, b*bPr=%3.3f, r=%3.3f, p=%3.3f'%(bPr,b*bPr,rPr,pPr))

# ================================================
# Critical value at p significance
N=1079 # Number of data points
k=2 # linear regression
f=N-k# Number of degrees of freedom
# Keep in mind that we look for |r|>rCrit, i.e., a two-sided hypothesis
# so, if we want 99% prob that |r|>rCrit, and since 
# f(r) is symmetrical around 0, we need to do:
p=0.99
q=1-p # 1\%
q=q/2 # 0.5% to the right, and also 0.5% to the left
rCrit=rdist.ppf(1-q,f)
print('N=%d, Two-sided critical value of r at %3.4f significance is %4.3f'%(N,p,rCrit))


# =================================================
# Now also the distribution of r^2, which is a Beta(1,N-1)
# or, for multiple regression Beta(k-1, N-k)
# which makes sense since we are only interested
# in absolute values of r.
a=(k-1)/2
b=(N-k)/2
r2Crit=beta.ppf(p,a,b)
print('N=%d, Critical value of r^2 at %3.4f significance is %4.5f'%(N,p,r2Crit))
print('=> Two-sided Critical value of r at %3.4f significance is %4.3f'%(p,r2Crit**0.5))

#######################################################
#### Solve problem 14.6 using both the standard beta distributions
print('prob14.6')
R2=0.5
N=20
m=5
f=N-m-1 
p=0.90
q=1-p # 10\%
q=q/2 # 5% to the right, and also 5% to the left
# Using rDist, or symmetric beta distr - this is NOT applicable to m>2
rCrit=rdist.ppf(1-q,f)
print('N=%d, m=%d Two-sided critical value of r at %3.4f significance is %4.3f'%(N,m,p,rCrit))
print('Note: this does not apply to R with m>2')
# using R^2, or standard beta distribution
a=m/2
b=(N-m-1)/2
r2Crit=beta.ppf(p,a,b)
print('N=%d, Critical value of R^2 at %3.4f significance is %4.5f'%(N,p,r2Crit))
print('=> Two-sided Critical value of R at %3.4f significance is %4.3f'%(p,r2Crit**0.5))


