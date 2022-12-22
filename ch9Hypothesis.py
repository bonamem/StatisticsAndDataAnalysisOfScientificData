exec(open('imports.py').read())

# Simulate a normal distribution
fig,ax=plt.subplots(figsize=(8,6))

xObs=61
xmin=50
xmax=100
mu=13
sigma=2**0.5
N=5
x=np.linspace(xmin,xmax,1000)
y=norm.pdf(x,N*mu,sigma*N**0.5)

ax.plot(x,y,color='black',linewidth=2,label='N($N\mu,N\sigma^2$)')
plt.grid(which='both')
ax.set_xlim(53,77)
ax.yaxis.set_major_locator(MultipleLocator(0.02))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1.0))
ax.set_xlabel('Statistic Y')
ax.set_ylabel('Probability Distribution')

# add the 95% confidence limits
x1=N*mu-1.96*(sigma*N**0.5)
x2=N*mu+1.96*(sigma*N**0.5)
x1range=[i for i in np.arange(1000) if x[i]<=x1]
x2range=[i for i in np.arange(1000) if x[i]>=x2]
ax.fill_between(x[x1range], y[x1range],facecolor='none',hatch='XXXX',edgecolor="b",label='95% rejection region')
ax.fill_between(x[x2range], y[x2range],facecolor='none',hatch='XXXX',edgecolor="b")
plt.legend(loc=1,prop={'size':12})
# repeat the same for 68.3% rejection region
x1=N*mu-1*(sigma*N**0.5)
x2=N*mu+1*(sigma*N**0.5)
x1range=[i for i in np.arange(1000) if x[i]<=x1]
x2range=[i for i in np.arange(1000) if x[i]>=x2]
ax.fill_between(x[x1range], y[x1range],facecolor='none',hatch='//',edgecolor="gray",label='68.3% rejection region')
ax.fill_between(x[x2range], y[x2range],facecolor='none',hatch='//',edgecolor="gray")
# also report the measurement, needed for the p-value
ax.vlines(xObs,0,norm.pdf(xObs,N*mu,sigma*N**0.5),color='r',linewidth=2,label='Measurement')
ax.vlines(N*mu+abs(N*mu-xObs),0,norm.pdf(N*mu+abs(N*mu-xObs),N*mu,sigma*N**0.5),color='r',linewidth=2,linestyle='--')
plt.legend(loc=1,prop={'size':12})
ax.set_ylim(0.,0.13)
plt.savefig('hypothesisTesting.pdf')


# ====== pdf of chi^2
xmin=0
xmax=20
dof=5
x=np.linspace(xmin,xmax,1000)
y=chi2.pdf(x,dof)
fig,ax=plt.subplots(figsize=(8,6))

ax.plot(x,y,linewidth=2,color='black',label='$\chi^2$(N)')
ax.set_xlabel('$\chi^2$')
ax.set_ylabel('Probability Distribution')
ax.yaxis.set_major_locator(MultipleLocator(0.02))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
plt.grid(which='both')
ax.set_xlim(0,20)
ax.set_ylim(0,0.16)
chi2Crit95=chi2.ppf(0.95,dof)
chi2Crit68=chi2.ppf(0.683,dof)
x1range=[i for i in np.arange(1000) if x[i]>=chi2Crit95]
x2range=[i for i in np.arange(1000) if x[i]>=chi2Crit68]
ax.fill_between(x[x2range], y[x2range],facecolor='none',hatch='//',edgecolor="gray",label='68.3% rejection region')
ax.fill_between(x[x1range], y[x1range],facecolor='none',hatch='XXXX',edgecolor="b",label='95% rejection region')

plt.legend(loc=1,prop={'size':12})
plt.savefig('chi2.pdf')

# example in section 9.3
# also calculate critical value for N=5 dof
p=[0.683,0.95]
chi2Crit68=chi2.ppf(p,5)
print('chi2 crit: ',chi2Crit68)

# p value
pVal=chi2.sf(9,5)
print('p-value',pVal)

# new table with Delta chi^2 values
print('New table in Ch. 12')
p=[0.683,0.90,0.99]
dof=[1,2,3,4,5]
for i in range(5):
    chi2Crit=chi2.ppf(p,dof[i])
    print('chi2 crit: ',chi2Crit)

# another example for the sampling distr. of variance
x=np.array([10, 12, 15, 11, 13, 16, 12, 10, 18, 13])
N=len(x)
xbar=np.mean(x)
s2=np.var(x,ddof=1)
S2=s2*(N-1)
print('xbar=%3.3f, S2=%3.3f'%(xbar,S2))
pVal=chi2.sf(32,9)
print('p-value',pVal)

# ==============================================
# ---- another chi^2 plot for the appendix

fig,ax=plt.subplots(figsize=(8,6))
colors=['darkgrey','gray','dimgray']
dof=[5,20,100]
xmin=0
for i in range(len(dof)):
	xmax=dof[i]+5*(2*dof[i])**0.5 # range of chi^2 distr. 
	x=np.linspace(xmin,xmax,1000)
	xOverf=(x-dof[i])/(2*dof[i])**0.5
	yNorm=chi2.pdf(x,dof[i])/max(chi2.pdf(x,dof[i])) # normalized chi^2 distr
	yNormCDF=chi2.cdf(x,dof[i])
	plt.plot(xOverf,yNorm,linewidth=2,color=colors[i],label='$\chi^2$(%d)'%dof[i])
	plt.plot(xOverf,yNormCDF,linewidth=2,color=colors[i])
x=np.linspace(-5,5,1000)
loc=0
scale=1
yGauss=norm.pdf(x,loc,scale)/norm.pdf(0,loc,scale)
plt.plot(x,yGauss,label='N(0,1)',color='black',linewidth=2,linestyle='-')
yGaussCDF=norm.cdf(x,loc,scale)
plt.plot(x,yGaussCDF,color='black',linewidth=2,linestyle='-')

ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.2))
plt.grid(which='both')
plt.xlim((-3,3))
plt.ylim((0,1))
plt.xlabel('Standardized x')
plt.ylabel('Normalized distributions')
plt.legend(loc=2,prop={'size':12})
plt.savefig('chi2Appendix.pdf')

# ==============================================
# F distribution
xmin=0
xmax=10
dfn=5   # degrees of freedom of numerator
dfd=5   # degrees of freedom of denominator
fig,ax=plt.subplots(figsize=(8,6))

x=np.linspace(xmin,xmax,1000)
y=f.pdf(x,dfn,dfd)
plt.plot(x,y,color='black',linewidth=2,label='F distr. ($f_1=5$, $f_2$=5)')
ax.set_xlabel('F')
ax.set_ylabel('Probability Distribution')
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
plt.grid(which='both')
ax.set_xlim(0,8)
ax.set_ylim(0,0.68)

fCrit68=f.ppf(0.683,dfn,dfd)
fCrit95=f.ppf(0.95,dfn,dfd)
x1range=[i for i in np.arange(1000) if x[i]>=fCrit95]
x2range=[i for i in np.arange(1000) if x[i]>=fCrit68]
ax.fill_between(x[x2range], y[x2range],facecolor='none',hatch='//',edgecolor="gray",label='68.3% rejection region')
ax.fill_between(x[x1range], y[x1range],facecolor='none',hatch='XXXX',edgecolor="b",label='95% rejection region')

# add also another distribution with 1 fewer dof
dfn=4
dfd=4
y=f.pdf(x,dfn,dfd)
plt.plot(x,y,color='black',linewidth=2,linestyle='--',label='F distr. ($f_1=4$, $f_2$=4)')

# rearrange the labels
handles,labels = ax.get_legend_handles_labels()

handles = [handles[0], handles[2], handles[3],handles[1]]
labels = [labels[0], labels[2], labels[3],labels[1]]
plt.legend(handles,labels,loc=1,prop={'size':12})


plt.savefig('f.pdf')

# Another F distribution for the appendix ========
# F distribution
xmin=0
xmax=10

dfn=[5,10,20,50,100]   # degrees of freedom of numerator
dfd=dfn   # degrees of freedom of denominator
fig,ax=plt.subplots(figsize=(8,6))
colors=['silver','darkgrey','gray','dimgray','black']
x=np.linspace(xmin,xmax,1000)
for i in range(len(dfd)):
    y=f.pdf(x,dfn[i],dfd[i])/max(f.pdf(x,dfn[i],dfd[i]))
    plt.plot(x,y,linewidth=2,color=colors[i],label='F distr. ($f_1=%d$, $f_2$=%d)'%(dfd[i],dfn[i]))
ax.set_xlabel('F')
ax.set_ylabel('Normalized Prob. Distr.')
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.025))
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
plt.grid(which='both')
ax.set_xlim(0,5)
ax.set_ylim(0,1)
plt.legend(loc=1,prop={'size':12})
plt.savefig('fAppendix.pdf')

# ============================================
'''
# use the 10 measurements for chi2 and F tests
print('Problem 9.5')

x=np.array([10,12,15,11,13,16,12,10,18,13])

x1=x[0:5]
x2=x[5:10]
mu=13
yMod=mu*np.ones(5)
sigma=2**0.5
yErr=sigma*np.ones(5)
chi21=chisquared(x1,yErr,yMod)
chi22=chisquared(x2,yErr,yMod)
chi2Crit=chi2.ppf(0.95,5)
N=5
print(x1,x2)
print('chi2: %3.3f, %3.3f (crit value 0.95: %3.3f'%(chi21,chi22,chi2Crit))
F=chi22/chi21
fCrit=f.ppf(0.95,N,N)
fPValue=f.sf(F,N,N)
print('F: %3.3f (crit value 0.95: %3.3f)'%(F,fCrit))
print('F: p-value %3.3f'%fPValue)
# now with sample means
m1=np.mean(x1)
m2=np.mean(x2)
s21=np.var(x1,ddof=1)
s22=np.var(x2,ddof=1)
F=s22/s21
print('means of two 5-point segments: %3.3f, %3.3f, variances: %3.3f, %3.3f, S2: %3.3f %3.3f, F: %3.3f'
        %(m1,m2,s21,s22,s21*(N-1),s22*(N-1),F))
fCrit=f.ppf(0.95,N-1,N-1)
fPValue=f.sf(F,N-1,N-1)
print('F: %3.3f (crit value 0.95: %3.3f)'%(F,fCrit))
print('F: p-value %3.3f'%fPValue)
# ==== z-score of first five measurements
N=5
z1=(m1-mu)/(sigma/N**0.5)
p1=norm.sf(abs(z1),loc=0,scale=1)*2
print('zscore of 5 measurements: %3.3f, pvalue: %3.3f'%(z1,p1))
# ==== t stat for the 5 measurements
tStat1=(m1-mu)/(s21/N)**0.5
tPvalue=t.sf(abs(tStat1),N-1) # one-sided prob.
tPvalue=tPvalue*2
print('t statistic: %3.3f, 2-sided p-value: %3.3f'%(tStat1,tPvalue))
# ==== t stat for the 10 measurements
s2=(s21*(N-1)+s22*(N-1))/(N+N-2)
tStatComp=(m1-m2)/s2**0.5/(1/N+1/N)**0.5
print('s2= %3.3f, t stat for comparison of two sample means: %3.3f'%(s2,tStatComp))
tPValueComp=2*t.sf(abs(tStatComp),2*(N-1))
print('p-value for the comparison: %3.3f'%tPValueComp)

'''
quit()
#=================================================
# ==== Student t distribution

xmin=-5
xmax=5
N=5
f=N-1
x=np.linspace(xmin,xmax,1000)
y=t.pdf(x,f)

fig,ax=plt.subplots(figsize=(8,6))

ax.plot(x,y,linewidth=2,color='black',label='t distr. (f=4)')
ax.set_xlabel('t')
ax.set_ylabel('Probability Distribution')
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.025))
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
plt.grid(which='both')
ax.set_xlim(-4,4.2)
ax.set_ylim(0,0.42)

# add the 95% confidence limits
p1=0.95 # probability
l1=p1+0.5*(1-p1) # upper bound of SF 
l2=0.5*(1-p1)   # lower bound of SF
x1=t.ppf(l1,f)
x2=t.ppf(l2,f)
x1range=[i for i in np.arange(1000) if x[i]>=x1]
x2range=[i for i in np.arange(1000) if x[i]<=x2]
ax.fill_between(x[x1range], y[x1range],facecolor='none',hatch='XXXX',edgecolor="b",label='95% rejection region')
ax.fill_between(x[x2range], y[x2range],facecolor='none',hatch='XXXX',edgecolor="b")
# repeat the same for 68.3% rejection region
p1=0.683 # probability
l1=p1+0.5*(1-p1) # upper bound of SF
l2=0.5*(1-p1)   # lower bound of SF
x1=t.ppf(l1,f)
x2=t.ppf(l2,f)
x1range=[i for i in np.arange(1000) if x[i]>=x1]
x2range=[i for i in np.arange(1000) if x[i]<=x2]
ax.fill_between(x[x1range], y[x1range],facecolor='none',hatch='//',edgecolor="gray",label='68.3% rejection region')
ax.fill_between(x[x2range], y[x2range],facecolor='none',hatch='//',edgecolor="gray")

# add a N(0,1) for comparison
y=norm.pdf(x)
ax.plot(x,y,linewidth=2,linestyle='--',color='black',label='N(0,1)')

# add t(1), also known as Cauchy distribution
y=t.pdf(x,1)
ax.plot(x,y,color='red',linewidth=2,label='t distr. (f=1) - Cauchy')

handles,labels = ax.get_legend_handles_labels()
handles = [handles[0], handles[3], handles[4],handles[1],handles[2]]
labels = [labels[0], labels[3], labels[4],labels[1],labels[2]]
plt.legend(handles,labels,loc=1,prop={'size':12})
plt.savefig('t.pdf')

# -- Add another table of Tcrit, with $p$ level fixed
pLev=[0.683,0.9,0.95,0.99] # this needs to be a central CI
fN=[1,2,5,10,20,50,100] # number of degrees of freedom
for i in range(len(fN)):
    print('%d'%fN[i])
    for j in range(len(pLev)):
        tcrit=t.ppf((1+pLev[j])/2,fN[i])
        normcrit=norm.ppf((1+pLev[j])/2)
        print("%3.3f %3.3f"%(tcrit, normcrit))

# ===== Add another t distribution for the appendix

xmin=-5
xmax=5
f=[1,5,10,30,100]
colors=['red','darkgrey','gray','dimgray','black']
x=np.linspace(xmin,xmax,1000)

fig,ax=plt.subplots(figsize=(8,6))
for i in range(len(f)):
    y=t.pdf(x,f[i])#/max(t.pdf(x,f[i]))
    ax.plot(x,y,linewidth=2,color=colors[i],label='t distr. (f=%d)'%f[i])
y=norm.pdf(x)#/max(norm.pdf(x))
ax.plot(x,y,linewidth=2,linestyle='--',color='black',label='N(0,1)')
ax.set_xlabel('t')
ax.set_ylabel('Probability Distribution')
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.025))
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
plt.grid(which='both')
plt.grid(which='major',color='black')
ax.set_xlim(-4,4.2)
ax.set_ylim(0,0.4)
plt.legend(loc=1,prop={'size':12})
plt.savefig('tAppendix.pdf')
