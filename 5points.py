exec(open('imports.py').read())

dataTypes=[('number','I'),('x','f'),('y','f')]
# 5 point data for prob11.2 
data = np.genfromtxt('5points.dat',skip_header=1,dtype=dataTypes)
print(data)
x=data['x']
y=data['y']
yerr=y**0.5

# ==================
print('Problem 11.2')
popt, pcov = curve_fit(linear, x,y, sigma=yerr,absolute_sigma=True)
print('curve_fit',popt,pcov)
print('a=%3.4f +- %3.4f, b= %3.6f +- %3.6f, sigma2_ab=%3.3f, r=%3.3f'
        %(popt[0],pcov[0][0]**0.5, popt[1],pcov[1][1]**0.5,pcov[1][0],pcov[1][0]/(pcov[0][0]*pcov[1][1])**0.5))

ymodel=linear(x,popt[0],popt[1])
#print(ymodel)
chimin=chisquared(y,yerr,ymodel)
pValue=chi2.sf(chimin,3)
print('chi^2=%3.2f (p value: %3.3f)'%(chimin,pValue))
print('critical value for chi2(3) at 0.9 and 0.99: %f, %f'%(chi2.ppf(0.9,3),chi2.ppf(0.99,3)))
print(y,ymodel,yerr)
# =================
#### intrinsic scatter for problem 17.5
print('Problem 17.5')
sigmaint=intrinsicScatter(y,ymodel,yerr,2)
print('intrinsic scatter: %3.3f'%sigmaint)

xmin=-1
xmax=5

xline=[xmin,xmax]
yline=np.zeros(2)
yline[0]=linear(xline[0],popt[0],popt[1])
yline[1]=linear(xline[1],popt[0],popt[1])

fig,ax=plt.subplots(figsize=(8,6))
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))

plt.errorbar(x,y,linewidth=2,yerr=yerr,linestyle='',fmt='o',capsize=4, fillstyle='none',label='Data',color='black')
plt.plot(xline,yline,linewidth=2,linestyle='--',label='Best-fit model',color='black')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(prop={'size': 12})
ax.set_xlim(xmin,xmax)
ax.set_ylim(10,100)

for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())


ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))
plt.grid(which='both')
plt.savefig('5points.pdf')
# ===========================================

# ===========================================
# 3. Now onto 2-D confidence contours 
# also problem 12.3
fig,ax=plt.subplots(figsize=(8,6))

Na=100
Nb=100
# search a range +-n sigma from the best-fit
n=3
amin=popt[0]-n*pcov[0][0]**0.5
amax=popt[0]+n*pcov[0][0]**0.5
bmin=popt[1]-n*pcov[1][1]**0.5
bmax=popt[1]+n*pcov[1][1]**0.5
a=np.linspace(amin,amax,Na)
b=np.linspace(bmin,bmax,Nb)
chi2Surface=np.zeros((Na,Nb))

for i in range(Na):
    for j in range(Nb):
        ymodelCurrent=linear(x,a[i],b[j])
        chi2Surface[i][j]=chisquared(y,yerr,ymodelCurrent)-chimin

#print(chi2Surface)
levels=[1.0,2.3,2.7,4.6]
CS=ax.contour(a,b,chi2Surface,levels,linewidth=2,colors=['red','green','blue','black'])
ax.clabel(CS,inline=True,fontsize=12,fmt='+%1.1f')
ax.set_xlabel('Parameter a')
ax.set_ylabel('Parameter b')
# adjust the view area to make room for label
plt.gcf().subplots_adjust(left=0.20)

ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(1))


plt.plot(popt[0],popt[1],linewidth=2,linestyle='',marker='+',color='black',label='Point of minimum $\chi^2$')
plt.grid(which='both')
plt.legend(prop={'size' : 12},loc=1)
ax.set_xlim(12,38)
ax.set_ylim(7,18)
plt.savefig('5pointsContours.pdf')
# ================================================

# Additional 2-d Contours with illustration of interesting parameters
fig,ax=plt.subplots(figsize=(8,6))
levels=[1.0,2.3,2.7,4.6]
CS=ax.contour(a,b,chi2Surface,levels,linewidth=2,colors=['red','green','blue','black'])
ax.clabel(CS,inline=True,fontsize=12,fmt='+%1.1f')
ax.set_xlabel('Parameter a')
ax.set_ylabel('Parameter b')
# adjust the view area to make room for label
plt.gcf().subplots_adjust(left=0.20)
print('68.3 pct crit value for 2 dof: %f'%(chi2.ppf(0.683,2)))
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(1))
plt.plot(popt[0],popt[1],linewidth=2,linestyle='',marker='+',color='black')
plt.grid(which='both')
# vertical line to illustrate marginalization over b parameter
aValue=27.4
plt.vlines(aValue,7,18,color='black',linestyle='--')
ax.set_xlim(12,38)
ax.set_ylim(7,18)

plt.vlines(popt[0]-pcov[0][0]**0.5,7,18,color='red',linestyle='-.')
plt.vlines(popt[0]+pcov[0][0]**0.5,7,18,color='red',linestyle='-.')
plt.vlines(aValue,7,18,color='black',linestyle='--')
ax.axvspan(popt[0]-pcov[0][0]**0.5,popt[0]+pcov[0][0]**0.5, alpha=0.5, color='red',
        label='68.3% confidence interval  on a')

#ax.axhspan(popt[1]-pcov[1][1]**0.5,popt[1]+pcov[1][1]**0.5, alpha=0.5, color='red',
#        label='p=0.68 confidenceinterval  on b')


plt.legend(prop={'size' : 12},loc=1)
plt.savefig('5pointsContours2.pdf')


######################

# F-test for the additional linear component (example 19.2)
print('Fit to a constant model')
popt, pcov = curve_fit(constant, x,y, sigma=yerr,absolute_sigma=True)
print('curve_fit',popt,pcov)
print('a=%3.4f +- %3.4f'%(popt[0],pcov[0][0]**0.5))
ymodelC=constant(x,popt[0])
print(ymodelC)
chiminC=chisquared(y,yerr,ymodelC)
print('chi^2 = %3.3f'%chiminC)

# now use only the middle three measurements
#iStart=1
#iEnd=4 # these are for example 19.2

# problem 19.3
print('Problem 19.3')

iStart=2
iEnd=5 # these are for prob19.3

xM=x[iStart:iEnd]
yM=y[iStart:iEnd]
yerrM=yerr[iStart:iEnd]
print('Using the  three measurements')
print(xM,yM,yerrM)
# linear model for middle three datapoints
poptM, pcovM = curve_fit(linear, xM,yM, sigma=yerrM,absolute_sigma=True)
print('Linear fit ',poptM,pcovM)
print('a=%3.4f +- %3.4f, b=%3.4f +- %3.4f'%(poptM[0],pcovM[0][0]**0.5,poptM[1],pcovM[1][1]**0.5))
ymodelM=linear(xM,poptM[0],poptM[1])
print(ymodelM)
chiminM=chisquared(yM,yerrM,ymodelM)
chiminMPval=chi2.sf(chiminM,1)
print('chi^2 = %3.3f (p-value %3.3f)'%(chiminM,chiminMPval))

# constant model for middle three datapoints
print('Constant fit')
poptMC, pcovMC = curve_fit(constant, xM,yM, sigma=yerrM,absolute_sigma=True)
print('curve_fit M',poptMC,pcovMC)
print('a=%3.4f +- %3.4f'%(poptMC[0],pcovMC[0][0]**0.5))
ymodelMC=constant(xM,poptMC[0])
print(ymodelMC)
chiminMC=chisquared(yM,yerrM,ymodelMC)
chiminMCPval=chi2.sf(chiminMC,2)
print('chi^2 = %3.3f (p-value %3.3f)'%(chiminMC,chiminMCPval))

# ftest for addition of slope to middle three points, for 1,1 d.o.f
print('F test for addition of slope parameter in three datapoints')
nu1=1
nu2=3-2
DeltaChiM=chiminMC-chiminM
print('Delta chi2 = %3.3f'%DeltaChiM)
FM=(DeltaChiM/nu1)/(chiminM/nu2)
FPval=f.sf(FM,nu1,nu2)
print('F=%3.3f, p-value: %3.3f'%(FM,FPval))

#################
#### bootstrap, prob20.6 (code from Hubble)
print('Problem 20.6, bootstrap of 5-point dataset')
NBoot=1000
N=len(x)
aBoot=np.zeros(NBoot)
bBoot=np.zeros(NBoot)
indices=np.arange(N)
print('Indices',indices,x,x[indices])
for i in range(NBoot):
    indicesChoice=choices(indices,k=N)
    # if all are the same index, need to re-draw b/c no fit is possible
    if(indicesChoice.count(indicesChoice[0]) == N):
        indicesChoice=choices(indices,k=N) # odds of two consecutive bad draws is ignored
        print('***')
    #print('%d: '%i,indicesChoice)
    bBoot[i],aBoot[i],r_value, p_value, std_err = stats.linregress(x[indicesChoice],y[indicesChoice])
    #print('%3.2f %3.2f'%(aBoot[i],bBoot[i]))
fig,ax=plt.subplots(figsize=(8,6))
aMin=0
aMax=40
bMin=0
bMax=18
binsa=np.linspace(aMin,aMax,200)
binsb=np.linspace(bMin,bMax,90)
#print('bins:',binsa,binsb)
plt.hist(aBoot,bins=binsa,linewidth=2,density=True,histtype='step',color='black',label='Parameter a')
plt.hist(bBoot,bins=binsb,linewidth=2,density=True,histtype='step',color='blue',label='Parameter b')
plt.xlabel('Parameter value')
plt.ylabel('Distribution')
plt.legend(prop={'size': 12})
plt.xlim((bMin,aMax))
plt.ylim((0,0.5))
ax.xaxis.set_major_locator(MultipleLocator(4))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(.05))
plt.grid(which='both')

# mean, median and percentiles
xaxis=np.linspace(bMin,aMax,2000) # shared by two distrib.
p=0.683
alpha=(1-p)/2
beta=(1+p)/2
print('Using percentiles %3.2f and %3.2f'%(alpha,beta))
#print('aBoot',aBoot)
aBootMean=np.mean(aBoot)
aBootMedian=np.median(aBoot)
erraBoot=np.std(aBoot)
alphaa=np.percentile(aBoot,alpha*100)
# do the best--fit gaussians from median and 68% C.I.
betaa=np.percentile(aBoot,beta*100)
yaxis=norm.pdf(xaxis,loc=aBootMedian,scale=(betaa-alphaa)/2)
plt.plot(xaxis,yaxis,linewidth=1,linestyle='--',color='black')

bBootMean=np.mean(bBoot)
bBootMedian=np.median(bBoot)
errbBoot=np.std(bBoot)
alphab=np.percentile(bBoot,alpha*100)
betab=np.percentile(bBoot,beta*100)
yaxis=norm.pdf(xaxis,loc=bBootMedian,scale=(betab-alphab)/2)
plt.plot(xaxis,yaxis,linewidth=1,linestyle='--',color='blue')
print('Bootstrap estimates a=%3.3f+-%3.3f, b=%3.3f+-%3.3f'
    %(aBootMean,erraBoot,bBootMean,errbBoot))
print('Bootstrap median and p=%3.3f central intervals:'%p)
print('a=%3.3f+-%3.3f (%3.3f-%3.3f), b=%3.3f+-%3.3f (%3.3f-%3.3f)'
    %(aBootMedian,(betaa-alphaa)/2,alphaa,betaa,
        bBootMedian,(betab-alphab)/2,alphab,betab))
plt.savefig('bootstrap5Point.pdf')

###### jackknife, prob20.7
print('Problem 20.7, jackknife on Hubble data')
aJackj=np.zeros(N)
bJackj=np.zeros(N)
aJackjStar=np.zeros(N)
bJackjStar=np.zeros(N)
xj=np.zeros((N,N-1)) # these are the resampled datasets, N sets of size N-1
yj=np.zeros((N,N-1))
# start with the estimates from the original data
bhat,ahat,r_value, p_value, std_err = stats.linregress(x,y)
for j in range(N):
    xj[j,:]=np.delete(x,j)
    yj[j,:]=np.delete(y,j)
    # these are the theta_j estimates from Zj
    bJackj[j],aJackj[j],r_value, p_value, std_err = stats.linregress(xj[j,:],yj[j,:])
    print(j, xj[j,:],yj[j,:],aJackj[j],bJackj[j])
    aJackjStar[j]=N*ahat-(N-1)*aJackj[j]
    bJackjStar[j]=N*bhat-(N-1)*bJackj[j]
meanaJackStar=np.mean(aJackjStar)
erraJackStar=np.std(aJackjStar,ddof=1)/N**0.5 # this is error on mean
meanbJackStar=np.mean(bJackjStar)
errbJackStar=np.std(bJackjStar,ddof=1)/N**0.5

print('Jackknife estimates: a=%3.3f +-%3.3f, b=%3.3f+-%3.3f'%(meanaJackStar,erraJackStar,meanbJackStar,errbJackStar))

