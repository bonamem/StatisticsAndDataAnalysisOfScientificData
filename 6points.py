exec(open('imports.py').read())

dataTypes=[('number','I'),('x','f'),('y','f')]
data = np.genfromtxt('6points.dat',skip_header=1,dtype=dataTypes)
print(data)
x=data['x']
y=data['y']
yerr=np.ones(6) # this is to day they all have same error

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
plt.savefig('6points.pdf')


fig,ax=plt.subplots(figsize=(8,6))
quit()
# below is unused as of now =============================
# 3. Now onto 2-D confidence contours 
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
levels=[1.0,2.7,4.6]
CS=ax.contour(a,b,chi2Surface,levels,linewidth=2,colors=['red','blue','black'])
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

# F-test for the additional linear component
print('Fit to a constant model')
popt, pcov = curve_fit(constant, x,y, sigma=yerr,absolute_sigma=True)
print('curve_fit',popt,pcov)
print('a=%3.4f +- %3.4f'%(popt[0],pcov[0][0]**0.5))
ymodelC=constant(x,popt[0])
print(ymodelC)
chiminC=chisquared(y,yerr,ymodelC)
print('chi^2 = %3.3f'%chiminC)

# now use only the middle three measurements
xM=x[1:4]
yM=y[1:4]
yerrM=yerr[1:4]
print(xM,yM,yerrM)
# linear model for middle three datapoints
poptM, pcovM = curve_fit(linear, xM,yM, sigma=yerrM,absolute_sigma=True)
print('curve_fit M',poptM,pcovM)
print('a=%3.4f +- %3.4f'%(poptM[0],pcovM[0][0]**0.5))
ymodelM=linear(xM,poptM[0],poptM[1])
print(ymodelM)
chiminM=chisquared(yM,yerrM,ymodelM)
chiminMPval=chi2.sf(chiminM,1)
print('chi^2 = %3.3f (p-value %3.3f)'%(chiminM,chiminMPval))

# constant model for middle three datapoints
poptMC, pcovMC = curve_fit(constant, xM,yM, sigma=yerrM,absolute_sigma=True)
print('curve_fit M',poptMC,pcovMC)
print('a=%3.4f +- %3.4f'%(poptMC[0],pcovMC[0][0]**0.5))
ymodelMC=constant(xM,poptMC[0])
print(ymodelMC)
chiminMC=chisquared(yM,yerrM,ymodelMC)
chiminMCPval=chi2.sf(chiminMC,2)
print('chi^2 = %3.3f (p-value %3.3f)'%(chiminMC,chiminMCPval))

# ftest, for 1,1 d.o.f
nu1=1
nu2=3-2
DeltaChiM=chiminMC-chiminM
print('Delta chi2 = %3.3f'%DeltaChiM)
FM=(DeltaChiM/nu1)/(chiminM/nu2)
FPval=f.sf(FM,nu1,nu2)
print('F=%3.3f, p-value: %3.3f'%(FM,FPval))
