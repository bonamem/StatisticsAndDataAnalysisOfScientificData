exec(open('imports.py').read())

# Install BCES libraries with: pip3 install bces
# For some systems, it may be needed: pip install bces

import bces.bces

# Python code to do statistics from Pressure Ratio data
# Bonamente 2020

# Import the data from pressureRatio.dat
# ===============================================================
# Character>---->-------Dominant>-------Recessive>------fraction
dataTypes=[('num','I'),('Radius','f'),('RadiusEP','f'),('RadiusEM','f'),
        ('Energy1','f'),('Energy1EP','f'),('Energy1EM','f'),
        ('Energy2','f'),('Energy2EP','f'),('Energy2EM','f'),
        ('ratio','f'),('ratioEP','f'),('ratioEM','f')]

data = np.genfromtxt('pressureRatio.dat',skip_header=1,dtype=dataTypes)
#print(data)


# Extract data
radius=data['Radius']
ratio=data['ratio']
Energy1=data['Energy1']
Energy2=data['Energy2']
# Now make symmetrical errors ==================================
Energy1E=0.5*(data['Energy1EP']+data['Energy1EM'])
Energy2E=0.5*(data['Energy2EP']+data['Energy2EM'])
ratioE=0.5*(data['ratioEP']+data['ratioEM'])
radiusE=0.5*(data['RadiusEP']+data['RadiusEM'])
print('Ratio errors',ratioE)
print('Energy 2 errors',Energy2E)


# First, basic statistics such as mean, weighted mean and median
N=len(ratio)
ratioMean=np.mean(ratio)
ratioWMean=np.average(ratio,weights=1/ratioE**2)
ratioMedian=np.median(ratio)
ratioMeanE=(np.var(ratio,ddof=1)/(N-1))**0.5
ratioWMeanE=(1./np.sum(1/ratioE**2))**0.5
ratioMedianE=ratioMeanE*(np.pi/2)**0.5
####################################################
print('Problems 8.1 and 8.2')
print('Mean: %3.2f+-%3.3f, Weigh. Mean: %3.2f+-%3.3f, Median: %3.2f+-%3.3f'
        %(ratioMean,ratioMeanE,ratioWMean,ratioWMeanE,ratioMedian,ratioMedianE))
# add an additional error of 0.1 to ratio
ratioESys=ratioE+0.1
ratioWMeanSys=np.average(ratio,weights=1/ratioESys**2)
print('Weigh. Mean with 0.1 sys. error: %3.2f'%ratioWMeanSys)
print('%%%%%%%%%%%%%%%')
# also do a correlation coefficient between radius and ratio prob18.3
rRR=pearsonr(ratio,radius)
print('Pearson r for ratio vs radius: %3.3f p-value:%3.2e'%(rRR[0],rRR[1]))

# Also do a histogram plot of ratio, and logarithmic average
# This is unused for now
fig,ax=plt.subplots(figsize=(8,6))
plt.hist(ratio)
plt.savefig('histPressureRatio.pdf')
logRatio=np.log(ratio)
#problem 8.5
print('Problem 8.5')
# implement equations  8.7 and 8.8 of 3rd Ed.
meanlogRatio=np.average(logRatio,weights=(ratio/ratioE)**2)
meanlogRatioE=np.sum((ratio/ratioE)**(2))**(-0.5)
# then, to turn the logarithm into the linear value, need to expand
# the confidence interval
meanRatio=np.exp(meanlogRatio)
meanRatioEPlus=np.exp(meanlogRatio+meanlogRatioE)-meanRatio
meanRatioEMinus=meanRatio-np.exp(meanlogRatio-meanlogRatioE)
print('logarithmic average of ratio: %3.3f+-%3.3f , => %3.4f+%3.4f-%3.4f'
        %(meanlogRatio,meanlogRatioE,meanRatio,meanRatioEPlus,meanRatioEMinus))
# also for the ad hoc relative-error weighted mean
meanRatioRelativeErr=np.average(ratio,weights=(ratio/ratioE)**2)

print('relative error weighted average: %3.3f +- %3.3f'%(meanRatioRelativeErr,ratioWMeanE))
# problem 8.4 
# calculate statistics for all other quantities
print('Problem 8.4')
radiusMean=np.mean(radius)
radiusWMean=np.average(radius,weights=1/radiusE**2)
radiusMedian=np.median(radius)
radiusMeanE=(np.var(radius,ddof=1)/(N-1))**0.5
radiusWMeanE=(1./np.sum(1/radiusE**2))**0.5
radiusMedianE=radiusMeanE*(np.pi/2)**0.5
Energy1Mean=np.mean(Energy1)
Energy1WMean=np.average(Energy1,weights=1/Energy1E**2)
Energy1Median=np.median(Energy1)
Energy1MeanE=(np.var(Energy1,ddof=1)/(N-1))**0.5
Energy1WMeanE=(1./np.sum(1/Energy1E**2))**0.5
Energy1MedianE=Energy1MeanE*(np.pi/2)**0.5
Energy2Mean=np.mean(Energy2)
Energy2WMean=np.average(Energy2,weights=1/Energy2E**2)
Energy2Median=np.median(Energy2)
Energy2MeanE=(np.var(Energy2,ddof=1)/(N-1))**0.5
Energy2WMeanE=(1./np.sum(1/Energy2E**2))**0.5
Energy2MedianE=Energy2MeanE*(np.pi/2)**0.5
print('Ratio: Mean: %3.2f+-%3.3f, Weigh. Mean: %3.2f+-%3.3f, Median: %3.2f+-%3.3f'
        %(ratioMean,ratioMeanE,ratioWMean,ratioWMeanE,ratioMedian,ratioMedianE))
print('Radius: Mean: %3.2f+-%3.3f, Weigh. Mean: %3.2f+-%3.3f, Median: %3.2f+-%3.3f'
        %(radiusMean,radiusMeanE,radiusWMean,radiusWMeanE,radiusMedian,radiusMedianE))
print('Energy1: Mean: %3.2f+-%3.3f, Weigh. Mean: %3.2f+-%3.3f, Median: %3.2f+-%3.3f'
        %(Energy1Mean,Energy1MeanE,Energy1WMean,Energy1WMeanE,Energy1Median,Energy1MedianE))
print('Energy2: Mean: %3.2f+-%3.3f, Weigh. Mean: %3.2f+-%3.3f, Median: %3.2f+-%3.3f'
        %(Energy2Mean,Energy2MeanE,Energy2WMean,Energy2WMeanE,Energy2Median,Energy2MedianE))
# ----------------------------------
# problem 11.6
# 1. Fit data radius vs. ratio using symmetrical errors 
# NOTE: absolute_sigma needs to be set, or else errors are re-scaled .... :-(
popt, pcov = curve_fit(linear, radius, ratio, sigma=ratioE,absolute_sigma=True)
print('curve_fit',popt,pcov)
print('a=%3.4f +- %3.4f, b= %3.6f +- %3.6f'%(popt[0],pcov[0][0]**0.5, popt[1],pcov[1][1]**0.5))

ymodel=linear(radius,popt[0],popt[1])
#print(ymodel)
chimin=chisquared(ratio,ratioE,ymodel)
print('chi^2=%3.2f'%chimin)

# 1.5 Also try numpy polyfit
p,cov=np.polyfit(radius, ratio,1, w=1/ratioE,cov='unscaled')
print('polyfit:',p,cov)
print('a=%3.4f +- %3.4f, b= %3.6f +- %3.6f'%(p[1],cov[1][1]**0.5, p[0],cov[0][0]**0.5))
# This seems to give a result that is consistent with the Delta chi^2 surface

# ======== do fits in small ranges for F test 
iMin=[0,5,21]
iMax=[5,10,26]
chiMinRange=np.zeros(3)
for j in range(3):
	print('Range %d-%d'%(iMin[j],iMax[j]))
	x=radius[iMin[j]:iMax[j]]
	y=ratio[iMin[j]:iMax[j]]
	yerr=ratioE[iMin[j]:iMax[j]]
	#print(x,y,yerr)
	poptRange, pcovRange = curve_fit(linear, x,y, sigma=yerr,absolute_sigma=True)
	#print('curve_fit range',popt,pcov)
	print('a=%3.3f +- %3.3f, b= %3.5f +- %3.5f'%(poptRange[0],pcovRange[0][0]**0.5, poptRange[1],pcovRange[1][1]**0.5))
	ymodelRange=linear(x,poptRange[0],poptRange[1])
#print(ymodel)
	chiMinRange[j]=chisquared(y,yerr,ymodelRange)
	pValue=chi2.sf(chiMinRange[j],3)
	print('chi^2=%3.2f (pValue=%3.3e)'%(chiMinRange[j],pValue))

# add f-test values
F1=chiMinRange[1]/chiMinRange[0]
pValueF=f.sf(F1,3,3)
print('F-stat: %3.3f, p value=%3.3e'%(F1,pValueF))
F1=chiMinRange[0]/chiMinRange[2]
pValueF=f.sf(F1,3,3)
print('F-stat: %3.3f, p value=%3.3e'%(F1,pValueF))

# ===========================================
# 2. plot with best-fit line
xmin=0
xmax=700

xline=[xmin,xmax]
yline=np.zeros(2)
yline[0]=linear(xline[0],popt[0],popt[1])
yline[1]=linear(xline[1],popt[0],popt[1])

fig,ax=plt.subplots(figsize=(8,6))

plt.errorbar(radius, ratio,yerr=ratioE,linestyle='',fmt='o',label='data',color='black')
plt.plot(xline,yline,label='best-fit model',color='blue')
plt.xlabel('Radius')
plt.ylabel('Ratio')
plt.legend(prop={'size': 12})
plt.savefig('ratioPressure.pdf')
# =====================================

# ===================================
# do the intrinsic scatter for radius vs. ratio 
#problem17.1
print('Problem 17.1')
fig,ax=plt.subplots(figsize=(8,6))

sigmaint=intrinsicScatter(ratio,ymodel,ratioE,2)
print('intrinsic scatter radius vs. ratio: %3.3f'%sigmaint)

# also plot the fit with the intrinsic error added
fig,ax=plt.subplots(figsize=(8,6))
ratioENew=(ratioE**2+sigmaint**2)**0.5

# get a new best-fit model with the new error added in quadrature
popt, pcov = curve_fit(linear, radius, ratio, sigma=ratioENew,absolute_sigma=True)
print('curve_fit with sys error added',popt,pcov)
print('a=%3.4f +- %3.4f, b= %3.6f +- %3.6f'%(popt[0],pcov[0][0]**0.5, popt[1],pcov[1][1]**0.5))
print('------------------------')
xline=[xmin,xmax]
yline=np.zeros(2)
yline[0]=linear(xline[0],popt[0],popt[1])
yline[1]=linear(xline[1],popt[0],popt[1])

plt.errorbar(radius, ratio,yerr=ratioENew,linestyle='',fmt='o',label='data',color='black')
plt.plot(xline,yline,label='best-fit model',color='blue')
plt.xlabel('Radius')
plt.ylabel('Ratio')
plt.legend(prop={'size': 12})
plt.savefig('ratioPressureint.pdf')

# problem 17.2
print('Problem 17.2 - stepping through intrinsic error')
# also additional estimate of sigmaint from deltachi2=1
# do this numerically
sigmaint=np.linspace(0.12,0.22,100)
for i in range(100):
    ratioENew=(ratioE**2+sigmaint[i]**2)**0.5 # new error
    popt, pcov = curve_fit(linear, radius, ratio, sigma=ratioENew,absolute_sigma=True) # new fit
    ymodel=linear(radius,popt[0],popt[1]) # new model
    chimin=chisquared(ratio,ratioENew,ymodel) # new chi2
    print('sigmaint=%3.3f, chimin=%3.3f (red: %3.2f) a=%3.2f, b=%3.5f'
            %(sigmaint[i],chimin,chimin/23,popt[0],popt[1]))

# =================================================
# ========== add the BCES fits to ratio vs. radius
print('BCES fits, Radius vs. Ratio')
AA=np.zeros(len(Energy1)) 
corrFactor=2.0 # for problem 18.3
#corrFactor=1.0 # for problem 18.2
radiusE=radiusE*corrFactor
ratioE=ratioE*corrFactor
b,a,berr,aerr,covab=bces.bces.bces(radius,radiusE,ratio,ratioE,AA)
print('BCES fits radius vs ratio (problem 18.2 or 18.3)',a,b,aerr,berr,covab)
# a,b aerr, berr and covab are for Y/X, X/Y, bisect., orthogonal prob18.2
fitLabel=['Y/X regression','X/Y regression','Bisector','Orthogonal']
fitColor=['blue','red','grey']
fitStyle=['-','-','--']
xmin=0
xmax=800
ymin=0
ymax=2
fig,ax=plt.subplots(figsize=(8,6))
plt.xscale('linear')
plt.yscale('linear')
NPoints=100
xaxis=np.linspace(xmin,xmax,100)
yaxis=np.zeros((4,100))
for i in range(3): # skip the orthogonal fit
        for j in range(NPoints):
                yaxis[i,j]=a[i]+xaxis[j]*b[i]
                yaxis[i,j]=a[i]+xaxis[j]*b[i]
        plt.plot(xaxis,yaxis[i],linestyle=fitStyle[i],linewidth=2,color=fitColor[i],label=fitLabel[i])

for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())

plt.errorbar(radius,ratio,yerr=ratioE,xerr=radiusE,capsize=2,linestyle='',color='black')
print(xaxis[0],yaxis[0,0],xaxis[1],yaxis[0,1])
plt.grid(which='both')
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.xlabel('Radius')
plt.ylabel('Ratio')
plt.legend(prop={'size' : 12})
# choose the one that applies
#plt.savefig('BCESRatio.pdf') # problem 18.2
plt.savefig('BCESRatioTwice.pdf') # problem 18.3

#==================
# add a bivariate chi2 minimization prob18.4
print('Problem 18.4 - numerical minimization of bivariate chi^2')
ab=minimize(chisquaredBivLin,x0=(1,0.4),args=(radius,ratio,radiusE,ratioE))
print('Results of bivariate chi2 minimization:',ab)
chi2Biv=chisquaredBivLin(ab.x,radius,ratio,radiusE,ratioE)
print('Results of bivariate chi2 minimization:',chi2Biv)
fig,ax=plt.subplots(figsize=(8,6))
a,b=ab.x
print('Using best-fit values',a,b)
xaxis=[0,800]
yaxis=np.zeros(2)
for i in range(2):
    yaxis[i]=a+xaxis[i]*b
plt.errorbar(radius,ratio,yerr=ratioE,xerr=radiusE,capsize=2,linestyle='',color='black')
plt.plot(xaxis,yaxis,color='red',linewidth=2,linestyle='--')
plt.grid(which='both')
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.xlabel('Radius')
plt.ylabel('Ratio')
plt.savefig('BCESRatioBiv.pdf')

#====================
# 3. Now onto 2-D confidence contours (problem 12.6)
fig,ax=plt.subplots(figsize=(8,6))
# repeat the same fit as for 11.6
# --
popt, pcov = curve_fit(linear, radius, ratio, sigma=ratioE,absolute_sigma=True)
print('curve_fit',popt,pcov)
print('a=%3.4f +- %3.4f, b= %3.6f +- %3.6f'%(popt[0],pcov[0][0]**0.5, popt[1],pcov[1][1]**0.5))
ymodel=linear(radius,popt[0],popt[1])
chimin=chisquared(ratio,ratioE,ymodel)
# --
Na=100
Nb=100
# search a range +-n sigma from the best-fit
amin=popt[0]-2*pcov[0][0]**0.5
amax=popt[0]+2*pcov[0][0]**0.5
bmin=popt[1]-2*pcov[1][1]**0.5
bmax=popt[1]+2*pcov[1][1]**0.5
print('Searching ranges %3.2e-%3.2e, %3.2e-%3.2e'%(amin,amax,bmin,bmax))
a=np.linspace(amin,amax,Na)
b=np.linspace(bmin,bmax,Nb)
chi2Surface=np.zeros((Na,Nb))

for i in range(Na):
    for j in range(Nb):
        ymodelCurrent=linear(radius,a[i],b[j])
        chi2Surface[i][j]=chisquared(ratio,ratioE,ymodelCurrent)-chimin

#print(chi2Surface)
levels=[1.0,2.7,4.6]
CS=ax.contour(a,b,chi2Surface,levels,colors=['red','blue','black'])
ax.clabel(CS,inline=True,fontsize=12,fmt='+%1.1f')
ax.set_xlabel('Parameter a')
ax.set_ylabel('Parameter b')
# adjust the view area to make room for label
plt.gcf().subplots_adjust(left=0.20)

plt.plot(popt[0],popt[1],linestyle='',marker='+',color='black')
plt.grid()
plt.savefig('ratioPressureContours.pdf')

# ======================================
# Do a fit for energy 1 vs. energy 2
popt, pcov = curve_fit(linear, Energy1, Energy2, sigma=Energy2E,absolute_sigma=True)
print('curve_fit',popt,pcov)
print('Energy 1 vs. Energy 2: a=%3.4f +- %3.4f, b= %3.6f +- %3.6f'
	%(popt[0],pcov[0][0]**0.5, popt[1],pcov[1][1]**0.5))

ymodel=linear(Energy1,popt[0],popt[1])
#print(ymodel)
chimin=chisquared(Energy2,Energy2E,ymodel)
chi2p=chi2.sf(chimin,23)
print('Energy 1 vs. Energy 2: chi^2=%3.2f (p-value: %e)'%(chimin,chi2p))

# ------ intrinsic scatter 
# 
sigmaint=intrinsicScatter(Energy2,ymodel,Energy2E,2)
print('E1 vs E2 sigmaint= %3.3f'%sigmaint)
#now also vary the errors to get reduced chi^2=1 
dof=len(Energy2)-2
for i in range(20):
	sigmaint=i/5 
	Energy2ENew=(Energy2E**2+sigmaint**2)**0.5
	popt, pcov = curve_fit(linear, Energy1, Energy2, sigma=Energy2ENew,absolute_sigma=True)
	ymodel=linear(Energy1,popt[0],popt[1])
	chimin=chisquared(Energy2,Energy2ENew,ymodel)
	print('for sigmaint=%3.3f: chi2=%3.3f (dof:%d)'%(sigmaint,chimin,dof))
# ------------------------
xmin=0
xmax=100

xline=[xmin,xmax]
yline=np.zeros(2)
yline[0]=linear(xline[0],popt[0],popt[1])
yline[1]=linear(xline[1],popt[0],popt[1])

ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))

fig,ax=plt.subplots(figsize=(8,6))
plt.errorbar(Energy1, Energy2,linewidth=2,yerr=Energy2E,linestyle='',fmt='o',capsize=4, fillstyle='none',label='Data',color='black')
plt.plot(xline,yline,linewidth=2,linestyle='--',label='Best-fit model',color='black')
plt.xlabel('Energy 1')
plt.ylabel('Energy 2')
plt.legend(prop={'size': 12})
ax.set_xlim(0.6,110)
ax.set_ylim(0.6,110)
ax.set_xscale('log')
ax.set_yscale('log')

for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())

plt.grid(which='both')
plt.savefig('Energies.pdf')

# =================================================
# also try energy 2 vs energy 1
popt, pcov = curve_fit(linear, Energy2, Energy1, sigma=Energy1E,absolute_sigma=True)
print('curve_fit',popt,pcov)
print('a=%3.4f +- %3.4f, b= %3.6f +- %3.6f'%(popt[0],pcov[0][0]**0.5, popt[1],pcov[1][1]**0.5))

ymodel=linear(Energy2,popt[0],popt[1])
#print(ymodel)
chimin=chisquared(Energy1,Energy1E,ymodel)
print('Energy 2 vs. Energy 1: chi^2=%3.2f'%chimin)

# ================================================
# =========== Now try BCES from Akritas & Bershady
# =========== as implemented in numpy ============
# pront problem 18.1, BCES fit energy 1 vs energy 2
print('================================================\n',
'The BCES routines (https://github.com/rsnemmen/BCES)\n',
       'return a 4-long list of the following quantities\n',
       'a,b : best-fit parameters a,b of the linear regression such that y = Ax + B.\n',
    'aerr,berr : the standard deviations in a,b\n',
    'covab : the covariance between a and b (e.g. for plotting confidence bands)\n',
    'Each value corresponds to the Y/X, X/Y, bisector and orthogonal (not used) fits\n')

print('BCES fit energy1 vs energy2 (Problem 18.1)')
AA=np.zeros(len(Energy1))
b,a,berr,aerr,covab=bces.bces.bces(Energy1,Energy1E,Energy2,Energy2E,AA)

print('BCES models for the Energy 1 vs Energy 2',a,b,aerr,berr,covab)
# a,b aerr, berr and covab are for Y/X, X/Y, bisect., orthogonal
fitLabel=['Y/X regression','X/Y regression','Bisector','Orthogonal']
fitColor=['blue','red','grey']
fitStyle=['-','-','--']
xmin=0.6
xmax=100
ymin=0.6
ymax=100
fig,ax=plt.subplots(figsize=(8,6))
plt.xscale('log')
plt.yscale('log')
NPoints=100
xaxis=np.logspace(np.log10(xmin),np.log10(xmax),100)
yaxis=np.zeros((4,100))
for i in range(3): # skip the orthogonal fit
	for j in range(NPoints):
		yaxis[i,j]=a[i]+xaxis[j]*b[i]
		yaxis[i,j]=a[i]+xaxis[j]*b[i]
	plt.plot(xaxis,yaxis[i],linestyle=fitStyle[i],linewidth=2,color=fitColor[i],label=fitLabel[i])

for axis in [ax.xaxis, ax.yaxis]:
	axis.set_major_formatter(ScalarFormatter())

plt.errorbar(Energy1,Energy2,yerr=Energy2E,xerr=Energy1E,capsize=2,linestyle='',color='black')
plt.scatter(xaxis[0],yaxis[0,0],marker='o')
plt.scatter(xaxis[1],yaxis[0,1],marker='o')
#print(xaxis[0],yaxis[0,0],xaxis[1],yaxis[0,1])
plt.grid(which='both')
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.xlabel('Energy 1')
plt.ylabel('Energy 2')
plt.legend(prop={'size' : 12})
plt.savefig('BCESE1vsE2.pdf')
