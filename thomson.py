# This program calculates means and std. dev. from Thomson data
# Making use of the two tabular datasets
# M. Bonamente, 2020
exec(open('imports.py').read())


# =====================================================
# error propagation formula for m/e(I,W/Q)
# it returns sigma^2/mean^2
def errPropme(I,sigmaI,wq,sigmawq):
    result = 4*(sigmaI/I)**2+(sigmawq/wq)**2
    return result
# error propagation formula for v(I,W/Q); include covariance
def errPropv(I,sigmaI,wq,sigmawq,sigma2Iwq):
    result = (sigmaI/I)**2+(sigmawq/wq)**2 -2*sigma2Iwq/(I*wq)
    return result
# ====================================================


# Read in the two tables =============================
dataTypes=[('gas','S3'),('wq','f'),('I','f'),('me','f'),('me*','f'),('v','f'),('v*','f')]
table1 = np.genfromtxt('thomson1.dat',skip_header=1,dtype=dataTypes)
#print(table1)
table2 = np.genfromtxt('thomson2.dat',skip_header=1,dtype=dataTypes)
#print(table2)

# Generate lists of relevant measurements
# Table1
table1Gas=table1['gas']
table1wq=table1['wq']
table1I=table1['I']
table1me=table1['me']
table1v=table1['v']
#Table2
table2Gas=table2['gas']
table2wq=table2['wq']
table2I=table2['I']
table2me=table2['me']
table2v=table2['v']
# =====================================================

# Do the statistics for the tables
#1. Means and standard deviations ======================
# NOTE: python calculates standard deviations dividing by 1/N, not 1/(N-1)
# the ddof parameters must be specified to calculate the sample standard dev.
table1Meanwq=np.mean(table1wq)
table2Meanwq=np.mean(table2wq)
table1MeanI=np.mean(table1I)
table2MeanI=np.mean(table2I)

table1Stdwq=np.std(table1wq,ddof=1)
table2Stdwq=np.std(table2wq,ddof=1)
table1StdI=np.std(table1I,ddof=1)
table2StdI=np.std(table2I,ddof=1)

# problem 5.2
print('For problem 5.2')
print('================================')
print('Table 1: %d measurements; table 2: %d measurements'%(len(table1wq),len(table2wq)))
print('Table1: w/q = %f +- %f, I = %f +- %f (mean and std. dev.)'%(table1Meanwq,table1Stdwq,table1MeanI,table1StdI))
print('Table2: w/q = %f +- %f, I = %f +- %f (mean and std. dev.)'%(table2Meanwq,table2Stdwq,table2MeanI,table2StdI))

# 2. Covariance and correlations coeff. between w/q and I
table1CovwqI=np.cov(table1wq,table1I,ddof=1)   # this is the 2x2 covariance matrix
table1CorrwqI=np.corrcoef(table1wq,table1I)     # this is Pearson's r coeff.

table2CovwqI=np.cov(table2wq,table2I,ddof=1)   # this is the 2x2 covariance matrix
table2CorrwqI=np.corrcoef(table2wq,table2I)     # this is Pearson's r coeff.

#print(table1CovwqI[0,1],table1CorrwqI[0,1])
print('Table1: Cov(wq,I)=%f, r=%f'%(table1CovwqI[0,1],table1CorrwqI[0,1]))
print('Table2: Cov(wq,I)=%f, r=%f'%(table2CovwqI[0,1],table2CorrwqI[0,1]))
print('================================')

# problem 5.2
# 3. Sample mean and std. dev. for m/e for Tube 1
table1Meanme=np.mean(table1me)
table1Stdme=np.std(table1me,ddof=1)
print('Table 1 m/e',table1me)
print('Table1: m/e = %f +- %f (+- %3.3f for mean), 90 C.I.: %3.3f - %3.3f'%(table1Meanme,table1Stdme,table1Stdme/len(table1me)**0.5,table1Meanme-1.65*table1Stdme,table1Meanme+1.65*table1Stdme))
print('Table1: Std. dev. of m/e from error prop. (assuming no correlation between I and W/Q)')
print('sigma m/e = %f'%(table1Meanme**2*errPropme(table1MeanI,table1StdI,table1Meanwq,table1Stdwq))**0.5)

# also add m/e for Tube 2 (example in Ch. 7 of 3rd Ed.)
table2Meanme=np.mean(table2me)
table2Stdme=np.std(table2me,ddof=1)
# problem 7.5
print('Problem 7.5')
print('Table 2 m/e',table2me)
print('Table2: Mean m/e = %f +- %f (+- %3.3f for mean), 90 C.I.: %3.3f - %3.3f'%(table2Meanme,table2Stdme,table2Stdme/len(table2me)**0.5,table2Meanme-1.65*table2Stdme,table2Meanme+1.65*table2Stdme))

# also add upper limits and lower limits, for both sample and mean
print('Table1: m/e >= %3.4f (%3.4f for mean)'%(table1Meanme-1.28*table1Stdme,table1Meanme-1.28*table1Stdme/len(table1me)**0.5))
# problem 7.5 3rd ed.
print('Table2: m/e >= %3.4f (%3.4f for mean)'%(table2Meanme-1.28*table2Stdme,table2Meanme-1.28*table2Stdme/len(table2me)**0.5))
print('================================')

# problem 5.1
# 4. Sample mean and st. dev. for v for Tube 1
table1Meanv=np.mean(table1v)
table1Stdv=np.std(table1v,ddof=1)
print('For Problem 5.1')
print('Table1: Mean v = %3.2f +- %3.2f'%(table1Meanv,table1Stdv))
print('Table1: Std. dev. of v from error prop. (assuming no correlation between I and W/Q)')
print('sigma v = %f'%(table1Meanv**2*errPropv(table1MeanI,table1StdI,table1Meanwq,table1Stdwq,0))**0.5)
print('Table1: Std. dev. of v from error prop. (using sample covariance)')
print('Sample covariance between W/Q and I',table1CovwqI[0,1])
print('sigma v = %f'%(table1Meanv**2*errPropv(table1MeanI,table1StdI,table1Meanwq,table1Stdwq,table1CovwqI[0,1]))**0.5)

# ==============================================================================
# 4. problem 7.1, calculate confidence intervals and lower/upper limits on v
table1meanv=np.mean(table1v)
table1sigmav=np.std(table1v,ddof=1)
table2meanv=np.mean(table2v)
table2sigmav=np.std(table2v,ddof=1)
print('Problem 7.1')
print('Table 1: v=%4.2f +- %4.2f; Table 2: %4.2f +- %4.2f'%(table1meanv,table1sigmav,table2meanv,table2sigmav))
# ==== 90% confidence intervals are +-1.65 sigma
nSigma=1.65
print('90pct. confidence intervals on v - Table 1: %4.2f - %4.2f; Table 2: %4.2f - %4.2f'%(table1meanv-nSigma*table1sigmav,table1meanv+nSigma*table1sigmav,table2meanv-nSigma*table2sigmav,table2meanv+nSigma*table2sigmav))

# ==== 90% upper and lower limits are +-1.28 sigma
nSigmaL=1.28
print('90pct. confidence upper limit on v - Table 1: %4.2f; Table 2: %4.2f'%(table1meanv+nSigmaL*table1sigmav,table2meanv+nSigmaL*table2sigmav))
print('90pct. confidence lower limit on v - Table 1: %4.2f; Table 2: %4.2f'%(table1meanv-nSigmaL*table1sigmav,table2meanv-nSigmaL*table2sigmav))

# ============================================================================================
# 5. F statistic for measurements in Air for Tables 1 and 2
print('Problem 9.4')
table1Air=table1me[0:7] # first 7 measurements
print('m/e air measurements for table1:',len(table1Air),table1Air)
s12Air = np.var(table1Air,ddof=1)
m1Air= np.mean(table1me[0:6])
# turn table2 data from array to list, for easier concatenation
table2me=np.ndarray.tolist(table2me)
table2meAir=table2me[0:3]+table2me[4:5]+table2me[6:7]+table2me[9:11]
print('m/e air measurements for table2:', len(table2meAir), table2meAir)

s22Air = np.var(table2meAir,ddof=1)
m2Air= np.mean(table2meAir)
print('mean in air for two tables: %4.3f, %4.3f'%(m1Air,m2Air))
print('variances for air measurements of the two tables: %3.4f, %3.4f'%(s12Air,s22Air))
print('mesurements: Air 1 = %3.4f+-%3.4f, Air 2 = %3.4f+-%3.4f'%(m1Air,s12Air**0.5,m2Air,s22Air**0.5))
F=s12Air/s22Air
print('F=%3.2f'%F)
# calculate critical value
fCrit=f.ppf(0.9,6,6)
print('Critical value of f at 0.9 probability: %3.3f'%fCrit)

# ============================================
# ==== K-S test on m/e problem 19.1 and problem 19.2
print('problem 19.1 and 19.2')
print(table1me,table2me)
muM=0.57
sigmaM=0.1
xaxis=np.linspace(0,1,100)
modelG=norm.cdf(xaxis,muM,sigmaM)
fig,ax=plt.subplots(figsize=(8,6))
# check the 'align' so that it is a CDF

# 0 and 1 are the limits along the x-axis, for nice step plot
a1,b1=sampleCDF(table1me,0,1)
a2,b2=sampleCDF(table2me,0,1)
print('a=',a1,'b=',b1)
for i in range(len(a1)):
	plt.vlines(a1[i],0,b1[i],linewidth=0.75,linestyle='--',color='blue')
for i in range(len(a2)):
	plt.vlines(a2[i],0,b2[i],linewidth=0.75,linestyle='--',color='red')
plt.plot(xaxis,modelG,linewidth=2,color='black',label='Model')
plt.step(a1,b1,where='post',linewidth=2,color='blue',label='$m/e$ (Tube 1)')
plt.step(a2,b2,where='post',linewidth=2,color='red',label='$m/e$ (Tube 2)')
# --------------------------------------------------------
plt.legend(prop={'size': 12})
plt.grid(which='both')
plt.xlim(0.2,0.9)
plt.ylim(0,1.05)
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
plt.xlabel('Variable $m/e$')
plt.ylabel('Cumulative distribution')
plt.savefig('meKS.pdf')
KSV,KSp=kstest(table1me,lambda x: norm.cdf(x, loc=muM, scale=sigmaM))
print('Result of KS test Table 1:',KSV,KSp)
KSV,KSp=kstest(table2me,lambda x: norm.cdf(x, loc=muM, scale=sigmaM))
print('Result of KS test Table 2:',KSV,KSp)

# two-sample KS test
KSV,KSp=ks_2samp(table1me,table2me)
print('Result of 2-sample KS test:',KSV,KSp)
