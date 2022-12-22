exec(open('imports.py').read())

# Simulate a normal distribution
fig,ax=plt.subplots(figsize=(8,6))

#############################################################
# Binomial distribution
fig,ax=plt.subplots(figsize=(8,6))
xmax=20
N,p=10,0.5
x=np.linspace(0,xmax,xmax+1,endpoint=True,dtype=int)
y=binom.pmf(x,N,p)
plt.plot(x,y,linestyle='--',marker='o',color='black',label='$N$=10, $p$=0.5')
# ----------- overplot a different p
N,p=10,0.10
x=np.linspace(0,xmax,xmax+1,endpoint=True,dtype=int)
y=binom.pmf(x,N,p)
plt.plot(x,y,linestyle='--',marker='o',color='blue',label='$N$=10, $p$=0.1')
# --------------------------------------------------
plt.legend()
plt.grid(which='both')
plt.xlim(0,10)
plt.xlabel('$n$')
#ax.xaxis.set_major_locator(MultipleLocator(1))
#ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.10))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
# linear tick mark labels
ax.xaxis.set_major_formatter(ScalarFormatter())
plt.ylabel('Probability $P_N(n)$')
plt.savefig('binomial.pdf')

# ============================================
# --------------------------------------------
#  Test the significance of the Astra-Zeneca and Pfizer tests
choice=3 #0-2 are AstraZeneca, 3 is Pfizer
choiceName=['AstraSD-SD','AstraAll','AstraLD-SD','Pfizer']
b=[27, 30, 3,8]     # no. of covid-19 cases for vaccinated
d=[71, 101, 30,162]    # no. of covid-10 cases for unvaccinated
n1=[4440, 5807, 1367,21720] # total vaccinated
n2=[4455, 5829, 1374,21728] #total un-vaccinated
mu=b[choice]+d[choice]
# vaccine efficacy formula, assuming same number in two samples
# --------------------------------------------
def ve(b,d,n1,n2):
	return 1.-(b/n1)/(d/n2) # if 2:1 ratio
# ------------------------------------
VE=ve(b[choice],d[choice],n1[choice],n2[choice])
print('Vaccine Efficacy (%s): %3.4f'%(choiceName[choice],VE))

pConf=0.95 # confidence probability, 95%
# 1. ---------------------------------
# endpoints of the range that contains 95% probability
# This is for the "observed" ratio, not the posterior on parent mean
# The observed ratio is b/mu2, sick and vaccinated
interval=binom.interval(pConf,mu,b[choice]/mu)
print('Interval containing 95pct. range of b=%d (%s);'%(b[choice],choiceName[choice]),interval)
VEInterval=np.zeros(2)
for i in range(2):
    VEInterval[i]=ve(interval[i],mu-interval[i],n1[choice],n2[choice])
print('corresponding VE range (from obs. ratio)',VEInterval)

# ======================================================
# 2. -----------------------------------
# Now also look at the posterior on parent value of beta/delta
# Lower and upper limit on b/mu, assuming mu is fixed

fig,ax=plt.subplots(figsize=(8,6))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.xaxis.set_major_locator(MultipleLocator(0.10))
ax.xaxis.set_minor_locator(MultipleLocator(0.025))

betaMu=np.linspace(0,1,100)
# cumulative distr. function for fixed b, d and the parent value of beta
cdfB=binom.cdf(b[choice],mu,betaMu)
sfB=binom.sf(b[choice],mu,betaMu)
ax.plot(betaMu,cdfB,color='black',linewidth=2,label='P($>$ %d out of %d)'%(b[choice],mu))
ax.plot(betaMu,sfB,color='blue',linewidth=2,label='P($\leq$ %d out of %d)'%(b[choice],mu))
ax.set_xlabel(r"Parent $\beta$/m")
ax.set_ylabel('Probability')
ax.grid(which='both')
plt.legend(loc=6)
plt.hlines(0.025,0,1,linestyle=':')
plt.hlines(0.975,0,1,linestyle=':')

#plt.hlines(0.0015,0,1,linestyle='--')
#plt.hlines(0.9985,0,1,linestyle='--')
ax.set_xlim(0,0.2)
# also convert beta/mu into vaccine efficacy
# the efficiency can become -infty, so be careful
#ax2=twiny()
#VE=ve(betaMu[0:50],1-betaMu[0:50])
#print(VE)

# look at the solution in the plot BY EYE and calculate VE
betaMu1= [0.20,0.17, 0.03,0.022] 
betaMu2= [0.375,0.31, 0.24,0.09] 
print('Posterior beta/m2: %3.3f-%3.3f, VE p=0.95 conf. interval: %3.3f-%3.3f'%(betaMu1[choice],betaMu2[choice],ve(betaMu1[choice],1-betaMu1[choice],n1[choice],n2[choice]),ve(betaMu2[choice],1-betaMu2[choice],n1[choice],n2[choice])))

plt.savefig('%s.pdf'%choiceName[choice])

# ==================================================
# --- add a check on whether two sub-samples
# --- are consistent with the general sample
m2=b[1]+d[1]
rho=b[1]/m2 # parent value of b/m2 

# Can the following two datasets come from a binomial with pi,
# assuming a fixed value of events from the marginals?
b1=b[0]
m21=b[0]+d[0]
b2=b[2]
m22=b[2]+d[2]
p=0.95
# this is done with the 95% intervals for m21 and m22
print('================================')
print('Using model from All: b=%d, m2=%d, rho=%3.3f'%(b[1],m2,rho))
print('Interval with rho=%3.3f and m=%d (observed: %d)'%(rho,m21,b1),binom.interval(p,m21,rho))
print('Interval with rho=%3.3f and m=%d (observed: %d)'%(rho,m22,b2),binom.interval(p,m22,rho))

# --- check the smaller subsample using the larger subsample
rho=b1/m21
print('================================')
print('Using model from SD/SD: b=%d, m2=%d, rho=%3.3f'%(b1,m21,rho))
print('Interval with rho=%3.3f and m=%d (observed: %d)'%(rho,m22,b2),binom.interval(0.95,m22,rho))



# ================================================
# ================================================
# Gaussian distribution
fig,ax=plt.subplots(figsize=(8,6))
xmin=-20
xmax=20
mu,sigma=5,(10*0.5*0.5)**0.5 # to match the first binomial
loc,scale=mu,sigma
# ------------------------------------------
x=np.linspace(xmin,xmax,10000,endpoint=True)
y=norm.pdf(x,loc,scale)
plt.plot(x,y,linestyle='-',linewidth=2,color='black',label='$\mu$=%2.1f, $\sigma^2$=%2.2f'%(mu,sigma))
# add vertical lines)
ax.vlines(mu,0,norm.pdf(mu,loc,scale),linestyle='-.',color='grey')
ax.vlines(mu+sigma,0,norm.pdf(mu+sigma,loc,scale),linestyle='--',color='grey')
ax.vlines(mu-sigma,0,norm.pdf(mu-sigma,loc,scale),linestyle='--',color='grey')

# ----------- overplot a different p
mu,sigma=0,1
loc,scale=mu,sigma
x=np.linspace(xmin,xmax,10000,endpoint=True)
y=norm.pdf(x,loc,scale)
plt.plot(x,y,linestyle='-',linewidth=2,color='blue',label='$\mu$=%1.0f, $\sigma^2$=%1.0f'%(mu,sigma))
# --------------------------------------------------
plt.legend()
plt.grid(which='both')
plt.xlim(-5,10)
ymin=-0.02
ymax=0.42
plt.ylim(ymin,ymax)
plt.xlabel('$x$')
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.10))
ax.yaxis.set_minor_locator(MultipleLocator(0.025))
# add vertical lines)
ax.vlines(0,0,norm.pdf(0,loc,scale),linestyle='-.',color='grey')
ax.vlines(1,0,norm.pdf(1,loc,scale),linestyle='--',color='grey')
ax.vlines(-1,0,norm.pdf(-1,loc,scale),linestyle='--',color='grey')
plt.ylabel('Distribution function $f(x)$')
plt.savefig('normal.pdf')

#calculate  FWHM ========================================
HWHM = (2*np.log(2))**0.5

peak=norm.pdf(0)
print('HWHM = %2.3f, FWHM= %2.3f'%(HWHM,2*HWHM))
print('Prob. in FWHM = %3.4f'%(1-2*norm.sf(HWHM)))
for i in range(5):
    sigma=i+1
    print('prob between +- %d: %5.6f'%(sigma,1-2*norm.sf(sigma)))
# ======================================================

# Add another Figure of gaussian for Appendix
fig,ax=plt.subplots(figsize=(8,6))
mu,sigma=0,1
loc,scale=mu,sigma
x=np.linspace(xmin,xmax,10000,endpoint=True)
y=norm.pdf(x,loc,scale)/norm.pdf(0,loc,scale) # normalized to peak value at x=0
plt.plot(x,y,linestyle='-',linewidth=2,color='black',label='Normalized distr. function')
y=norm.cdf(x,loc,scale)
plt.plot(x,y,linestyle='-',linewidth=2,color='red',label='Cumulative distr. function')
# --------------------------------------------------
plt.legend(loc=2,prop={'size': 12})
#plt.grid(which='both')
plt.grid(which='major',color='black')
plt.grid(which='minor')
plt.xlim(-5,5)
plt.ylim((0,1.0))
plt.xlabel('$z$')
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.10))
ax.yaxis.set_minor_locator(MultipleLocator(0.025))
plt.ylabel('Distributions')
plt.savefig('normalAppendix.pdf')

# ==========================================================================

#================================================
# === Poisson distribution
fig,ax=plt.subplots(figsize=(8,6))
xmax=20
mu=5
x=np.linspace(0,xmax,xmax+1,endpoint=True,dtype=int)
y=poisson.pmf(x,mu)
plt.plot(x,y,linestyle='--',marker='o',color='black',label='$\mu$=%2.1f'%mu)
# ----------- overplot a different p
mu=1
x=np.linspace(0,xmax,xmax+1,endpoint=True,dtype=int)
y=poisson.pmf(x,mu)
plt.plot(x,y,linestyle='--',marker='o',color='blue',label='$\mu$=%2.1f'%mu)
# --------------------------------------------------
plt.legend()
plt.grid(which='both')
plt.xlim(0,10)
plt.xlabel('$n$')
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.10))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
plt.ylabel('Probability $P(n)$')
plt.savefig('poisson.pdf')

# ===============================================
# Comparison of the three distributions
# Binomial distribution
fig,ax=plt.subplots(figsize=(8,6))
xmax=20
# -------- binomial ---------------------------------
# --------------------------------------------------
N,p=10,0.5 # mean of 5
x=np.linspace(0,xmax,xmax+1,endpoint=True,dtype=int)
y=binom.pmf(x,N,p)
plt.plot(x,y,linestyle='--',marker='o',color='black',label='Binom(%d,%2.1f)'%(N,p))
print('binom: mean=%3.2f, variance=%3.2f'%(N*p,N*p*(1-p)))
# ----------- overplot a different p
N,p=10,0.10 # mean of 1
x=np.linspace(0,xmax,xmax+1,endpoint=True,dtype=int)
y=binom.pmf(x,N,p)
plt.plot(x,y,linestyle='--',marker='o',color='blue',label='Binom(%d,%2.1f)'%(N,p))
print('binom: mean=%3.2f, variance=%3.2f'%(N*p,N*p*(1-p)))
# --------------------------------------------------
# ------------------------------------------------
# ---------- gaussian ----------------------------
xmin=-20
xmax=20
mu,sigma=5,(10*0.5*0.5)**0.5 # to match the first binomial
loc,scale=mu,sigma
x=np.linspace(xmin,xmax,10000,endpoint=True)
y=norm.pdf(x,loc,scale)
plt.plot(x,y,linestyle='-',linewidth=2,color='black',label='N(%2.1f,%2.1f)'%(mu,sigma**2))
# ----------- overplot a different gaussian
mu,sigma=1,1
loc,scale=mu,sigma
x=np.linspace(xmin,xmax,10000,endpoint=True)
y=norm.pdf(x,loc,scale)
plt.plot(x,y,linestyle='-',linewidth=2,color='blue',label='N(%2.1f,%2.1f)'%(mu,sigma**2))
# --------------------------------------------------
# --------------------------------------------------
# -------------- Poisson ---------------------------
xmax=20
mu=5
x=np.linspace(0,xmax,xmax+1,endpoint=True,dtype=int)
y=poisson.pmf(x,mu)
plt.plot(x,y,linestyle=':',marker='o',fillstyle='none',color='black',label='Poiss(%2.1f)'%mu)
# ----------- overplot a different mu
mu=1
x=np.linspace(0,xmax,xmax+1,endpoint=True,dtype=int)
y=poisson.pmf(x,mu)
plt.plot(x,y,linestyle=':',marker='o',fillstyle='none',color='blue',label='Poiss(%2.1f)'%mu)
# ------------------------------------------------

plt.legend(loc=1,prop={'size': 12})
plt.grid(which='both')
plt.xlim(-2,12)
plt.xlabel('$x$ or $n$')
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.10))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
plt.ylabel('Probability function')
plt.savefig('comparison.pdf')

# ============================================================
# Quantify difference between Poisson and Gaussian of same mean
fig,ax=plt.subplots(figsize=(8,6))
N=100
mu=np.arange(1,N+1)
avgFracDiff=np.zeros((3,N))
for m in range(N):
    # find integer closest to mean, to use for Poisson distribution
    G=norm.pdf(m,loc=mu[m],scale=mu[m]**0.5)
    P=poisson.pmf(m,mu[m])
    print('ind=',m, 'mu[m]=',mu[m], 'G=',G,'P=', P)#, 'P ,',P, 'G ,',G)
    print('for mu=%3.3f, G=%3.3f, for ind=%d, P=%3.3f'%(mu[m],G,m,P))
    avgFracDiff[0,m]=np.mean((P-G)/G) # percent, using one point
    # now another comparison, by averaging for values within 3 sigma
#    if (m>=3):
#        rangem=np.arange(m-3,m+3,7) # if m>=3
#        G=norm.pdf(rangem,loc=mu[m],scale=mu[m]**0.5)
#        P=poisson.pmf(rangem,mu[m])
#        avgFracDiff[1,m]=np.mean((P-G)/G) # percent
    # now another comparison, by averaging all values up to 3 sigma
    #indMin=max(0,mu[m]-3*mu[m]**0.5)
    #indMax=mu[m]+3*mu[m]**0.5
    #print(int(indMin),int(indMax)) # this should cover approx the +-3 sigma range
    #murangePlus=np.arange(int(indMin),int(indMax))
    #G=norm.pdf(murangePlus,loc=mu[m],scale=mu[m]**0.5)
    #P=poisson.pmf(murangePlus,mu[m])
    #avgFracDiff[2,m]=100*np.mean((G-P)/P) # percent
    # or maybe calculate 90% confidence intervals from the two distributions

# these are the 5th and 95th percentiles; the medians too
# They are normalized by the mean - is it needed?
G50=norm.ppf(0.50,loc=mu,scale=mu**0.5) # median
G95=norm.ppf(0.95,loc=mu,scale=mu**0.5) #- G50   # upper error
G05=norm.ppf(0.05,loc=mu,scale=mu**0.5) #- G50   # lower error

P50=poisson.ppf(0.5,mu) # median
P95=poisson.ppf(0.95,mu)# - P50 # upper error
P05=poisson.ppf(0.05,mu)# - P50 # lower error

# print confidence intervals
for i in range(100):
    #i=2
    print("%d: mu=%2.1f, G: %3.3f - %3.3f - %3.3f; P: %3.3f - %3.3f - %3.3f"%
            (i,mu[i],G05[i],G50[i],G95[i],P05[i],P50[i],P95[i]))

P50Ratio=(P50-mu)/mu
P95Ratio=(P95-mu)/mu
P05Ratio=(P05-mu)/mu

G50Ratio=(G50-mu)/mu
G95Ratio=(G95-mu)/mu
G05Ratio=(G05-mu)/mu

ax.yaxis.set_major_locator(MultipleLocator(0.20))
ax.yaxis.set_minor_locator(MultipleLocator(0.10))
ax.xaxis.set_major_locator(LogLocator())
ax.xaxis.set_minor_locator(LogLocator())

ax2=ax.twinx()
#ax.plot(mu[3:N],avgFracDiff[1,3:N],linewidth=2,color='blue',label='for x=$\mu - 3, \ldots, \mu+3$')
#ax.plot(mu,avgFracDiff[2],linewidth=2,color='red',label='for $x=\mu - 3 \sigma, \ldots, \mu+3 \sigma$')
print((P05-G05)/G05,P05,G05)
ax2.xaxis.set_major_locator(LogLocator())
ax2.xaxis.set_minor_locator(LogLocator())
ax2.yaxis.set_major_locator(MultipleLocator(0.20))
ax2.yaxis.set_minor_locator(MultipleLocator(0.10))

#ax2.plot(mu,P50Ratio,linewidth=2,color='red',label='test')
ax2.plot(mu,P95Ratio,linewidth=1,color='black')
ax2.plot(mu,P05Ratio,linewidth=1,color='black')

#ax2.plot(mu,G50Ratio,linewidth=2,color='black',linestyle='--')
ax2.plot(mu,G95Ratio,linewidth=2,color='black',linestyle='--',label='N($\mu$,$\mu^2$) Confidence Interval')
ax2.plot(mu,G05Ratio,linewidth=2,color='black',linestyle='--')
#ax.xaxis.set_major_locator(MultipleLocator(1))
#ax.xaxis.set_minor_locator(MultipleLocator(1))
ax2.fill_between(mu,P95Ratio,P05Ratio,facecolor='red',alpha=1.0,label='Poiss($\mu$) Confidence Interval')

ax.plot(mu,avgFracDiff[0],linewidth=3,color='black',label='Peak probability ratio')
ax.hlines(0,1,100,linestyle=':',linewidth=2,color='black')
ax.legend(loc=4,prop={'size': 12})
ax.set_xscale('log')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xlabel('$\mu$')
ax.set_ylabel('Poiss($\mu$)/N($\mu$,$\mu^2$)- 1')

ax2.set_ylabel('Normalized Confidence Interval')
l=ax2.legend(loc=1,prop={'size': 12})
#l.set_zorder(10)
ax.set_ylim(-1.10,1.30)
ax2.set_ylim(-1.10,1.30)
ax.set_xlim(1,100)
plt.tight_layout()
#ax.grid(axis='x',which='both')
ax.set_zorder(ax2.get_zorder()+1)
ax.patch.set_visible(False)
plt.savefig('difference.pdf')

# compare Stirling's approximation to exact factorial
for i in np.arange(30):
     print('n=%d, (S-n!)/n! = %3.4f'%(i,(Stirling(i)-math.factorial(i))/math.factorial(i)))

