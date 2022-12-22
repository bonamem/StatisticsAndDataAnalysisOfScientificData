exec(open('imports.py').read())

dataTypes=[('number','I'),('x','f'),('y','f')]
# Do MCMC on the 5-point dataset
# ============================
data = np.genfromtxt('5points.dat',skip_header=1,dtype=dataTypes)
print(data)
x=data['x']
y=data['y']
yerr=y**0.5
n=len(x)

popt, pcov = curve_fit(linear, x,y, sigma=yerr,absolute_sigma=True)
print('curve_fit',popt,pcov)
print('a=%3.4f +- %3.4f, b= %3.6f +- %3.6f'%(popt[0],pcov[0][0]**0.5, popt[1],pcov[1][1]**0.5))

# Do three chains: with widths 1, 10 and 50
NMCMC=1000 # 1000000
# proposal distribution widths
DeltaaAux=[1,10,50]
DeltabAux=DeltaaAux
nAux=len(DeltaaAux)	# number of chains with difference proposal distr.
# ============================
aMCMC=np.zeros((nAux,NMCMC))
bMCMC=np.zeros((nAux,NMCMC))
aPrime=np.zeros(nAux)
bPrime=np.zeros(nAux)
# Initial values =============
aMCMC[:,0]=np.ones(nAux)*12
bMCMC[:,0]=np.ones(nAux)*6
# ============================
yMCurrent=np.zeros((nAux,n))
yMPrevious=np.zeros((nAux,n))
chi2Current=np.zeros(nAux)
chi2Previous=np.zeros(nAux)
alpha=np.zeros(nAux)
u=np.zeros(nAux)
NAccept=np.zeros(nAux)
fout=open('mcmc5pointsLong.txt','w')
# --- initial chi^2 -----------------------------------
for j in range(nAux):
	yMPrevious[j]=linear(x,aMCMC[j,0],bMCMC[j,0])
	chi2Previous[j]=chisquared(y,yerr,yMPrevious[j])
	fout.write('%3.5f  %3.5f  '%(aMCMC[j,0],bMCMC[j,0]))
fout.write('\n')
# -----------------------------------------------------

for i in range(1,NMCMC):
	for j in range(nAux):
		aPrime[j]=uniform.rvs(loc=aMCMC[j,i-1]-DeltaaAux[j]/2,scale=DeltaaAux[j],size=1) 
		bPrime[j]=uniform.rvs(loc=bMCMC[j,i-1]-DeltabAux[j]/2,scale=DeltabAux[j],size=1)
        # calculate the likelihood, aka chi^2
		yMCurrent[j]=linear(x,aPrime[j],bPrime[j])
		chi2Current[j]=chisquared(y,yerr,yMCurrent[j])
		alpha[j]=min(1,np.exp((chi2Previous[j]-chi2Current[j])/2))
		print('a=%3.3f, b=%3.3f, chi2Current=%3.4f, chi2Previous=%3.4f, alpha=%3.4f'
		%(aPrime[j],bPrime[j],chi2Previous[j],chi2Current[j],alpha[j]))
		u[j]=uniform.rvs(size=1)
        	# Decide whether to accept or reject
		if (alpha[j]>=u[j]):  # accept
			aMCMC[j,i]=aPrime[j]
			bMCMC[j,i]=bPrime[j]
			chi2Previous[j]=chi2Current[j]# this becomes the previous chi2
			print('Accepted')
			NAccept[j]=NAccept[j]+1
		if (alpha[j]<u[j]):
			aMCMC[j,i]=aMCMC[j,i-1]
			bMCMC[j,i]=bMCMC[j,i-1]
			print('Rejected')
		fout.write('%3.5f  %3.5f  '%(aMCMC[j,i],bMCMC[j,i]))
	fout.write('\n')
fout.close()
print('MCMC acceptance:',NAccept/NMCMC)
fig,ax=plt.subplots(1,2,figsize=(8,6))
color=['black','red','blue']
for j in range(nAux):
	ax[0].hist(aMCMC[j],bins=40,linewidth=2,histtype='step',color=color[j],
	label='$\Delta a$=%2.1f'%DeltaaAux[j])
	ax[1].hist(bMCMC[j],bins=40,linewidth=2,histtype='step',color=color[j],
	label='$\Delta b$=%2.1f'%DeltabAux[j])
ax[0].set_xlabel('Parameter a')
ax[0].xaxis.set_major_locator(MultipleLocator(10))
ax[0].xaxis.set_minor_locator(MultipleLocator(2))
ax[0].yaxis.set_major_locator(MultipleLocator(NMCMC*0.01))
ax[0].yaxis.set_minor_locator(MultipleLocator(NMCMC*0.005))
ax[1].set_xlabel('Parameter b')
ax[1].xaxis.set_major_locator(MultipleLocator(5))
ax[1].xaxis.set_minor_locator(MultipleLocator(1))
ax[1].yaxis.set_major_locator(MultipleLocator(NMCMC*0.01))
ax[1].yaxis.set_minor_locator(MultipleLocator(NMCMC*0.005))
ax[0].set_ylabel('Number of occurrence')
ax[0].set_ylim((0,0.09*NMCMC))
ax[1].set_ylim((0,0.09*NMCMC))
ax[0].grid(which='both')
ax[1].grid(which='both')
ax[0].legend(prop={'size': 10})
ax[1].legend(prop={'size': 10})
for axs in ax.flat:
    axs.label_outer()
plt.savefig('5PointMCMCHist.pdf')

xaxis=np.arange(NMCMC)
fig,ax=plt.subplots(2,1,figsize=(8,6))
for j in range(nAux):
	ax[0].scatter(xaxis,aMCMC[j],s=3,color='none',edgecolor=color[j],marker='o',
	label='$\Delta a$=%2.1f'%DeltaaAux[j])
	ax[1].scatter(xaxis,bMCMC[j],s=5,color='none',edgecolor=color[j],marker='o',
	label='$\Delta b$=%2.1f'%DeltabAux[j])
ax[0].set_ylabel('Parameter a')
ax[1].set_xlabel('Iteration Number')
ax[1].set_ylabel('Parameter b')
ax[0].set_xlim(0,NMCMC)
ax[1].set_xlim(0,NMCMC)
ax[0].yaxis.set_major_locator(MultipleLocator(10))
ax[0].yaxis.set_minor_locator(MultipleLocator(2))
ax[1].yaxis.set_major_locator(MultipleLocator(5))
ax[1].yaxis.set_minor_locator(MultipleLocator(1))
ax[1].xaxis.set_major_locator(MultipleLocator(NMCMC*0.100))
ax[0].xaxis.set_major_locator(MultipleLocator(NMCMC*0.100))
ax[0].grid(which='both')
ax[1].grid(which='both')
ax[0].legend(prop={'size': 10})
ax[1].legend(prop={'size': 10})
for axs in ax.flat:
	axs.label_outer()
plt.savefig('5PointMCMCTime.pdf')

