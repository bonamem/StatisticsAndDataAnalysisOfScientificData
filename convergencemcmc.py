exec(open('imports.py').read())
# Read in data from the MCMC
DeltaaAux=[1,10,50]
DeltabAux=DeltaaAux
color=['red','black','blue']
ZGType=['Full Chain','Thinned Chain','Batch Means']
data = np.genfromtxt('mcmc5points.txt',skip_header=0)

a1=data[:,0]
b1=data[:,1]
a2=data[:,2]
b2=data[:,3]
a3=data[:,4]
b3=data[:,5]
N=len(a1)	# number of links in the chain
# put the chains in a compact form
a=[a1,a2,a3]
b=[b1,b2,b3]
print(len(a[0]))

# this is saved for later use 
out=open('a1.txt','w')
for i in range(N):
	out.write('%3.4f\n'%a1[i])
out.close()

out=open('a2.txt','w')
for i in range(N):
        out.write('%3.4f\n'%a2[i])
out.close()

out=open('a3.txt','w')
for i in range(N):
	out.write('%3.4f\n'%a3[i])
out.close()

out=open('b1.txt','w')
for i in range(N):
	out.write('%3.4f\n'%b1[i])
out.close()

out=open('b2.txt','w')
for i in range(N):
	out.write('%3.4f\n'%b2[i])
out.close()

out=open('b3.txt','w')
for i in range(N):
	out.write('%3.4f\n'%b3[i])
out.close()


thin=10	# thinning of the chain

NA=int(0.1*N)
NB=int(0.5*N)
print('Making Geweke Z scores for %d and %d segments'%(NA,NB))

NSteps=int((N-NA-NB)/thin)
NAthin=int(NA/thin)
NBthin=int(NB/thin)
print('Preparing %d ZG scores'%NSteps)
# setup means and variances
aMeanA=np.zeros((3,NSteps))
aMeanB=np.zeros((3,NSteps))
aVarA=np.zeros((3,NSteps))
aVarB=np.zeros((3,NSteps))
aVarABM=np.zeros((3,NSteps))
aVarBBM=np.zeros((3,NSteps))
aVarAThin=np.zeros((3,NSteps))
aVarBThin=np.zeros((3,NSteps))
bMeanA=np.zeros((3,NSteps))
bMeanB=np.zeros((3,NSteps))
bVarA=np.zeros((3,NSteps))
bVarB=np.zeros((3,NSteps))
bVarABM=np.zeros((3,NSteps))
bVarBBM=np.zeros((3,NSteps))
bVarAThin=np.zeros((3,NSteps))
bVarBThin=np.zeros((3,NSteps))
aZG=np.zeros((3,NSteps))
aZGThin=np.zeros((3,NSteps))
aZGBM=np.zeros((3,NSteps))

bZG=np.zeros((3,NSteps))
bZGThin=np.zeros((3,NSteps))
bZGBM=np.zeros((3,NSteps))

# Test the batch-means method to estimate variance
aVar=np.zeros(3)
aVarBM=np.zeros(3)
aVarThin=np.zeros(3)
iThin=np.arange(0,N,thin,dtype=int)
for j in range(3):
	aVar[j]=np.var(a[j][:],ddof=1) # sample variance
	aVarThin[j]=np.var(a[j][iThin],ddof=1) # thinned sample variance
	aVarBM[j]=VarBM(a[j])	# Batch Mean estimator for variance
	print('Variance of a[%d]: %3.3f, thinned a: %3.3f,  BM Variance: %3.3f'
		%(j,aVar[j],aVarThin[j],aVarBM[j]))

# ==================================================

fig,ax=plt.subplots(2,1,figsize=(8,6))
xaxis=np.arange(0,N-NB-NA,thin,dtype=int)
for i in range(NSteps):
	istart=i*thin
	inA=np.arange(istart,istart+NA,1,dtype=int)
	inAThin=np.arange(istart,istart+NA,thin,dtype=int)	# this moves forward
	inB=np.arange(N-NB,N,1,dtype=int)
	inBThin=np.arange(N-NB,N,thin,dtype=int)		# this is constant
	for j in range(3):	# for the 3 chains with different proposals
		# estimates of the mean in intervals A and B
		aMeanA[j,i]=np.mean(a[j][inA])
		aMeanB[j,i]=np.mean(a[j][inB])
		bMeanA[j,i]=np.mean(b[j][inA])
		bMeanB[j,i]=np.mean(b[j][inB])

		# 3 types of variance estimators: straight, thinned and BM
		# for segment A
		aVarA[j,i]=np.var(a[j][inA],ddof=1)/NA
		aVarAThin[j,i]=np.var(a[j][inAThin],ddof=1)/NAthin # variance of sample mean
		aVarABM[j,i]=VarBM(a[j][inA])/NA # Batch Mean variance
		aVarB[j,i]=np.var(a[j][inB],ddof=1)/NB
		aVarBThin[j,i]=np.var(a[j][inBThin],ddof=1)/NBthin
		aVarBBM[j,i]=VarBM(a[j][inB])/NB #    "      "
		# for segment B
		bVarA[j,i]=np.var(b[j][inA],ddof=1)/NA
		bVarAThin[j,i]=np.var(b[j][inAThin],ddof=1)/NAthin
		bVarABM[j,i]=VarBM(b[j][inA])/NA # Batch Mean variance
		bVarB[j,i]=np.var(b[j][inB],ddof=1)/NB 
		bVarBThin[j,i]=np.var(b[j][inBThin],ddof=1)/NBthin
		bVarBBM[j,i]=VarBM(b[j][inB])/NB #    "      "

		aZG[j,i]=(aMeanA[j,i]-aMeanB[j,i])/((aVarA[j,i]+aVarB[j,i])**0.5)
		aZGThin[j,i]=(aMeanA[j,i]-aMeanB[j,i])/((aVarAThin[j,i]+aVarBThin[j,i])**0.5)
		aZGBM[j,i]=(aMeanA[j,i]-aMeanB[j,i])/((aVarABM[j,i]+aVarBBM[j,i])**0.5)

		bZG[j,i]=(bMeanA[j,i]-bMeanB[j,i])/((bVarA[j,i]+bVarB[j,i])**0.5)
		bZGThin[j,i]=(bMeanA[j,i]-bMeanB[j,i])/((bVarAThin[j,i]+bVarBThin[j,i])**0.5)
		bZGBM[j,i]=(bMeanA[j,i]-bMeanB[j,i])/((bVarABM[j,i]+bVarBBM[j,i])**0.5)

		print('%d: ZG scores A: %2.1f B: %2.1f'%(i,aZG[j,i],bZG[j,i]))
# Now make plots
for j in range(1,2,1):
	ax[0].plot(xaxis,aZG[j,:],color=color[j],label='$\Delta a$=%2.1f %s'%(DeltaaAux[j],ZGType[0]))
	ax[0].plot(xaxis,aZGThin[j,:],color=color[j],linestyle=':',label='$\Delta a$=%2.1f %s'%(DeltaaAux[j],ZGType[1]))
	ax[0].plot(xaxis,aZGBM[j,:],color=color[j],linestyle='--',label='$\Delta a$=%2.1f %s'%(DeltaaAux[j],ZGType[2]))
	ax[1].plot(xaxis,bZG[j,:],color=color[j])#,label='$\Delta b$=%2.1f'%DeltabAux[j])
	ax[1].plot(xaxis,bZGThin[j,:],color=color[j],linestyle=':')
	ax[1].plot(xaxis,bZGBM[j,:],color=color[j],linestyle='--')
ax[0].grid(which='both')
ax[1].grid(which='both')
ax[0].set_ylabel('$Z_G$ Parameter $a$')
ax[1].set_xlabel('Start of segment A')
ax[1].set_ylabel('$Z_G$ Parameter $b$')
for axs in ax.flat:
        axs.label_outer()
ax[0].legend(prop={'size': 12})
#ax[1].legend(prop={'size': 12})
ax[0].set_xlim(xaxis[0],xaxis[-1])
ax[1].set_xlim(xaxis[0],xaxis[-1])
ax[0].hlines(0,xaxis[0],xaxis[-1],linestyle='--',color='black')
ax[0].hlines(3,xaxis[0],xaxis[-1],linestyle=':',color='black')
ax[0].hlines(-3,xaxis[0],xaxis[-1],linestyle=':',color='black')
ax[1].hlines(0,xaxis[0],xaxis[-1],linestyle='--',color='black')
ax[1].hlines(3,xaxis[0],xaxis[-1],linestyle=':',color='black')
ax[1].hlines(-3,xaxis[0],xaxis[-1],linestyle=':',color='black')
ax[1].yaxis.set_major_locator(MultipleLocator(2))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.5))
ax[0].yaxis.set_major_locator(MultipleLocator(2))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.5))

ax[0].set_ylim(-5,5)
ax[1].set_ylim(-5,5)
plt.savefig('mcmcZg.pdf')
