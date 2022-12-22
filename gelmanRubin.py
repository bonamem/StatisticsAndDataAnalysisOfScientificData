exec(open('imports.py').read())
# Read in data from the MCMC
# Test with the same chains as for Geweke Z scores
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
print(a1)
# put the chains in a compact form
aMCMC=[a1,a2,a3]
bMCMC=[b1,b2,b3]
quit()
# Set up the RHat score
m=3

n=len(a1)       # number of links in the chain
b=100 # length of each batch
N=int(n/b) # number of batches
thetaMeanI=np.zeros((m,N)) # by stepping the window forward
thetaMean=np.zeros(N) # mean of means
B=np.zeros(N)
W=np.zeros(N)
VHat=np.zeros(N)
RHat=np.zeros(N)
s2i=np.zeros((m,N)) # sample variance of  i-th chain
Ni=np.zeros(N) # number of chain elements considered)
for i in range(N):
	Ni[i]= b*(i+1) # N-i # number of elements of chain considered
	indexRange=np.arange(0,b*(i+1),1) 
	print('from %d to %d'%(min(indexRange),max(indexRange)))
	for j in range(m):  #j-th chain
		thetaMeanI[j,i]=np.mean(aMCMC[j][indexRange]) # from i to N
		s2i[j,i]=np.var(aMCMC[j][indexRange],ddof=1)
	thetaMean[i]=np.mean(thetaMeanI[:,i]) # mean of m chains
	print('thetaMean=%3.2f'%thetaMean[i])
	B[i]=Ni[i]*np.var(thetaMeanI[:,i],ddof=1) 
	W[i]=np.mean(s2i[:,i]) # mean of sample variances
	VHat[i]=((Ni[i]-1)/Ni[i])*W[i] +B[i]/Ni[i] +B[i]/(m*Ni[i])
	RHat[i]=VHat[i]/W[i]
	print('%d N=%d thetaMean=%3.3f, B=%3.3f, B/N=%3.3f, W=%3.3f, VHat=%3.3f, RHat=%3.3f'
		%(i,Ni[i],thetaMean[i], B[i],B[i]/Ni[i],W[i],VHat[i],RHat[i]))
xaxis=np.arange(b,(N+1)*b,b)
# exclude the last few RHat scores because not meaningful
fig,ax=plt.subplots(1,1,figsize=(8,6))
plt.plot(xaxis[0:N],RHat[0:N]**0.5,color='black',linewidth=2,label='$\sqrt{\hat{R}}$ statistic')
plt.plot(xaxis[0:N],VHat[0:N]**0.5,color='blue',linewidth=2,label='$\sqrt{\hat{V}}$ (variance)')
plt.plot(xaxis[0:N],W[0:N]**0.5,color='red',linewidth=2,label='$\sqrt{W}$ (within-chain variance)')
plt.plot(xaxis[0:N],(B[0:N]/Ni)**0.5,color='green',linewidth=2,label='$\sqrt{B/N}$ (between-chain variance)')
plt.plot(xaxis[0:N],thetaMean/10,color='grey',linewidth=2,label='Mean of ${a}/{10}$')
plt.legend(prop={'size': 12})
plt.xlabel('Number of Iterations')
plt.ylabel('Statistic')
plt.grid(which='both')
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.2))
ax.xaxis.set_major_locator(MultipleLocator(2000))
ax.xaxis.set_minor_locator(MultipleLocator(500))
plt.hlines(25.44/10,xaxis[0],xaxis[-1],linestyle='--',color='grey')
plt.hlines(4.26,xaxis[0],xaxis[-1],linestyle='--',color='blue')
plt.ylim((0,8))
plt.xlim((0,10000))
plt.savefig('gelmanRubin.pdf')
