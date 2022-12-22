exec(open('imports.py').read())

# This code is for Problem 22.4 of the textbook
# Set up the RHat score

m=2
# NOTE: these need to be arrays, not lists
a=np.asarray([7,8,10,11,10,8])
b=np.asarray([11,11,8,10,9,12])
MCMC=[a,b]
print(a)
n=len(MCMC[0])       # number of links in the chain
print('Read %d chains of length %d'%(m,n))

indexRange=np.arange(0,3,dtype=int)
print(MCMC[0][indexRange],MCMC[1])

b=3 # length of each batch
N=int(n/b) # number of batches
print('Using %d batches'%N)
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
    indexRange=np.arange(0,b*(i+1),1,dtype=int)
    print(indexRange)
    print('from %d to %d'%(min(indexRange),max(indexRange)))
    for j in range(m):  #j-th chain
        thetaMeanI[j,i]=np.mean(MCMC[j][indexRange]) # from i to N
        print('mean %3.3f'%thetaMeanI[j,i])
        s2i[j,i]=np.var(MCMC[j][indexRange],ddof=1)
    thetaMean[i]=np.mean(thetaMeanI[:,i]) # mean of m chains
    print('thetaMean=%3.2f'%thetaMean[i])
    B[i]=Ni[i]*np.var(thetaMeanI[:,i],ddof=1)
    W[i]=np.mean(s2i[:,i]) # mean of sample variances
    VHat[i]=((Ni[i]-1)/Ni[i])*W[i] +B[i]/Ni[i] +B[i]/(m*Ni[i])
    RHat[i]=VHat[i]/W[i]
    print('%d N=%d thetaMean=%3.3f, B=%3.3f, B/N=%3.3f, W=%3.3f, VHat=%3.3f, RHatSq=%3.3f'
    %(i,Ni[i],thetaMean[i], B[i],B[i]/Ni[i],W[i],VHat[i],RHat[i]**0.5))

