# Make inferences on longer MCMC for 5-point data
# ===============================================
exec(open('imports.py').read())
# Read in data from the MCMC
DeltaaAux=[1,10,50]
DeltabAux=DeltaaAux
color=['red','black','blue']
ZGType=['Full Chain','Thinned Chain','Batch Means']
# This MCMC must be generated with mcmc5points.py, and the Long suffix
# can be replaced with any other string 
data = np.genfromtxt('mcmc5pointsLong.txt',skip_header=0)

a1=data[:,0]
b1=data[:,1]
a2=data[:,2]
b2=data[:,3]
a3=data[:,4]
b3=data[:,5]
N=len(a1)       # number of links in the chain
# put the chains in a compact form
a=[a1,a2,a3]
b=[b1,b2,b3]
N=len(a[0])
print('Chains of length %d'%N)

# Choose a burn--in period. For a long chain maybe 10,000, 
# but smaller for a shorter chain
#nburn=10000 # to be excised
nburn=100

Mean=np.zeros((2,3))
Stdev=np.zeros((2,3))
StdevThin=np.zeros((2,3))
nThin=10
Median=np.zeros((2,3))
LL=np.zeros((2,3))
UL=np.zeros((2,3))
lowerp=0.16
upperp=0.84
for i in range(3):		# for the 3 choices of proposals
    for j in range(2): 	# for a and b
   # === mean and std.dev
        Mean[j,i]=np.mean(data[nburn:N,2*i+j])
        Median[j,i]=np.median(data[nburn:N,2*i+j])
        Stdev[j,i]=np.std(data[nburn:N,2*i+j])  
        StdevThin[j,i]=np.std(data[nburn:N:nThin,2*i+j])
        LL[j,i]=np.quantile(data[nburn:N,2*i+j],lowerp)
        UL[j,i]=np.quantile(data[nburn:N,2*i+j],upperp)
        print("E[X]=%3.3f  median=%3.3f  stdev=%3.3f (thinned: %3.3f) LL(%2.1f)=%3.3f UL(%2.1f)=%3.3f"%(Mean[j,i],Median[j,i],Stdev[j,i],StdevThin[j,i],lowerp,LL[j,i],upperp,UL[j,i]))
        print("Using C.I.: %3.3f +- %3.3f"%(0.5*(LL[j,i]+UL[j,i]),0.5*(UL[j,i]-LL[j,i])))
