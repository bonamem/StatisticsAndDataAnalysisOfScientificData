exec(open('imports.py').read())

# Simulate a normal distribution
fig,ax=plt.subplots(figsize=(8,6))

def function(x):
    result=x*np.log(x)
    return result

N=100
mu=np.arange(N)+1
ENorm=np.zeros(N)
for i in range(len(mu)):
    P=poisson(mu[i])
    ENorm[i]=P.dist.expect(lambda x: function(x), P.args,lb=1,ub=1000)-function(mu[i])
    print(mu[i], ENorm[i])

plt.plot(mu,ENorm,linewidth=2,color='black',label='E[$y_i \ln y_i$]-$\mu_i \ln \mu_i$' )
plt.hlines(0.5,0,100,color='red',linestyle='--',linewidth=2,label='Asymptotic value')
plt.xlabel('Parent mean $\mu_i$')
plt.ylabel('Expectation')
plt.legend()
plt.xlim(0,100)
plt.grid()
plt.savefig('CiAsymptotic.pdf')
