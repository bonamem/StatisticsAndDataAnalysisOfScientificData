exec(open('imports.py').read())

from scipy.optimize import fsolve
import numpy as np
X=np.array([0,1,2,3,4]) 
# choose the Y value for the following problems
#Y=np.array([0,0,1,0,1]) # prob16.2 without solution
Y=np.array([1,0,1,0,1]) # prob16.1
N=len(X)
def equations(p):
    a, b = p
    x=X
    y=Y
    N=len(x)
    
    return (sum(y/(a+b*x))-N,sum(y*x/(a+b*x)) - sum(x))

a, b =  fsolve(equations, (1, 0))

print(equations((a, b)))
print('Solution:', (a,b))

# report cstat
ymod=np.zeros(N)
for i in range(N):
    ymod[i]=a+X[i]*b

Cstat=cstat(Y,ymod)
print('best-fit cstat=%3.3f'%Cstat)
