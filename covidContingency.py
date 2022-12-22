exec(open('imports.py').read())
from math import factorial as fact
from scipy.stats import fisher_exact,chi2

# Analyze the OpenSAFELY data using contingency tables

data = np.genfromtxt('openSafely.dat')
print(data)

r=6     # 6 rows
c=2     # 2 columns

totals=data[:,1]
deaths=data[:,2]
survivals=totals-deaths
print(deaths,survivals)
N=sum(totals)

# re-arrange the table with survivals, deaths only as 6x2 table
M=np.zeros((r,c))   # this is the table with measured values
M[:,0]=deaths
M[:,1]=survivals

print(M)

#   ===================================================
# Now find the margins
p=deaths/survivals  # measured probabilities
n=totals
m=np.zeros(c)
m[0]=np.sum(deaths)
m[1]=np.sum(survivals)
print('Margins n:', n)
print('Margins m:',m)
print('Measured probabilities',p)
print('m1/N',m[0]/N)

# ===================================================
# Calculate the expected rates, assuming independence
# Also calculate chi^2

E=np.zeros((r,c)) # this is the table with expectations, from independence
for i in range(r):
    for j in range(c):
        E[i,j]=m[j]*n[i]/N
        print('%f (%f) '%(M[i][j],E[i][j]), end=' ')
    print(' ')

# All chi2 values can be obtained as 
Chi2Table=(E-M)**2/E
print('Table with chi2 values:',Chi2Table)
chi2Measured=np.sum(Chi2Table)
print('Total chi2=%f'%chi2Measured)
dof=(r-1)*(c-1) # degrees of freedom for thic chi2
# Use numpy's inverse survival function for critical value, at 90% level
print('Critical value for parent distrib. chi2(%d)=%f'%(dof,chi2.isf(0.1,dof)))

print('Sums of deaths (%d), survivals (%d), combined (%d) and expectations (%f)'%(m[0],m[1],m[0]+m[1],np.sum(E[:,0])))
