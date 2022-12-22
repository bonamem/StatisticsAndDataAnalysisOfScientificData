exec(open('imports.py').read())

# Python code to do statistics from Hubble data
# Bonamente 2020

# Import the data from pressureRatio.dat
# ===============================================================
# Character>---->-------Dominant>-------Recessive>------fraction
# SL	SW	PL	PW
dataTypes=[('SL1','f'),('SW1','f'),('PL1','f'),('PW1','f'),('species1','S8'),
        ('SL2','f'),('SW2','f'),('PL2','f'),('PW2','f'),('species2','S8'),
        ('SL3','f'),('SW3','f'),('PL3','f'),('PW3','f'),('species3','S8')]

# 1 - Iris setosa
# 2 - Iris versicolor
# 3 - Iris Virginica
data = np.genfromtxt('irisData.dat',skip_header=1,dtype=dataTypes)
print(data)


# =======================================
# ID: (select one that applies)
#     1 - Iris Setosa 
#     2 - Iris versicolor 
#     3 - Iris Virginica
IDLabel=['Iris Setosa','Iris versicolor','Iris Virginica']
ID=2
# problems  13.1, 13.2, 13.3 and 13.6, 13.7 and 13.8
print('*** NOTE: Selected %s'%IDLabel[ID-1])
y=data['SL%d'%ID]   # sepal length
x1=data['SW%d'%ID]  # sepal width
x2=data['PL%d'%ID]  # petal length
x3=data['PW%d'%ID]  # petal width
m=3
N=len(y)
print('m=%d, N=%d'%(m,N))

# generate the matrix A and the vector beta
X = np.c_[np.ones(N),x1,x2,x3]

# Do this two ways: Eq. (13.5) and solving the system
# 1. Eq. 12.9

A = np.matmul(np.transpose(X),X) 
betaV = np.matmul(np.transpose(X),y)
a= np.matmul(np.linalg.inv(A),betaV)
# this is also problem 13.1, when ID=1 is selected
print('best-fit parameters',a)
print('==============================')

# 2. Using linalg.solve

a = np.linalg.solve(A,betaV)
print('Equivalent solution')
print(a)

# ========================================
# Also get chi^2 assuming sigma^2 is known

# a value of sigma=1 ignores the variance
sigma=1.0
chicrit=chi2.ppf(0.99,46)
ymean=np.mean(y)
yvar=np.var(y,ddof=1) # this is divided by N-1
S2=yvar*(N-1)
print('mean y: %3.3f'%ymean)

# Residual variance ===================
yhat=np.zeros(N)
for i in range(N):
    yhat[i]=(a[0]+a[1]*x1[i]+a[2]*x2[i]+a[3]*x3[i])
print('yhat',yhat)
Sr2=((y-yhat)/sigma)**2
# Explained variance ===================
Se2=((yhat-ymean)/sigma)**2
print('S2=%3.3f, Sr2=%3.3f, Se2=%3.3f, S2=Sr2+Se2=%3.3f, critical value 0.99:%3.3f'%
	(S2,sum(Sr2),sum(Se2),sum(Sr2)+sum(Se2),chicrit))
R2=sum(Se2)/(sum(Se2)+sum(Sr2))
print('Coefficient of determination R^2: %3.3f'%R2)
# problem 13.2

FStat=(sum(Se2)/m)/(sum(Sr2)/(N-m-1))
pF=f.sf(FStat,m,N-m-1)
print('F=%3.3f (p-value %3.2e)'%(FStat,pF))
# estimate the model sample variance for Iris Setosa
s2hat = sum((y-yhat)**2)/(N-m-1)
print('model sample variance: %3.3f, std. dev: %3.3f'%(s2hat,s2hat**0.5))

# Now get the parameter errors using \hat{s}^2
sigma=s2hat**0.5
# error or covariance matrix
e= s2hat*np.linalg.inv(A)
print(e)
aErr=np.zeros(m+1)
aErr=[ e[i,i]**0.5 for i in range(m+1)]
tStat=a/aErr #t-statistic scores
# 2-sided p-value
pValue=2*t.sf(abs(tStat),N-m-1)
print(pValue)
for i in range(m+1):
	print('Parameter %d: %3.3f error: %3.3f, tscore: %3.3f , pvalue=%3.3e'%(i,a[i],aErr[i],tStat[i],pValue[i])) 

# ==================================================
# ==================================================
# now fit with just one predictor 
# this is the usual linear regression
# x1; sepal width
# x2: petal length
# problem 14.4
#xReg=x1 #using sepal width as predictor variable, for problem 14.4
xReg=x2 # using petal length as predictor variable, for problem 14.5
results=stats.linregress(xReg,y)
b=results.slope
a=results.intercept
r=results.rvalue
p=results.pvalue
berr=results.stderr
aerr=results.intercept_stderr
print('====================================================')
print("Simple linear regression with one predictor variable")
print("a: %f+-%f    b: %f+-%f" % (a,aerr, b,berr))
print("r=%3.4f, R-squared: %f (p value: %3.3e)" % (r,r**2,p))
# best-fit model
for i in range(N):
    yhat[i]=(a+b*xReg[i])
print('yhat',yhat)

sigma=1.0 # there is no error reported for the data
m=1
Sr2=((y-yhat)/sigma)**2
# Explained variance ===================
Se2=((yhat-ymean)/sigma)**2
print('S2=%3.3f, Sr2=%3.3f, Se2=%3.3f, Sr2+Se2=%3.3f'%
        (S2,sum(Sr2),sum(Se2),sum(Sr2)+sum(Se2)))
FStat=(sum(Se2)/m)/(sum(Sr2)/(N-m-1))
pF=f.sf(FStat,m,N-m-1)
print('F=%3.3f (p-value %3.2e)'%(FStat,pF))
f=N-2
pr2=beta.sf(r**2,1/2,f/2)
print('r2=%3.3f (p-value %3.2e)'%(r**2,pr2))
# estimate the model sample variance for Iris Setosa
s2hat = sum((y-yhat)**2)/(N-m-1)
print('model sample variance: %3.3f, std. dev: %3.3f'%(s2hat,s2hat**0.5))

