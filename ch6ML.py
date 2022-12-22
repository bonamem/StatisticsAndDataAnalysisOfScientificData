exec(open('imports.py').read())

# 1. use the example with gaussian and Poisson treatment
# number of counts and times 
x=np.array([100,180,33])
t=np.array([10,20,3])

cr=x/t

print("count rate",cr)

wAvg=np.average(x,weights=t**(-0.5))
print("weighted average of counts",wAvg)
# assuming Gaussian errors
crErr=x**0.5/t
print('Gaussian errors',crErr)


# 2. do weighted average of the count rates, which
# is foolish because we can add number together, but it's an example

wAvgcr=np.average(cr,weights=1/crErr**2)
print('Weighted average of count rates',wAvgcr)
wAvgcrErr=(1/sum(1/crErr**2))**0.5
print('Std. dev. of weighted average',wAvgcrErr)

# 3. Combine the measurements with a Direct average

crD=sum(x)/sum(t)
crDErr=sum(x)**0.5/sum(t)
print("Poisson average and std. dev:",crD,crDErr)
