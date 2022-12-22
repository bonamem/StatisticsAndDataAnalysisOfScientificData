exec(open('imports.py').read())

# ==================================================================
# Define ad-hoc functions for variance and covariance using binned data
# These will be extended to non-integer data, as in Pearson's biometric data

# x: x coordinate of the binned array, n-dim. array/list
# multX: number of occurrences for the x value, n-dim. array/list
def varBinned(x,multX):
    xMean=np.average(x,weights=multX) 
    N=sum(multX) # total number of counts
    result=0
    for i in range(len(x)): #using the binned data
        result+=multX[i]*(x[i]-xMean)**2
    return result/(N-1)

# for the covariance, 
# x: x coordinate of the binned array, n-dim. array/list
# y: y coordinate, m-dim.
# ( grid: nxm matrix with (x,y) coordinates of the 2-D grid, using meshgrid )
# mult[i,j] is an nxm matrix
# n: # of bins for x variable
# m: # of bins for y variable
def covBinned(x,y,mult):
    # first need multX and multY, the multiplicities at fixed X or Y coordinate
    n=len(x) # 18
    m=len(y) # 19
    N=sum(mult.flatten()) # it needs to be flattened before summing over
    print('N=%f'%N)
    multX=np.zeros(n)
    multY=np.zeros(m)
    print('n=%d, m=%d, x[0]=%f, y[0]=%3.2f'%(n,m,x[0],y[0]))
    for i in range(n):
        multX[i]=sum(mult[:,i]) #multX[j]=sum(data[:,j])
    for j in range(m):
        multY[j]=sum(mult[j,:])

    xMean=np.average(x,weights=multX)
    yMean=np.average(y,weights=multY)
    print('Average father height: %3.4f, Average mother height: %3.4f, N=%d'%(xMean,yMean,N))
    result=0
    for i in range(n):
        for j in range(m):
            # x is mean of father's hight
            currentElement=(x[i]-xMean)*(y[j]-yMean)*mult[j,i]
            #print('i=%d,j=%d,mult[%d,%d]=%f'%(i,j,j,i,mult[j,i]))
            result+=currentElement# i,j]
    result/=(N-1) # check the N-1
    print('result:',result)
    return result

# Read data and debug ================================================
data = np.genfromtxt('fatherMotherStature.dat')
print('Dimensions of data:',len(data))
#print(data)
# mother[i,:] are rows of data for a fixed mother's height
motherRows=19 # this is y axis, m
fatherCols=18 # this is x axis, n
n=fatherCols
m=motherRows
print('Last data point: data[%d,%d]=%f'%(m,n,data[m-1,n-1]))
motherMin=52
motherMax=70
fatherMin=58
fatherMax=75
xaxisFather=np.linspace(fatherMin,fatherMax,num=fatherCols)+0.5
yaxisMother=np.linspace(motherMin,motherMax,num=motherRows)+0.5
print(xaxisFather,yaxisMother)

# -------------------------------------
multX=np.zeros(fatherCols) # 
multY=np.zeros(motherRows)
avgMotherHeightY=np.zeros(fatherCols)
avgFatherHeightX=np.zeros(motherRows)
# also sum of occurrence of that bin
sumNumberMotherHeight=np.zeros(motherRows)
sumNumberFatherHeight=np.zeros(fatherCols)

# Print rows and columns, for debugging
for i in range(motherRows):
    #print('row %d'%i,end=' ')
    #print(data[i,:])
    multY[i]=sum(data[i,:])
    # also add the average X=father's height for that Y
    avgFatherHeightX[i]=sum(data[i,:]*xaxisFather[:])/sum(data[i,:])
    # below is for the marginal distribution
    sumNumberMotherHeight[i]=sum(data[i,:])
print('Fathers height',avgFatherHeightX)
for j in range(fatherCols):
    #print('col %d'%j,end=' ');
    #print(data[:,j])
    multX[j]=sum(data[:,j])
    # also add the average Y=mother's height for that X
    avgMotherHeightY[j]=sum(data[:,j]*yaxisMother[:])/sum(data[:,j])
    sumNumberFatherHeight[j]=sum(data[:,j])
print('Mothers height',avgMotherHeightY)

# test the variance and covariance routines
x=xaxisFather
y=yaxisMother
meanx=np.average(x,weights=multX)
meany=np.average(y,weights=multY)
varx=varBinned(x,multX)
vary=varBinned(y,multY)
covxy=covBinned(x,y,data)
N=sum(data.flatten()) # Number of counts
print('N=%d, E[X]=%3.3f, E[Y]=%3.3f'%(N,meanx,meany))
print('var x: %3.2f, var y: %3.2f, cov: %3.2f'%(varx,vary,covxy))

# ======================================================================

# 1.1 Plot the data to reproduce the figure in the 2nd edition
#plt.imshow(data)
#plt.show()

fig,ax=plt.subplots(figsize=(8,8))
# 1.2 Now try again with a flattened array
dataFlat=data.flatten(order='C')
for i in range(motherRows):
    for j in range(fatherCols):
        # rescale the size of the datapoints
        counts=dataFlat[j+i*fatherCols]
#        counts=dataFlat[i+j*fatherCols] 
        markerSize=4*((counts>0)+counts)
        plt.scatter(xaxisFather[j],yaxisMother[i],s=markerSize,facecolors='none',color='black',marker='o')
        if (counts>0):
            plt.text(xaxisFather[j]-0.45,yaxisMother[i]-0.45,"%2.2f"%counts,fontsize=4)

# Set the tick mark locations
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))

plt.xlabel('X (Father\'s Height)')
plt.ylabel('Y (Mother\'s Height)')
plt.xlim(fatherMin,fatherMax+2) # to preserve the aspect ratio
plt.ylim(motherMin,motherMax+1)
plt.grid(which='both')
#plt.savefig('pearsonScatter2D.pdf')


# ==============================================================
# Now do the two linear regressions
# Use variances and covariance calculated above
# 2.1 X= father's height; Y= mother's height
bYgX=covxy/varx
aYgX=meany-bYgX*meanx
print('Regression Y given X: a=%3.3f, b=%3.3f'%(aYgX,bYgX))
xPlot=np.zeros(2)
xPlot=[fatherMin,fatherMax+1]
yPlot=np.zeros(2)
for i in range(2):
    yPlot[i]=(bYgX*xPlot[i])+aYgX
plt.plot(xPlot,yPlot,linewidth=2,color='blue',label='Regression of Y on X')

# 2.2 X = mother's height; Y= father's height

bXgY=vary/covxy
aXgY=meany-bXgY*meanx
print('Regression X given Y: a=%3.3f, b=%3.3f'%(aXgY,bXgY))
for i in range(2):
    yPlot[i]=(bXgY*xPlot[i])+aXgY
plt.plot(xPlot,yPlot,linewidth=2,color='red',label='Regression of X on Y')
plt.legend(loc=4,prop={'size': 12})
plt.savefig('pearsonScatter2D.pdf')


# problem 14.2 (r statistic)
# Also calculate r^2
r=covxy/(varx*vary)**0.5
print('Problem 14.2')
print('Correlation coefficient r=%3.3f, r**2=%3.3f'%(r,r**2))
print('b*b\'=%3.3f'%(bYgX/bXgY))


# ======================================================================================
# Evaluate the critical values based on symmetric beta r-distribution 
# and the standard beta distr. for r^2
f=N-2
print('Number of degrees of freedom for r: %d'%f)
# Double--sided critical value
# The two--sided probability is accounted for by 0.99 -> 0.995 for the Critical Value
# NOTE: PYTHON HAS NUMERICAL PROBLEMS WITH THIS P-VALUE 
pValuer=rdist.sf(abs(r), f) # this is one-sided survival probability
pValuer=pValuer*2 # add the other tail
critr=rdist.ppf(0.995,f)
print('critical value of r: %3.4f (for r^2=%3.5f), p value of r: %3.3e'%(critr,critr**2,pValuer))
# Single--sided critical value:
pValuer2=beta.sf(r**2, 1/2,f/2)
critr2=beta.ppf(0.99,1/2,f/2)
print('(One--sided) critical value of r2: %3.5f, p value of r2: %3.3e'%(critr2,pValuer2))
# =======================================================================
# Check an alternative way to do regression by using X vs <Y> values
# 3.1 Using X= Father's height, get average <Y> = Mother's height
# this is multX[j]
# problem 14.1
print('Problem 14.1')
fig,ax=plt.subplots(figsize=(8,8))
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))

print('Now analyzing father columns')
for j in range(fatherCols):
    
    print('%d: X=%2.2f, sum <Y>=%2.2f, <Y>=%2.2f'%(j,xaxisFather[j],multX[j],avgMotherHeightY[j]))

plt.plot(xaxisFather,avgMotherHeightY,color='blue',linestyle='',marker='o',label='(X, <Y>) data ')

slope, intercept, r_value, p_value, std_er=linregress(xaxisFather,avgMotherHeightY)
print('Fit of Avg Y given X:', slope,intercept)
for i in range(2):
    yPlot[i]=(slope*xPlot[i])+intercept
plt.plot(xPlot,yPlot,linewidth=2,color='blue',linestyle='--',label='Regression of <Y> on X')
# repeat the plot from the contingency table
for i in range(2):
    yPlot[i]=(bYgX*xPlot[i])+aYgX
plt.plot(xPlot,yPlot,linewidth=2,color='blue',label='Regression of Y on X')

#3.2, usinx Y=Mother's Height, and <X> = Father's height
print('Now analyzing mother rows')
for i in range(motherRows):

    print('%d: X=%2.2f, sum <Y>=%2.2f, <Y>=%2.2f'%(i,yaxisMother[i],multY[i],avgFatherHeightX[i]))

plt.plot(avgFatherHeightX,yaxisMother,color='red',linestyle='',marker='o',label='(<X>,Y) data')
slope, intercept, r_value, p_value, std_er=linregress(avgFatherHeightX,yaxisMother)
print('Fit of Avg. X given Y',slope,intercept)
for i in range(2):
    yPlot[i]=(slope*xPlot[i])+intercept
plt.plot(xPlot,yPlot,linewidth=2,color='red',linestyle='--',label='Regression of <X> on Y')
# repeat the plot from the contingency table
for i in range(2):
    yPlot[i]=(bXgY*xPlot[i])+aXgY
plt.plot(xPlot,yPlot,linewidth=2,color='red',label='Regression of X on Y')

plt.legend(loc=4,prop={'size': 12})
plt.xlabel('X (Father\'s Height)')
plt.ylabel('Y (Mother\'s Height)')
plt.xlim(fatherMin,fatherMax+2) # To preserve the aspect ratio
plt.ylim(motherMin,motherMax+1)
plt.grid(which='both')
plt.savefig('pearsonXvsavgY.pdf')

# ==============================================================
# Plot another figure for Chapter 2, including marginals
# Use the sums by row and column
fig,ax=plt.subplots(figsize=(8,6))
ax.plot(xaxisFather,sumNumberFatherHeight,marker='o',linewidth=2,color='black',label='Father\'s Height')
ax.plot(yaxisMother,sumNumberMotherHeight,marker='o',linewidth=2,color='blue',label='Mother\'s Height')
ax.set_xlabel('Height')
ax.set_ylabel('Number of occurrence')
plt.ylim(0,200)
plt.legend(loc=2,prop={'size': 12})
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(20))
ax.yaxis.set_minor_locator(MultipleLocator(5))
plt.grid(which='both')

plt.savefig('pearsonMarginals.pdf')
# Check the sample means of father's and mother's heights
meanFHeight=np.average(xaxisFather,weights=sumNumberFatherHeight)
print('Mean height of father: %3.2f'%meanFHeight)
