exec(open('imports.py').read())

# Make a plot with the Gaussians to illustrate fitting


ymin=0
ymax=5.5
6
xline=np.array([0,10])
#yline=np.array([0,0])
# assume this line model
a=1.0
b=0.4
yline=a+xline*b


fig,ax=plt.subplots(figsize=(8,6))

plt.plot(xline,yline,linewidth=2,color='black')



# add Gaussians on the other axis
nPoints=4
# add a fictional data set
xi=np.array([0.7,3,5,7.5])
yi=np.array([1.7,1.8,3.4,3.7])
sigmai=np.array([0.3,0.3,0.2,0.5])
# plot the data with error bars
plt.errorbar(xi,yi,yerr=sigmai,color='blue',capsize=4,fmt='o',capthick=2,label='Gaussian data')

x=np.linspace(-10,10,1000)
mu=np.zeros(nPoints)
# mean of the Gaussians
mu=a+xi*b
x=np.linspace(ymin,ymax,1000)
for i in range(nPoints):
    y=norm.pdf(x,loc=mu[i],scale=sigmai[i]) # distribution of i-th data point
    ax.plot(y+xi[i],x,color='grey',linewidth=2)
    ax.vlines(xi[i],ymin,ymax,linestyle=':',linewidth=2,color='black')

ax.scatter(xi,mu,marker='_',linewidth=4,color='black')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_ylim(ymin,ymax)
ax.set_xlim(0,10)
plt.savefig('gaussianFit.pdf')
