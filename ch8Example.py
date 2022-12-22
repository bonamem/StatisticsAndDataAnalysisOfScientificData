exec(open('imports.py').read())


x=np.array([1.2,0.8])
xE=np.array([0.24,0.16])

xWAvg=np.average(x,weights=1/xE**2)
print('Weighted average: %3.4f'%xWAvg)


log10X=np.log10(x)
log10XE=xE/x/np.log(10)
lnX=np.log(x)
lnXE=xE/x
print('log10:',log10X)
print('ln:',lnX)
print(log10XE,lnXE)

lnXAvg=np.mean(lnX)
log10XAvg=np.mean(log10X)
print('average of log10: %3.4f (%3.4f), average of ln: %3.4f (%3.4f)'
        %(log10XAvg,10**log10XAvg,lnXAvg,np.exp(lnXAvg)))
