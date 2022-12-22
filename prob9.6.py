exec(open('imports.py').read())

data=[20.3, 20.4, 19.8, 20.4, 19.9, 20.7]
N=len(data)
smean=np.mean(data)
stdev=np.std(data,ddof=1)
print('measurement: %3.3f +- %3.3f'%(smean,stdev))
mu=20
tStat=(smean-mu)/(stdev/N**0.5)
print(' t stat: %3.2f'%tStat)

#  critical value needs to be two--sided
# Use symmetry of t dist around 0, for 90% probability
tCrit=t.ppf(0.95,6)
print('Critical value of t at 0.9 probability: %3.3f'%tCrit)
