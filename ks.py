exec(open('imports.py').read())

# critical values of KS test in asymptotic regime
N=100
pVal=[0.5,0.6,0.7,0.8,0.9,0.95,0.99]
NpVal=len(pVal)
for i in np.arange(NpVal):
	Cvalbign=kstwobign.ppf(pVal[i])
	Cval=kstwo.ppf(pVal[i],N)
	print('%3.3f %3.3f  %3.3f'%(pVal[i],Cvalbign,Cval))
#kstest
