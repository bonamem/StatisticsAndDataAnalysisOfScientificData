# Python code to do statistics from Mendel data
# Bonamente 2020

exec(open('imports.py').read())
# Import the data from mendel.dat
# ===============================================================
# Character		Dominant	Recessive	fraction
dataTypes=[('character','S20'),('dominant','I'),('recessive','I'),('fraction','f')]

data = np.genfromtxt('mendel.dat',skip_header=1,dtype=dataTypes)

character = data['character']
dominant = data['dominant']
recessive = data['recessive']
fraction = data['fraction']

#print(dominant)
#print(recessive)
#print(dominant/(recessive+dominant))
print(character)
print('Fraction of dominants')
print(fraction)

# ======================================================
# 2. Calculate standard deviations of number of dominant
# problem 6.2
total=dominant+recessive
print('Problem 6.2')
p=0.75   # parent value for fraction of dominant
q=0.25  # parent value for fraction of recessive

stdDominant = np.sqrt(total*p*q)
print('Standard deviation of dominants')
print(stdDominant)
stdDominantFraction = stdDominant/total
print('Standard deviation of fraction of dominants')
print(stdDominantFraction)
# also sample variance of fraction of dominants

# =======================================================
# 3.8 Simple calculation of the sample variance of binomial

# 7th line has the long vs short stem
print('Problem 9.7')
index=6
N7=total[index]
D7=dominant[index]
s27=N7*p*q
E7=N7*p
print('Measurement of dominants: %d'%D7)
print('Expectation of dominant  (long vs. short stem): %3.3f'%E7)
print('Variance of binomial (long vs. short stem): %3.3f, N=%d'%(s27,N7))
# Also estimate p7 from the data
p7=dominant[index]/N7
s2p7=N7*p7*(1-p7)
print('"Sample" variance of binomial (long vs. short stem): %3.3f, N=%d'%(s2p7,N7))
zScore=(D7-E7)/s27**0.5
print('z score for number of dominants (long vs. short stem):%3.3f'%zScore)



#problem 3.9, similar to 3.8
print('Problem 3.9')
N=np.sum(total)
print('Moments of distribution for fraction: mean=%3.2f, variance=%3.2e'%(p,p*q/N))
pStDev=(p*q/N)**0.5
print('Standard deviation: %3.2e'%(pStDev))

pData=sum(dominant)/N
print('Measurement of the fraction of dominants: %3.3f'%pData)

zScore=(pData-p)/pStDev
zScorepValue=norm.cdf(abs(zScore))-norm.cdf(-abs(zScore))
print('Approximate zscore and associated probability: %3.2f (%3.3f)'%(zScore,zScorepValue))
print('======================')
# ======================================================
# 3.1 Weighted average of the seven dominants
# problem 9.8
print('Problem 9.8')
weightedAvgDominantFraction=np.average(fraction,weights=1./stdDominantFraction**2)
# also calculate the error on the weighted mean
weightedAvgDominantFractionError=np.sqrt(1./np.sum(1/stdDominantFraction**2))
print("Weighted average of dominant fraction: %4.5f+-%4.5f"%(weightedAvgDominantFraction,weightedAvgDominantFractionError))

# 3.2 this can be checked by a direct sum of all dominants
fractionSum=np.sum(dominant)/np.sum(total)
fractionSumError = np.sqrt(np.sum(total)*p*q)/np.sum(total)
print("Overall dominant fraction: %4.5f+-%4.5f"%(fractionSum,fractionSumError))
# also need the sample variance of fractions
fraction=dominant/total
fractionMean=np.mean(fraction)
fractionStd=np.std(fraction,ddof=1)
print('Mean and Std. dev. of the fractions: %3.4f+-%3.4f %3.3f (%3.3e)'%(fractionMean,fractionStd,p,fractionStd**2))
tStat = (fractionMean-p)/(fractionStd/np.sqrt(7))
# need also the sample variance of the fractions
print('t statistic for dominant characters: %3.4f'%tStat)
tpValue=t.cdf(abs(tStat),7)-t.cdf(-abs(tStat),7)
print('p value for t: %3.2f'%tpValue)
