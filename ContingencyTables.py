exec(open('imports.py').read())
from math import factorial as fact
from scipy.stats import fisher_exact,chi2
# ==================================================
# The chi^2 evaluated for a 2x2 contingency table,
# assuming the measured margins, and approximating the
# variance of p1-p2 (Grenwood and Yule 1915)
def chi2Yule(a,b,c,d,N,m1,m2,n1,n2):
    return (a*d-b*c)**2*N/(m1*m2*n1*n2)
# ==================================================

# ==================================================
# Also chi^2 following the original definition 
# by Pearson, summing over 4 terms

def chi2Yates(a,b,c,d,N,m1,m2,n1,n2):
    return (np.absolute(a*d-b*c)-N/2)**2*N/(m1*m2*n1*n2)

def chi2Table(a,b,c,d,N,m1,m2,n1,n2):
    # calculate the expected values
    m1=a+c
    n1=a+b
    m2=b+d
    n2=c+d
    N=n1+n2

    E=[m1*n1/N,m2*n1/N,m1*n2/N,m2*n2/N] # expected values, from proportions
    m=[a,b,c,d] # measured values
    # Now the chi^2 is the sum over 4 terms, carried out explicitly
    result=0
    chiIndividual=np.zeros(4) # to keep track of each deviation
    for i in range(4):
        chiIndividual[i]=(m[i]-E[i])**2/E[i]
        result+=chiIndividual[i]
    return result,chiIndividual,E

# assuming a fixed value of p, from binomial distribution
def chi2TableFixedP(a,b,c,d,N,p):
	m1=a+c
	n1=a+b
	m2=b+d
	n2=c+d
	N=n1+n2

	E=[p*(1-p)*N,(1-p)*(1-p)*N,p*p*N,p*(1-p)*N] # expected values, from p
	m=[a,b,c,d] # measured values
	chiIndividual=np.zeros(4)
	result=0
	for i in range(4):
		chiIndividual[i]=(m[i]-E[i])**2/E[i]
		result+=chiIndividual[i]
	return result,chiIndividual,E
# this is Fisher's exact test, still assuming the observed margins
# it assumes n1<=n2 and m1<=m2 
def fisherExact(a,b,c,d,N,m1,m2,n1,n2):
    result = fact(m1)*fact(m2)*fact(n1)*fact(n2)/(fact(a)*fact(b)*fact(c)*fact(d)*fact(N))
    return result

def barnardExact(a,b,c,d,N,m1,m2,n1,n2,p):
    result = fact(n1)*fact(n2)/(fact(a)*fact(b)*fact(c)*fact(d))*(p**m1*(1-p)**m2)
    return result

# Calculate the probability of having a_i>=a_observed.
# Careful: the total number N and m1, and m2 are held fixed at observe values
# but changing a_observed->a_i means that also b and c must be changed
#   (a) a->a_i (a_i is the trial value of the distribution)
#   (b) b->b_i=n1-a_i 
#   (c) c->c_i=m1-a_i
#   (d) d->d_i=m2-b_i (or n2-c_i, should check that this is the same)


# ========================================================
# This is done already by scipy.stats.fisher_exact
# I am trying to understand its algorithm/logic
# Think of stepping a up to n1; accordingly
#   b steps down
#   c steps down
#   d steps up to keep n1, n2, m1 and m2 all fixed
def fisherExactCumulative(a,b,c,d,N,m1,m2,n1,n2):
    # First, find our how many values of "a" we have
    # Yates says there are min(m1,n1)+1 values of "a".
    # But I can't see how, if observed margins must be held fixed.
    # Line below is OK for the case of Table VIII, given the observed numbers
    #rangeA=np.arange(a-d,n1+1,dtype=int)

    # this is the general criterion, I think
    if n1<=m1:
        amax=n1
    if n1>m1:
        amax=m1
    rangeA=np.arange(0,amax+1,dtype=int)


    #index=np.arange(0,(n1+1)-a,dtype=int)
    probObserved=fisherExact(a,b,c,d,N,m1,m2,n1,n2)
    cumA=0 # cumulative probability 
    for ai in rangeA:
        bi=n1-ai    #b-index[i]
        ci=m1-ai    #c-index[i]
        di=m2-bi    #d+index[i]
        cumContrib=fisherExact(ai,bi,ci,di,N,m1,m2,n1,n2)
        pi=0.75 # THIS IS A TEST FOR NOW
        cumContribB=barnardExact(ai,bi,ci,di,N,m1,m2,n1,n2,pi)
# ========================================================================
# Note that the cumulative probability extends over all values of "a"
# that give more "extreme" probability than observed, not a>=a_observed.
# This seems to be the consensus in the literature (and scipy),
# Although not what Yates1984 says
        #if(cumContrib<=1): #probObserved): # testing that it is properly normalized
        if(cumContrib<=probObserved):
            cumA+=cumContrib
            print('[%d,%d,%d,%d], %e (Barnard: %e) %f ***'%(ai,bi, ci,di,cumContrib,cumContribB,cumContribB/cumContrib))
        else:
            print('[%d,%d,%d,%d], %e (Barnard: %e)'%(ai,bi, ci,di,cumContrib,cumContribB))
# ==========================================================================    

    return cumA # right now it should return 1
# =======================================================

# =================================================
# Python codes for the study of 2x2 contingency tables
# For the new material in the 3rd edition of the textbook
# Notice the requirement that m1<=m2 and n1<=n2 so that
# we can count all possibile tables properly

#problem 10.1
#Simple N=10 table, to illustrate
'''
a=2
b=3 
c=1
d=4
tabLabel='Problem 10.1'
'''
# ==========================================
'''
# problem 10.3 and Example 10.4 of textbook
a=1
b=4
c=5
d=0
tabLabel='Problem 10.3 and Example 10.4'
# Invert columns to satisfy m1<m2, equivalent table
a=0
b=5
c=4
d=1
# ==========================================
'''
# Table 2 of Greenwood and Yule (1915), also reported 
# by Fisher (1925, but inverted columns)
#tabLabel='Yule Table II'
#a=6759
#b=56
#c=11396
#d=272
# Table VIII, with smaller counts
#tabLabel='Yule Table VIII'
#a=105
#b=5
#c=88
#d=11

# ========================================
# Table XV of Greenwood and Yule
# Rearrange so that n1<n2 and m1<m2
tabLabel='Yule Table XV'
a=54
b=5
c=46
d=15

a=5
b=54
c=15
d=46
# ========================================
#tabLabel='Mendel'
#a=101
#b=32
#c=315
#d=108

#tabLabel='prob10.3'
#a=4
#b=1
#c=0
#d=5

M=[[a,b],[c,d]] # this is the measured table
# ===================================
# Evaluate margins of the 2x2 table
N=a+b+c+d
n1=a+b
n2=c+d
p1=a/n1
p2=b/n2
m1=a+c
m2=b+d

print('N=%d, m1=%d, m2=%d, n1=%d, n2=%d'%(N,m1,m2,n1,n2))
print('Assumed n1/N (fixed margins): %3.3f, m1/N: %3.3f'%(n1/N,m1/N))
print('Measured ratios: a/n1: %3.3f b/n1: %3.3f, c/n2=%3.3f'%(a/n1,b/n1,c/n2))
print('Expectation of a: %3.3f'%(n1/N * m1/N * N))
# =================================
# Report also the probability to exceed measured chi^2
# based on the "wrong" chi^2(3) and the correct chi^2(1)
CHI2YULE=chi2Yule(a,b,c,d,N,m1,m2,n1,n2)
print('Yule\'s chi^2 for %s = %4.4f (P=%3.2e (3) or P=%3.2e (1))'%(tabLabel,CHI2YULE,chi2.sf(CHI2YULE,3),chi2.sf(CHI2YULE,1)))

chiPearson,chiIndividual,E=chi2Table(a,b,c,d,N,m1,m2,n1,n2)
CHI2YATES=chi2Yates(a,b,c,d,N,m1,m2,n1,n2)

# example 10.5 with a fixed p=0.75
p=0.75 # parent binomial probability
CHI2P,CHI2PIndiv,EP=chi2TableFixedP(a,b,c,d,N,p)


print('Yates\' chi^2 for %s = %4.4f (P=%3.2e (3) or P=%3.2e (1))'%(tabLabel,CHI2YATES,chi2.sf(CHI2YATES,3),chi2.sf(CHI2YATES,1)))
print('Pearson\'s chi^2 for %s = %4.4f'%(tabLabel,chiPearson))
print('Measured data',M)
print('Expectations based on fixed margins',E)
print('Individual chi^2 contributions',chiIndividual)

CHI2PPval=chi2.sf(CHI2P,3)
print('(Chi2 with p=%3.2f: %3.3f, p-value: %3.3f)'%(p,CHI2P,CHI2PPval))
print('(Expected values for p=%3.2f)'%p,EP)

# ==== EXACT TEST ================
EXACT=1
if EXACT==1:

# For the exact test, need to find probability that a >= a_observed
# 1. find how many values of "a" there are, a=0,...,m1 if m1<=n1
# and a=0,...,n1 if m1>n1
    numberA=min(m1,n1)+1 # this should take care of it

    print('Fisher\'s exact probability for %s that a=%d: %f'%(tabLabel,a,fisherExact(a,b,c,d,N,m1,m2,n1,n2)))
    print('Fisher\'s exact cumulative probability for %s that p(a)<=p(%d): %e'%(tabLabel,a,fisherExactCumulative(a,b,c,d,N,m1,m2,n1,n2)))
    print(fisher_exact(M))


# ===============================================
# Also do testing on Mendel's Table 1.3


