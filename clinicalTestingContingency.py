exec(open('imports.py').read())
from math import factorial as fact

# describe the positive predictive value (posterior)
# as a function of sensitivity (SE), specificity (SP) and prevalence

# positive and negative likelihood ratios
def LR12n(SE, SP, n):
    result=(SE/(1-SP))**n
    return result

def LR21n(SE, SP, n):
    result=((1-SE)/SP)**n
    return result
# ---------------------------------------

def PPV(SE, SP, PR):
    result=SE*PR/(SE*PR+(1-SP)*(1-PR))
    return result

# let's try PPV after two positive tests
def PPV2(SE, SP, PR):
    #result = SE**2*PR/(SE**2*PR + (1-SP)*(1-PR)*(SE*PR+(1-SP)*(1-PR)))
    PPV1=PPV(SE, SP,PR) # this is PPV after one test
    result=SE*PPV1/(SE*PPV1+(1-SP)*(1-PPV1))
    return result

# Let's generalize the PPV after n repeated tests
def PPVn(SE, SP, PR, n):
    ppv=np.zeros(n+1)
    ppv[0]=PR # is this right?
    for i in range(1,n+1):
        ppv[i]=PPV(SE, SP,ppv[i-1])
    return ppv[n]


# test that this is equivalent to the formua in the Zhou book
def PPVnLR(SE, SP, PR, n):
    lr=LR12n(SE, SP, n) # first the likelihood ratio
    result=1/(1+(1-PR)/(PR*lr))
    return result

def NPV(SE, SP, PR):
    result=SP*(1-PR)/(SP*(1-PR)+(1-SE)*PR) 
    return result

def NPVn(SE, SP, PR, n):
    npv=np.zeros(n+1)
    pr=np.zeros(n+1) # prevalence, updated by successive negative tests
    pr[0]=PR # is this right? Same as the PPVn function
    for i in range(1,n+1):
        npv[i]=NPV(SE, SP,pr[i-1])
        pr[i]=1-npv[i] # this is the new prevalence
    return npv[n]

def NPVnLR(SE, SP, PR, n):
    lr=LR21n(SE, SP, n) # likelihood ratio
    result=1/(1+PR*lr/(1-PR)) # notice lr at different location than for PPVnLR
    return result


# make the contingency table by specifying N
# question is: how to ensure that PP, SE and SP are consistent with one another??
# I think that prevalence is needed too
'''
def makeTable(SE, SP, PP, PR, N):
# USING PRevalence; beware that both SE and SP can't be used cnsistently
#    n1=PP*N
#    n2=N-n1 # n1 and n2 will always be positive.
#    m1=N*PR
#    m2=N*(1-PR)
# NOT using PRevalence
    n1=PP*N
    n2=N-n1
    m1=N*(1-PP-SP)/(1-SE-sp)
    m2=N-m1
    a=m1*SE
    b=n1-a
    c=n2-m2*SP
    d=n2-c
    prevalence=m1/N
    return(a,b,c,d,n1,n2, m1,m2,prevalence)


def makeTablePR(SE, SP, PP,PR,N): 
    # Let's analyze in detail the ORDER in which quantities are detemined.
    n1=PP*N
    n2=N-n1
    a=n1*PPV(SE,SP,PR)
    b=n1-a
    # m1=PR*N  # was this also for the calibration test?
    #m2=N-m1
    # a=m1*SE was this for the calibration test?
    d=n2*NPV(SE,SP,PR)
    c=N-a-b-d
    m1=a+c
    m2=b+d
    estimatedPrevalence=m1/N # is this even meaningful for these data?!
    # I don't think one can estimate prevalence from a sample, only
    # from a calibration set with a Golden Standard 
    return(a,b,c,d,n1,n2, m1,m2,estimatedPrevalence)

# What is the point of "making" a contingency table from a test?
# We don't have the Golden Standard in the sample data.
# We have assumed a prevalence, and we don't get that prevalence back
# if we use the PPV() and NPV() functions, using Bayes' theorem

SP=0.55
SE=0.65
PP=0.55
PR=0.1

# test calculations of PPV
print('PPV for SPecificity=%3.2f, SEnsitivity=%3.2f and PRevalence=%3.2f: %f'%(SP,SE,PR,PPV(SE,SP,PP)))

# test my PPV formula  vs. the one in the Zhou book
# looks like they are equivalent - but not fully tested
test1=PPVn(SE, SP, PR, 3)
test2=PPVnLR(SE, SP, PR, 3)
print('PPV=%3.4f PPVnLR=%3.4f'%(test1,test2))
test1=NPVn(SE, SP, PR, 3)
test2=NPVnLR(SE, SP, PR, 3)
print('NPV=%3.4f NPVnLR=%3.4f'%(test1,test2))

N=100
#print('PP+SP=%3.4f, check PP<SE and PP+SP>1 to have m1>0 (PP=%3.2f, SE=%3.2f)'%(PP+SP,PP, SE))

print(makeTablePR(SE, SP, PP, PR,N))
'''

#problem 10.8 - use a golden-standard test to calibrate a diagnostic test
print('Problem 10.8')
a=120 # true positives
b=20  # false positives
c=40 # true negatives
d=100  # false negatives

N=a+b+c+d
m1=a+c # people with disease
n1=a+b # people who test positive
m2=N-m1
n2=N-n1
SE=a/m1
SP=d/m2
print('Using calibration table %d, %d, %d, %d'%(a,b,c,d))
print('N=%d, SE=%2.3f, SP=%2.3f, PLR=%3.3f, NLR=%3.3f'%(N, SE, SP, SE/(1-SP),(1-SE)/SP))

#problem 10.9 - Now use a diagnostic test with same SE and SP as prob10.8
# 
print('Problem 10.9')
PR1=0.1 # 10 % prevalence
ppv1=PPV(SE,SP,PR1)
PR2=0.01 # 10 % prevalence
ppv2=PPV(SE,SP,PR2)
print('PPV= %3.2f (PR=%2.3f), %3.2f (PR=%3.3f)'%(ppv1,PR1,ppv2,PR2))
npv1=NPV(SE,SP,PR1)
print('NPV = %3.3f (PR=%3.3f)'%(npv1,PR1))

#problem 10.10 - flying in the age of covid-19
print('Problem 10.10')
SE=0.7
SP=0.99
PR=0.10
npv1=NPV(SE,SP,PR) # prob. of being healthy, after 1 test
npv3=NPVn(SE,SP,PR,3) # after 3 tests
print('For PR=%3.2f 1 test NPV: %3.3f (therefore %3.3f), 3 tests %3.3f (therefore %3.3f)'%(PR,npv1,1-npv1,npv3,1-npv3))
# what if PR=0.5??
PR=0.50
npv1=NPV(SE,SP,PR) # prob. of being healthy, after 1 test
npv3=NPVn(SE,SP,PR,3) # after 3 tests
print('For PR=%3.2f 1 test NPV: %3.3f (therefore %3.3f), 3 tests %3.3f (therefore %3.3f)'%(PR,npv1,1-npv1,npv3,1-npv3))
print('problem 10.11')
# Now see if a positive test should deny boarding on a plane
PR=0.01
ppv=PPV(SE,SP,PR)
print('For PR=%3.2f 1 test PPV: %3.3f'%(PR,ppv))
quit()

# Below here to reproduce figures in the textbook 
# ==========================================
# 2. Now plot  PPV and NPV as a function of its parameters
#pr is prevalence
NPoints=11
se=np.linspace(0.0,1.0,NPoints)#0.99999,NPoints)
sp=np.linspace(0.0,1.0,NPoints)#0.99999,NPoints)
pr=np.linspace(0.0,1.0,NPoints*10) # more points for prevalence

PPVFixedSE=np.zeros((NPoints*10,NPoints))
PPVFixedSP=np.zeros((NPoints*10,NPoints))
NPVFixedSE=np.zeros((NPoints*10,NPoints))
NPVFixedSP=np.zeros((NPoints*10,NPoints))
# 2.1 for a fixed sensitivity, pr vs. PPV 
fixedSE=0.5
fixedSP=0.5


for j in range(NPoints): # j: specificity or sensitivity
    for i in range(NPoints*10):    # i: prevalence
        PPVFixedSE[i,j]=PPV(fixedSE,sp[j],pr[i])
        NPVFixedSE[i,j]=NPV(fixedSE,sp[j],pr[i])
        PPVFixedSP[i,j]=PPV(se[j],fixedSP,pr[i])
        NPVFixedSP[i,j]=NPV(se[j],fixedSP,pr[i])


# plot PPV versus prevalence, as a function of specificity
fig,ax=plt.subplots(figsize=(8,6))
for j in range(NPoints): # j: specificity or sensitivity, alpha=1 for SP=1 or SE=1
    plt.plot(pr,PPVFixedSE[:,j],linewidth=2+0.1*j,alpha=0.5+0.05*j,color='black')
    plt.plot(pr,PPVFixedSP[:,j],linewidth=2+0.1*j,alpha=0.5+0.05*j,linestyle='--',color='black')

# plot the last curves for the legend on them
plt.plot(pr,PPVFixedSE[:,NPoints-1],linewidth=2+0.1*j,alpha=0.5+0.05*j,color='black',label='Fixed SE=%2.1f, SP=%2.1f-%2.1f'%(fixedSE,sp[0],sp[NPoints-1]))
plt.plot(pr,PPVFixedSP[:,NPoints-1],linewidth=2+0.1*j,alpha=0.5+0.05*j,linestyle='--',color='black',label='Fixed SP=%2.1f, SE=%2.1f-%2.1f'%(fixedSP,se[0],se[NPoints-1]))
# --------------------------------------------
plt.legend(loc=4,prop={'size': 12})
plt.xlabel('Prevalence')
plt.ylabel('Positive Predictive Value')
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
plt.grid(which='both')
eps=0.01
plt.xlim(0-eps,1+eps)
plt.ylim(0-eps,1+eps)
plt.savefig('ppv.pdf')

# plot PPV versus prevalence, as a function of specificity
fig,ax=plt.subplots(figsize=(8,6))
for j in range(NPoints): # j: specificity or sensitivity
    plt.plot(pr,NPVFixedSE[:,j],linewidth=2+0.1*j,alpha=0.5+0.05*j,color='black')
    plt.plot(pr,NPVFixedSP[:,j],linewidth=2+0.1*j,alpha=0.5+0.05*j,linestyle='--',color='black')
#plt.legend(loc=4,prop={'size': 10})
# plot the last curves for the legend on them
plt.plot(pr,NPVFixedSE[:,NPoints-1],linewidth=2+0.1*j,alpha=0.5+0.05*j,color='black',label='Fixed SE=%2.1f, SP=%2.1f-%2.1f'%(fixedSE,sp[0],sp[NPoints-1]))
plt.plot(pr,NPVFixedSP[:,NPoints-1],linewidth=2+0.1*j,alpha=0.5+0.05*j,linestyle='--',color='black',label='Fixed SP=%2.1f, SE=%2.1f-%2.1f'%(fixedSP,se[0],se[NPoints-1]))
# -------------------------------------------
plt.xlabel('Prevalence')
plt.ylabel('Negative Predictive Value')
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
plt.grid(which='both')
plt.legend(loc=3,prop={'size': 12})
plt.xlim(0-eps,1+eps)
plt.ylim(0-eps,1+eps)

plt.savefig('npv.pdf')

# ======== Now one more figure for a typical COVID-19 test
# case a: high sensitivity, medium specificity
PPV19a=np.zeros(NPoints*10)
NPV19a=np.zeros(NPoints*10)
fixedSE19a=0.95
fixedSP19a=0.5
# case b: medium sensitivity, high specificity
PPV19b=np.zeros(NPoints*10)
NPV19b=np.zeros(NPoints*10)
fixedSE19b=0.5
fixedSP19b=0.95
# case c: a useless test with 50% sens. and spec.
PPV19c=np.zeros(NPoints*10)
NPV19c=np.zeros(NPoints*10)
# ----------------------------------------------
for i in range(NPoints*10):
    PPV19a[i]=PPV(fixedSE19a,fixedSP19a,pr[i])
    NPV19a[i]=NPV(fixedSE19a,fixedSP19a,pr[i])
    PPV19b[i]=PPV(fixedSE19b,fixedSP19b,pr[i])
    NPV19b[i]=NPV(fixedSE19b,fixedSP19b,pr[i])
# also for 50% SP and 50% SE
    PPV19c[i]=PPV(fixedSE19b,fixedSP19a,pr[i])
    NPV19c[i]=NPV(fixedSE19b,fixedSP19a,pr[i])
fig,ax=plt.subplots(figsize=(8,6))
plt.plot(pr,PPV19a,color='blue',linewidth=2,label='PPV (SE=%2.2f, SP=%2.2f)'%(fixedSE19a,fixedSP19a))
plt.plot(pr,NPV19a,color='blue',linestyle='--',linewidth=2,label='NPV (SE=%2.2f, SP=%2.2f)'%(fixedSE19a,fixedSP19a))
plt.plot(pr,PPV19b,color='red',linewidth=2,label='PPV (SE=%2.2f, SP=%2.2f)'%(fixedSE19b,fixedSP19b))
plt.plot(pr,NPV19b,color='red',linewidth=2,linestyle='--',label='NPV (SE=%2.2f, SP=%2.2f)'%(fixedSE19b,fixedSP19b))
plt.plot(pr,PPV19c,color='black',alpha=0.5,linewidth=2,linestyle='-',label='PPV (SE=%2.2f, SP=%2.2f)'%(fixedSE19b,fixedSP19a))
plt.plot(pr,NPV19c,color='black',alpha=0.5,linewidth=2,linestyle='--',label='NPV (SE=%2.2f, SP=%2.2f)'%(fixedSE19b,fixedSP19a))

ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
plt.xlabel('Prevalence')
plt.ylabel('Predictive Value')
plt.grid(which='both')
plt.legend(loc=8,prop={'size': 10})
plt.xlim(0-eps,1+eps)
plt.ylim(0-eps,1+eps)
plt.savefig('ppvnpv19.pdf')

# ==================================================
# ==================================================
# Compare the PPV from one and 2 positive tests, 
#fixedSE19a=0.95
#fixedSP19a=0.5

# This is a test with a small PPV 
fixedSE19a=0.95
fixedSP19a=0.6


PPV219a=np.zeros(NPoints*10)
NPV219a=np.zeros(NPoints*10)
PPV319a=np.zeros(NPoints*10)
NPV319a=np.zeros(NPoints*10)

for i in range(NPoints*10):
    PPV19a[i]=PPV(fixedSE19a,fixedSP19a,pr[i])
#    PPV219a[i]=PPV2(fixedSE19a,fixedSP19a,pr[i])
    PPV219a[i]=PPVnLR(fixedSE19a,fixedSP19a,pr[i],2)
    PPV319a[i]=PPVnLR(fixedSE19a,fixedSP19a,pr[i],3)

    NPV19a[i]=NPV(fixedSE19a,fixedSP19a,pr[i])
#    NPV219a[i]=NPV2(fixedSE19a,fixedSP19a,pr[i])
    NPV219a[i]=NPVnLR(fixedSE19a,fixedSP19a,pr[i],2)
    NPV319a[i]=NPVnLR(fixedSE19a,fixedSP19a,pr[i],3)

fig,ax=plt.subplots(figsize=(8,6))
plt.plot(pr,PPV19a,color='darkblue',linewidth=2,label='PPV (SE=%2.2f, SP=%2.2f)'%(fixedSE19a,fixedSP19a))
plt.plot(pr,PPV219a,color='blue',linewidth=2,label='PPV(2)')
plt.plot(pr,PPV319a,color='cornflowerblue',linewidth=2,label='PPV(3)')

plt.plot(pr,NPV19a,color='red',linewidth=2,label='NPV')
plt.plot(pr,NPV219a,color='darkorange',linewidth=2,label='NPV(2)')
plt.plot(pr,NPV319a,color='orange',linewidth=2,label='NPV(3)')


plt.xlabel('Prevalence')
plt.ylabel('Predictive Value')
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
plt.grid(which='both')
plt.legend(loc=4,prop={'size': 9})
plt.xlim(0-eps,1+eps)
plt.ylim(0-eps,1+eps)

PR=0.25 # this is the prevalence, or initial prevalence


# we need to illustrate this plot with an example
# start at PR prevalence
PPV1=PPV(fixedSE19a,fixedSP19a,PR) # PPV from one positive test
PPV2=PPVn(fixedSE19a,fixedSP19a,PR,2)#PPV1)
PPV3=PPVn(fixedSE19a,fixedSP19a,PR,3)
PPV4=PPVn(fixedSE19a,fixedSP19a,PR,4)
plt.vlines(PR,0,1,linestyle='--',color='black')
plt.hlines(PPV1,0,1,linestyle='--',color='black')
plt.vlines(PPV1,0,1,linestyle='--',color='black')
plt.vlines(PPV2,0,1,linestyle='--',color='black')
plt.hlines(PPV2,0,1,linestyle='--',color='black')
plt.hlines(PPV3,0,1,linestyle='--',color='black')
#plt.hlines(PPV4,0,1,linestyle='--',color='black')
plt.annotate('A',xy=(PR,PPV1),xytext=(-15,2),textcoords='offset points')
plt.annotate('B',(PR,PPV2),xytext=(-15,2),textcoords='offset points')
plt.annotate('B\'',(PPV1,PPV2),xytext=(-19,2),textcoords='offset points')
plt.annotate('C',(PR,PPV3),xytext=(-15,2),textcoords='offset points')
plt.annotate('C\'',(PPV2,PPV3),xytext=(-19,2),textcoords='offset points')

plt.savefig('ppv2-19.pdf')

# ----------------------------------------------------------
# now also for two negative tests, see how the NPV increases

# This is a test with a small NPV; see what additional
# negative tests will do to improve the NPV

fixedSE19a=0.6
fixedSP19a=0.95

for i in range(NPoints*10):
    PPV19a[i]=PPV(fixedSE19a,fixedSP19a,pr[i])
#    PPV219a[i]=PPV2(fixedSE19a,fixedSP19a,pr[i])
    PPV219a[i]=PPVn(fixedSE19a,fixedSP19a,pr[i],2)
    PPV319a[i]=PPVn(fixedSE19a,fixedSP19a,pr[i],3)

    NPV19a[i]=NPV(fixedSE19a,fixedSP19a,pr[i])
#    NPV219a[i]=NPV2(fixedSE19a,fixedSP19a,pr[i])
    NPV219a[i]=NPVn(fixedSE19a,fixedSP19a,pr[i],2)
    NPV319a[i]=NPVn(fixedSE19a,fixedSP19a,pr[i],3)


fig,ax=plt.subplots(figsize=(8,6))
plt.plot(pr,PPV19a,color='darkblue',linewidth=2,label='PPV (SE=%2.2f, SP=%2.2f)'%(fixedSE19a,fixedSP19a))
plt.plot(pr,PPV219a,color='blue',linewidth=2,label='PPV(2)')
plt.plot(pr,PPV319a,color='cornflowerblue',linewidth=2,label='PPV(3)')

plt.plot(pr,NPV19a,color='red',linewidth=2,label='NPV')
plt.plot(pr,NPV219a,color='darkorange',linewidth=2,label='NPV(2)')
plt.plot(pr,NPV319a,color='orange',linewidth=2,label='NPV(3)')


plt.xlabel('Prevalence')
plt.ylabel('Predictive Value')
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
plt.grid(which='both')
plt.legend(loc=3,prop={'size': 9})
plt.xlim(0-eps,1+eps)
plt.ylim(0-eps,1+eps)

PR=0.75 # this is the prevalence, or initial prevalence

# we need to illustrate this plot with an example
# start at PR prevalence
NPV1=NPV(fixedSE19a,fixedSP19a,PR) # NPV from one positive test
NPV2=NPVn(fixedSE19a,fixedSP19a,PR,2)
NPV3=NPVn(fixedSE19a,fixedSP19a,PR,3)
plt.vlines(PR,0,1,linestyle='--',color='black')
plt.hlines(NPV1,0,1,linestyle='--',color='black')
plt.vlines(1-NPV1,0,1,linestyle='--',color='black')
plt.vlines(1-NPV2,0,1,linestyle='--',color='black')
plt.hlines(NPV2,0,1,linestyle='--',color='black')
plt.hlines(NPV3,0,1,linestyle='--',color='black')

plt.annotate('A',(PR,NPV1),xytext=(1,2),textcoords='offset points')
plt.annotate('B',(PR,NPV2),xytext=(1,2),textcoords='offset points')
plt.annotate('B\'',(1-NPV1,NPV2),xytext=(1,2),textcoords='offset points')
plt.annotate('C',(PR,NPV3),xytext=(1,2),textcoords='offset points')
plt.annotate('C\'',(1-NPV2,NPV3),xytext=(1,2),textcoords='offset points')

plt.savefig('npv2-19.pdf')

