# Expectations for C and its variance from Kaastra 2017
def Ce(mu):
        Ce=0
        if (0 <= mu <= 0.5):
                Ce= -0.25*mu**3 +1.38*mu**2-2.0*mu*np.log(mu)
        if (0.5 < mu <= 2):
                Ce=-0.00335*mu**5+0.04259*mu**4 -0.27331*mu**3+1.381*mu**2-2*mu*np.log(mu)
        if (2 < mu <= 5):
                Ce=1.019275+0.1345*mu**(0.461-0.9*np.log(mu))
        if (5 < mu <= 10):
                Ce=1.00624+0.604*mu**(-1.68)
        if (mu>10):
                Ce=1.0+0.1649*mu**(-1)+0.226*mu**(-2)
        return Ce

def Cv(mu):
        Cv=0
        if (0 <= mu <= 0.1):
                Cv=0
                print("Warning this needs to be updated")
        if (0.1 < mu <= 0.2):
                Cv=-262*mu**4+195*mu**3-51.24*mu**2+4.34*mu+0.77005
        if (0.2 < mu <= 0.3):
                Cv=4.23*mu**2-2.8254*mu+1.12522
        if (0.3 < mu <= 0.5):
                Cv=-3.7*mu**3+7.328*mu**2-3.6926*mu+1.20641
        if (0.5 < mu <= 1):
                Cv=1.28*mu**4-5.191*mu**3+7.666*mu**2-3.5446*mu+1.15431
        if (1 < mu <= 2):
                Cv=0.1225*mu**4-0.641*mu**3+0.859*mu**2+1.091*mu-0.05748
        if (2 < mu <= 3):
                Cv=0.089*mu**3-0.872*mu**2+2.8422*mu-0.67539
        if (3 < mu <= 5):
                Cv=2.12366+0.012202*mu**(5.717-2.6*np.log(mu))
        if (5 < mu <= 10):
                Cv=2.05159+0.331*mu**(1.343-np.log(mu))
        if (10 < mu):
                Cv=12.0*mu**(-3)+0.79*mu**(-2)+0.6747*mu**(-1)+2.0
        return Cv

# define Gaussian function
def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

