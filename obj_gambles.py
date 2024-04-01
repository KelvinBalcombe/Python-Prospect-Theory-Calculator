# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:43:24 2016

@author: aes05kgb
This sets out the lot class which calculates all the relevent quantities
"""

from shortercuts import *

def sortc(x,n=0,ascending=True):
    x=pd.DataFrame(x)
    z=twodma(x.sort_values(by=n,ascending=ascending))
    return(z)  

#__________________________________________________________________
#equalize the dimensions of a set (by adding 0s with 0 probabilities)
def equalize(x,p):
    x=np.array(x)
    p=np.array(p)
    if rows(x) != rows(p):
       print("Payoff and probabilty matrices are not the same length")
    n=1
    try:
        for i in range(len(x)):
            if len(x[i]) >n:
                n=len(x[i])
        for i in range(len(x)):
            while len(x[i])<n:
                x[i]=np.append(x[i],0)
                p[i]=np.append(p[i],0) 
        x=np.array(list(x))
        p=np.array(list(p))
    except:
        pass
    return x,p
#__________________________________________________________________ 
#Removes repeated values of payoffs and merges them (for one lottery)
def compressone(x,p):
    x=twodmca(x,False)
    p=twodmca(p,False)
    #print(p)
    for i in range(cols(x)):
        for j in range(i+1,cols(x)):
            if x[:,i]==x[:,j]:
                x[:,j]=0
                p[:,i]=p[:,i]+p[:,j]
                p[:,j]=0
    return x,p
#__________________________________________________________________ 
#Compresses and equalises a set of 
def equalize_and_compress(x,p,compress=True) :
    x,p=equalize(x,p)
    if compress:
        
        x=twodmca(x,False)
        p=twodmca(p,False)
        
        for i in range(rows(x)):
            x[i,:],p[i,:]=compressone(x[i,:],p[i,:])
    return(x,p)
#__________________________________________________________________ 
    
def equalize_and_compress_and_sort(x,p,compress=True):
    #Sorts the lotteries in ascending order with  to payoffs
    x,p=equalize_and_compress(x,p,compress)# lotteries in rows
    #print(p)
    p=twodmca(p,False).T
    z=((np.sum(p,axis=0))-1.0)
    #print(z)
    test=twodmca([np.abs(z) <10**-10])
    #print(test)
    ntest=nobool(test)
    
    if np.all(test)==False:
       print("warning, one of the lotteries has probabilities that do not sum to one, the number of the lottery is below")
       for i in range(len(ntest)):
           if ntest[i]==0:
              print(i+1)
       #sys.exit()
       
    x=twodmca(x,False).T
    
    k=cols(p)
    listp_=[]
    listx_=[]
    s=np.shape(x)
    if s[0]==1:
       p=p.T
       x=x.T
       k=cols(p)
    
    for i in range(k):
        z=cc([p[:,[int(i)]],x[:,[int(i)]]])
        z=sortc(z,1)
        p_=z[:,[0]]
        x_=z[:,[1]]
        listp_.append(p_.T)
        listx_.append(x_.T)
            
    listp_=twodmca(np.squeeze(listp_),False)      
    listx_=twodmca(np.squeeze(listx_),False)
    #The following just gets the maximums and minimums of x and associated robs
    p=listp_
    x=listx_
    
    listxmin=[]
    listpmin=[]
    for i in range(rows(p)):
        j=0
        while p[i,j]==0:
            j=j+1
        listxmin.append(x[i,j]) 
        listpmin.append(p[i,j])
    
    listxmax=[] 
    listpmax=[]     
    for i in range(rows(p)):
        j=cols(p)-1
        try:
            while p[i,j]==0:
                j=j-1
        except:
            pass
        listxmax.append(x[i,j]) 
        listpmax.append(p[i,j])
        
    minx=np.array(listxmin)
    maxx=np.array(listxmax)    
    minp=np.array(listpmin)
    maxp=np.array(listpmax)   
    
    return(listx_,listp_,minx,maxx,minp,maxp)

    
#__________________________________________________________________ 
def domain(x):
    x=np.array(x)
    s=[x>=0]
    t=[x<0]
    pos=np.array(s).astype(int)
    neg=np.array(t).astype(int)
    return pos,neg
#__________________________________________________________________     
#Construct a utility Function (u) and its cetertainty equivalent (iu)
def u_function(alpha=1.0,beta=1.0,lamda=1.0,uform='power',usym=False,expscale=1):
    np.seterr(all='ignore')    
    if usym:
        alpha=beta 
    if uform=='power':
        def u(x):
            x=np.array(x)
            s,t=domain(x)
            u=(x*s)**alpha[0] -lamda*((np.abs(x)*t)**beta[0])
            return np.squeeze(u)
        def iu(u):
            u=np.array(u)
            s,t=domain(u)
            x=(np.abs(u)*s)**(1/alpha[0]) - ((np.abs(u)*t/lamda)**(1/beta[0]))
            return np.squeeze(x)    
        
    if uform=='exp': 
        talpha=alpha/expscale
        tbeta=beta/expscale
        u_=np.isclose(talpha[0],0)
        _u=np.isclose(tbeta[0],0)
        def u(x):
            x=np.array(x)
            s,t=domain(x)
            if u_:
                upper=x
            else:
                upper=(1-np.exp(-talpha[0]*x))/talpha[0]
                upper=np.nan_to_num(upper)
            if _u:
                lower=x
            else:
                lower=(1-np.exp(-tbeta[0]*x))/tbeta[0]                    
                lower=np.nan_to_num(lower)            
            u=s*upper+ lamda*t*lower
            
            return np.squeeze(u)
            
        def iu(u):
            #print(alpha[0])
            u=np.array(u)
            s,t=domain(u)
            if u_:
                upper=u
            else:
                upper=(-np.log(1-talpha[0]*u)/talpha[0])
                upper=np.nan_to_num(upper)
            if _u:
                lower=u
            else: 
                lower=(-np.log(1-tbeta[0]*u/lamda)/tbeta[0])
                lower=np.nan_to_num(lower)
            x=s*upper+t*lower
            
            return np.squeeze(x)  
            
        def iu(u):
            #print(alpha[0])
            u=np.array(u)
            s,t=domain(u)
            if u_:
                upper=u
            else:
                upper=(-np.log(1-talpha[0]*np.abs(u))/talpha[0])
                upper=np.nan_to_num(upper)
            if _u:
                lower=u
            else: 
                lower=(np.log(1+tbeta[0]*u/lamda)/tbeta[0])
                lower=np.nan_to_num(lower)
            x=s*upper+t*lower
            return np.squeeze(x)  

    return u,iu
#__________________________________________________________________     
#The cumulative distribution    
def cumdist(p):
    return np.cumsum(p,axis=1)      
#__________________________________________________________________    
#The inverse of the cumlative distribution   
def invcumdist(p):
    z=p.copy()
    x=np.diff(p,n=1,axis=1)
    z[:,1:]=x
    return(z)
#__________________________________________________________________ 
#The decumulative distribution    
def dcumdist(p):
    pd=np.ones([rows(p),cols(p)])
    b=(1-cumdist(p))
    pd[:,1:]=b[:,0:-1]
    return(pd)
#__________________________________________________________________     
#The inverse of the decumulative distribution    
def invdcumdist(p):
    x=np.ones([rows(p),cols(p)])
    z=np.abs(np.diff(p,n=1,axis=1))
    x[:,0:-1]=z
    h=1-((np.sum(z,axis=1)))
    x[:,cols(x)-1]=h
    return(x)
#__________________________________________________________________ 
def w_function(gama,delta,wform='k&t'):
    np.seterr(all='ignore')
    if wform=='k&t':
        def weight_pos(p):
            s=p>1
            p[s]=1
            f1=p**gama[0]
            f2=(p**gama[0] + (1-p)**gama[0])**(1/gama[0]);
            fp=f1/f2
            return fp
        def weight_neg(p):
            s=p>1
            p[s]=1
            f1=p**delta[0]
            f2=(p**delta[0] + (1-p)**delta[0])**(1/delta[0]);
            fn=f1/f2
            return fn
    elif wform=='power':
        def weight_pos(p):
            fp=p**gama[0]
            return fp
        def weight_neg(p):
            fn=p**delta[0]
            return fn
    elif wform=='prelecI':            
        def weight_pos(p):
            f1=(np.abs(np.log(p)))**gama[0]
            fp=np.exp(-f1)
            return fp
        def weight_neg(p):
            f1=(np.abs(np.log(p)))**delta[0]
            fn=np.exp(-f1)
            return fn
    elif wform=='prelecII':
        def weight_pos(p):
            f1=(gama[1]*np.abs(np.log(p))**gama[0])
            fp=np.exp(-f1)
            return fp
        def weight_neg(p):
            f1=(delta[1]*np.abs(np.log(p))**delta[0])
            fn=np.exp(-f1)
            return fn
    elif wform=='g&h':
        def weight_pos(p):
            f1=gama[1]*(p**gama[0])
            f2=(gama[1]*(p**gama[0]) + (1-p)**gama[0])
            fp=f1/f2
            return fp
        def weight_neg(p):
            f1=delta[1]*(p**delta[0])
            f2=delta[1]*(p**delta[0]) + (1-p)**delta[0]
            fn=f1/f2
            return fn
    elif wform=='beta':
        def weight_pos(p):
            alpha=gama[0]
            beta=gama[1]
            fp=sps.beta.cdf(p,alpha,beta)
            return fp
        def weight_neg(p):
            alpha=delta[0]
            beta= delta[1]
            fn=sps.beta.cdf(p,alpha,beta)
            return fn
        
    return weight_pos,weight_neg

#__________________________________________________________________ 

#__________________________________________________________________        
def mav(x,p):
    ex=np.sum(p*x,axis=1)         #Expected values
    exs=np.sum(p*(x**2.0),axis=1) #Expected squaredvalues
    va=exs-ex**2.0                #Variance
    sd=va**0.5                    #Standard deviation
    return ex,sd

def nobool(x):
    z=twodmca(np.squeeze(twodmca(x).astype(int)))
    return z   

#__________________________________________________________________     
class lot():
    def __init__(self,x=[-1,1],p=[.5,.5],alpha=1,beta=1,lamda=1,delta=[1,1],gama=[1,1],uform='power',wform='k&t'):
        self.x=x
        self.p=p
        self.alpha=alpha
        self.beta=beta
        self.lamda=lamda
        self.delta=delta
        self.gama=gama
        self.uform=uform
        self.wform=wform
    
    def _(self,compress=True):
        X,P,minx,maxx,minp,maxp=equalize_and_compress_and_sort(self.x,self.p,compress=compress)
        self.X=np.squeeze(X)
        self.P=np.squeeze(P)
        self.minx=float(minx)
        self.maxx=float(maxx)
        self.minp=float(minp)
        self.maxp=float(maxp)
        self.zp=list(zip(self.X,self.P))
        self.ev=0
        self.var=0
        for i in self.zp:
            self.ev+=i[0]*i[1]
        for i in self.zp:
            self.var+=i[1]*(i[0]-self.ev)**2
        self.std=np.sqrt(self.var)    
        return self

    #The value function
    def vf(self,usym=False,expscale=1):
        a=self.alpha
        b=self.beta
        l=self.lamda
        self.uf,self.iuf=u_function(alpha=[a],beta=[b],lamda=[l],uform=self.uform,usym=usym,expscale=expscale)
        return self
    
    #The probability Transformation
    def wf(self):
        self.wpos,self.wneg=w_function(self.gama,self.delta,wform=self.wform)
        return self
    
    def cddist(self,dom='gain'):
        p=twodma(self._().P).T
        wd=np.squeeze(dcumdist(p))
        wc=np.squeeze(cumdist(p))
        if dom=='gain':
            return self.wf().wpos(wd)
        else:
            return self.wf().wneg(wc)
            
    def W(self):
        c_loss=self.cddist(dom='loss')
        c_gain=self.cddist(dom='gain')
        w_gain=np.squeeze(invdcumdist(twodma(c_gain).T))
        w_loss=np.squeeze(invcumdist(twodma(c_loss).T))
        #print(w_gain)
        #print(w_loss)
        w=[]
        i=0
        for x in self._().X:
            #print(x)
            if x>=0:
                w+=[w_gain[i]]
            else:
                w+=[w_loss[i]]
            i=i+1
        return w
    
    def val(self):
        v=self.vf().uf(self._().X)
        p=self._().P
        w=self.W()
        #print(w)
        pt=0
        eu=0
        i=0
        for j in v:
            pt+=j*w[i]
            eu+=j*p[i]
            i=i+1
        self.ut_pt=pt
        self.ut_eu=eu
        return self
    
    def U(self):
        return self.vf().uf(self._().X)
    
    def u_eu(self):
        return self.val().ut_eu
    
    def u_pt(self):
        return self.val().ut_pt
    
    def ce_eu(self):
        return float(self.vf().iuf(self.u_eu()))
    
    def ce_pt(self):
        return float(self.vf().iuf(self.u_pt()))
    
    def summary(self,verbose=False):
        q=[self._().ev,self._().std,self.u_eu(),self.u_pt(),self.ce_eu(),self.ce_pt()]
        q=frame(q).T;
        q.columns=['ev','std','u_eu','u_pt','ce_eu','ce_pt']
        if verbose:
            print('ev: the expected value of the prospect')
            print('std: the standard deviation of the prospect')
            print('u_eu: the expected utility of the prospect')
            print('u_ut: the expected utility of the prospect under the PT capacities')
            print('ce_eu: the certainty equivalent of the prospect')
            print('ce_pt: the certainty equivalent of the prospect under the PT capacties')
        return q
    
    def graphp(self):
        return plotprob(self.gama,self.delta,self.wform)
    
    def graphu(self,lowerx=-10,upperx=10):
        return plotutility(self.alpha,self.beta,self.lamda,self.uform,usym=False,lowerx=lowerx,upperx=upperx)
    
    
    
def plotprob(gama,delta,wform):
    f1,f2=w_function(gama,delta,wform)
    P=np.linspace(.0,1,100)
    T=[]
    S=[]
    for p in P:
        dcum=f1(np.squeeze(dcumdist(twodma([(1-p),p]).T)))
        cum=f2(np.squeeze(cumdist(twodma([p,1-p]).T)))
        w_pos=np.squeeze(invdcumdist(twodma(dcum).T))
        w_neg=np.squeeze(invcumdist(twodma(cum).T))
        #print(w_pos)
        T+=[w_pos[1]]
        S+=[w_neg[0]]

    S=frame(S);S.index=P; S.columns=['Loss']; S.index.name='Probability of Higher (Absolute) Payoff'
    T=frame(T);T.index=P; T.columns=['Gain']
    Pr=frame(P);Pr.index=P; Pr.columns=['Prob']
    fig,ax=plt.subplots()
    S.plot(ax=ax,grid=True);
    T.plot(ax=ax,grid=True)
    Pr.plot(ax=ax,grid=True)   
    
def plotutility(alpha=1,beta=1,lamda=1,uform='power',usym=False,lowerx=-100,upperx=100):
    X=np.linspace(lowerx,upperx,100)
    uf,iuf=u_function(alpha=[alpha],beta=[beta],lamda=[lamda],uform=uform,usym=usym,expscale=1)
    U=[]
    for x in X:
        u=uf(x)
        U+=[u]    
    U=frame(U)
    U.index=X; U.index.name='Payoff'
    U.columns=['Utility']
    U.plot(grid=True)

def FOD(x1,p1,x2,p2,sod=1000):    
    x1=twodmca(x1)
    x2=twodmca(x2)
    p1=twodmca(p1)
    p2=twodmca(p2)    
        
    x=[]
    P1=[]
    P2=[]
    n1=rows(x1)-1
    n2=rows(x2)-1
    
    while n1>=0 or n2>=0:
        if (n1>=0 and n2>=0):
            if x1[0] < x2[0]:
                x,x1,n1=augment(x,x1)
                P1,p1,m1,=augment(P1,p1)
                P2=np.append(P2,0)
            elif x2[0] < x1[0]:
                x,x2,n2=augment(x,x2)
                P2,p2,m2,=augment(P2,p2)
                P1=np.append(P1,0)
            elif x2[0] == x1[0]:
                x,x2,n2=augment(x,x2)
                x1=twodmca(np.delete(x1,0))
                n1=n1-1
                P1,p1,m1,=augment(P1,p1)
                P2,p2,m2,=augment(P2,p2)
        elif n2>=0:
            x,x2,n2=augment(x,x2)
            P2,p2,m2,=augment(P2,p2)
            P1=np.append(P1,0)
        elif n1>=0:
            x,x1,n1=augment(x,x1)
            P1,p1,m1,=augment(P1,p1)
            P2=np.append(P2,0)
    x=twodmca(x,False)
    mx=np.max(x)
    mn=np.min(x)
    z=np.linspace(mn,mx,sod)##Accurate sod increase 5 to 500
    
    P1=twodmca(P1,False)
    P2=twodmca(P2,False)
    
    W1=twodmca(dcumdist(P1),False)
    W2=twodmca(dcumdist(P2),False)
    Q1=twodmca((cumdist(P1)),False)
    Q2=twodmca((cumdist(P2)),False)

    I1=np.zeros([rows(z),1])
    I2=np.zeros([rows(z),1])
    
    Z1=z.copy()
    Z2=z.copy()
    for j in range(rows(z)-1):
        for k in range(cols(x)-1):
            
            if z[j]<x[:,k+1] and z[j]>=x[:,k]:
                Z1[j]=Q1[:,k]
                Z2[j]=Q2[:,k]
                
    for j in range(rows(z)-1):    
        I1[j+1]=I1[j]+Z1[j+1]*(z[j+1]-z[j])
        I2[j+1]=I2[j]+Z2[j+1]*(z[j+1]-z[j])
    
    dominant1=True
    dominant2=True
    do1=False
    do2=False

    for i in range(rows(I1)):
        if I1[i] < I2[i]:
             do1=True
        if I2[i] < I1[i]:    
             do2=True
    
    for i in range(rows(I1)):             
        if I1[i] > I2[i]:
            do1=False
        if I2[i] > I1[i]:
            do2=False         

    for i in range(cols(W1)):
        
        if W1[:,i] < W2[:,i]:
            dominant1=False
        if W2[:,i] < W1[:,i]:    
            dominant2=False
     
            
    if dominant1==True and dominant2==True:
        dom="none"
    elif dominant1==False and dominant2==False:
        dom="none"    
    elif dominant1==True and dominant2==False:
        dom="One"
    elif dominant1==False and dominant2==True:
        dom="Two"
    
    if do1==True and do2==True:
        som="none"
    elif do1==False and do2==False:
        som="none"    
    elif do1==True and do2==False:
        som="One"
    elif do1==False and do2==True:
        som="Two"    
    return dom,som   

def augment(X,x):
    X=np.append(X,x[0])
    x=twodmca(np.delete(x,0))
    n=rows(x)-1
    return X,x,n

def compare(a,b):
    print('Lottery1 - Lottery2 \n',a.summary()-b.summary())
    print('\nDominance First Order, Second Order \n',FOD(a.X,a.P,b.X,b.P))
    return

#If you want to look at corresponding things like payoffs and probabilities etc for a given lottery
def lookup(thing1,thing2):
    z=[]
    for i in zip(thing1,thing2):
        z+=[i]
    z= frame(z)
    z.columns=[1,2]
    z=findex(z)
    return z