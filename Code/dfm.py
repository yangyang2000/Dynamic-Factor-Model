from scipy.linalg import block_diag
#     from scipy.stats import zscore
import datetime
# seasonal component:
import numpy as np, pandas as pd
from scipy import eye, asarray, dot
from scipy.linalg import svd
from numpy import diag
from scipy.signal import detrend
def local_trend_component(p):
    return np.kron(np.triu(np.ones([2,2])), np.eye(p))
def seasonal_component(seasons, p):
    Ip = np.eye(p)
    x = np.kron(np.diag(np.ones(seasons-2), k = -1), Ip)
    y = np.kron(np.ones([1, seasons-1]), -Ip)
    x[:p, :] = y
    return x
def combine_components(listofcomponents):
    return block_diag(*listofcomponents)
def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    from scipy import sum
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in xrange(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return dot(Phi, R)
from numpy import dot
from numpy.linalg import pinv, norm,inv
class KF_DFM(object):
    def __init__(self,y,x,W,Sigma_u,Sigma_v,A,B,C,D, p=2, n = 'default',sig0=5):
        self.p = p
        if n == 'default':
            self.n = y.shape[1]
        else:
            self.n = n
        self.mu_0 = np.zeros([C.shape[0], 1])
        self.Sigma_0 = np.eye(C.shape[0])*sig0
        self.Sigma_u = Sigma_u
        self.Sigma_v = Sigma_v
        self.ll = None
        if len(y.shape)!=3:
            self.y = y.reshape([y.shape[0], y.shape[1], 1])
        else:
            self.y = y
        ymean = np.nanmean(y, axis = 0)
        self.yhat = self.y.copy()
        for i in range(y.shape[1]):
            self.yhat[np.isnan(y[:, i]), i] = ymean[i]
        try:
            if len(x.shape)!=3:
                self.x = x.reshape([x.shape[0], x.shape[1], 1])
            else:
                self.x = x
        except:
            self.x = None

        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def kfilter(self, returns = False):
        Hs=list()
        Ls=list()
        zs=list()
        Qs=list()
        if (self.D == None) | (np.count_nonzero(self.D)==0):
            z=dot(self.C,self.mu_0)
        else:
            z=dot(self.C,self.mu_0)+dot(self.D, self.W[0])
        Q=dot(self.C,dot(self.Sigma_0,self.C.T))+self.Sigma_v
        zs.append(z)
        Qs.append(Q)
        for t in range(len(self.y)):
            H_t_dot = dot(self.A.T, inv(dot(dot(self.A,Q), self.A.T)+self.Sigma_u))
            K_t_dot=dot(dot(self.C,Q),H_t_dot)
            L_t = self.C-dot(K_t_dot,self.A)
            if ((self.D == None) | (np.count_nonzero(self.D)==0))&((self.B == None) | (np.count_nonzero(self.B)==0)):
                z = dot(L_t,z)+dot(K_t_dot,self.yhat[t])
            elif (self.D == None) | (np.count_nonzero(self.D)==0):
                z = dot(L_t,z)+dot(K_t_dot,(self.yhat[t]-dot(self.B,self.x[t])))
            elif (self.B == None) | (np.count_nonzero(self.B)==0):
                z = dot(L_t,z)+dot(K_t_dot,self.yhat[t])+dot(self.D,self.w[t])
            else:
                z = dot(L_t,z)+dot(K_t_dot,(self.yhat[t]-dot(self.B,self.x[t])))+dot(self.D,self.w[t])
            Q=dot(dot(L_t,Q),self.C.T)+self.Sigma_v
            Hs.append(H_t_dot)
            Ls.append(L_t)
            zs.append(z)
            Qs.append(Q)

        self.Hs = Hs
        self.Ls = Ls
        self.zs = zs
        self.Qs = Qs
        yhat = np.array([dot(self.A, z) for i,z in enumerate(zs[1:])])
        self.yhat[np.isnan(self.y)] = yhat[np.isnan(self.y)]
        for c in range(self.y.shape[1]):
            maxy = np.nanmax(self.y[:,c,0])
            miny = np.nanmin(self.y[:,c,0])
            self.yhat[self.yhat[:,c,0]>maxy]= maxy
            self.yhat[self.yhat[:,c,0]<miny]= miny
        if returns == True:
            return self.zs, self.Qs
    def ksmoother(self, returns = False):
        zsmooth = list()
        zvarsmooth = list()
        zcovarsmooth = list()
        rs = list()
        r = np.zeros(self.zs[0].shape)
        R = np.zeros(self.Qs[0].shape)
        I = np.eye(self.zs[-1].shape[0])
        for t in range(len(self.y))[::-1]:
            H_t_dot = self.Hs[t]
            y_t = self.yhat[t]
            if self.x != None:
                x_t = self.x[t]
            z = self.zs[t]
            Q=self.Qs[t]
            L_t=self.Ls[t]
            if (self.B == None) | (np.count_nonzero(self.B)==0):
                r = dot(H_t_dot,y_t-dot(self.A,z))+dot(L_t.T,r)
            else:
                r = dot(H_t_dot,(y_t-dot(self.A,z)-dot(self.B,x_t)))+dot(L_t.T,r)
            R = dot(H_t_dot,self.A)+dot(dot(L_t.T, R), L_t)
            if t ==0:
                r0 = r
                R0 = R
            z_T = z+dot(Q,r)
            var_z_T = Q-dot(dot(Q,R),Q)
            if t>=1:
                s_t_1 = dot(dot((I-dot(Q,R)),L_t),self.Qs[t-1])
                zcovarsmooth.append(s_t_1)
            zsmooth.append(z_T)
            zvarsmooth.append(var_z_T)
            rs.append(r)
        z_T=self.mu_0+dot(dot(self.Sigma_0,self.C.T), r0)
        var_z_T=self.Sigma_0-dot(dot(dot(dot(self.Sigma_0,self.C.T),R0),self.C),self.Sigma_0)
        S_0 = dot(dot(I-dot(Q,R0), self.C),self.Sigma_0)
        zsmooth.append(z_T)
        zvarsmooth.append(var_z_T)
        zcovarsmooth.append(S_0)
        self.zsmooth = zsmooth[::-1]
        self.zvarsmooth = zvarsmooth[::-1]
        self.zcovarsmooth = zcovarsmooth[::-1]
        self.rs = rs[::-1]
        if returns == True:
            return self.zsmooth
    def maximize(self, compute_ll=False):
        A,B,C,D = self.A, self.B, self.C, self.D
        p = self.p
        seasons =int((self.C.shape[0]-p*2)/p+1)
        mu_0 = self.zsmooth[0]
        Sigma_0 = np.diag(np.diag(self.zvarsmooth[0]))
        A_alphagamma = np.concatenate([np.eye(p),
                                       np.zeros([p,p]),
                                       np.eye(p),
                                       np.zeros([p,(seasons-2)*p])],
                                      axis = 1) # level+season state opeartor

        Z_T_star = [dot(A_alphagamma,zT) for zT in self.zsmooth[1:]] # Get State combined state level

        # Compute loading factors
        if (self.B != None):
            # upper block
            v2_1 = np.concatenate([sum([dot(zt, zt.T)
                                        +dot(dot(A_alphagamma,self.zvarsmooth[i+1]),A_alphagamma.T)
                                        for i,zt in enumerate(Z_T_star)]),
                                   sum([dot(zt, self.x[i].T)
                                        for i,zt in enumerate(Z_T_star)])])
            # Bottom block
            v2_2 = np.concatenate([sum([dot(self.x[i], zt.T) for i,zt in enumerate(Z_T_star)]),
                                   sum([dot(self.x[i], self.x[i].T) for i,zt in enumerate(Z_T_star)])])
            v2 = np.concatenate([v2_1, v2_2], axis = 0)
        else:
            v2 = sum([dot(zt, zt.T)+dot(dot(A_alphagamma,self.zvarsmooth[i+1]),A_alphagamma.T)
                      for i,zt in enumerate(Z_T_star)])
        # Compute loading factors
        LBs = list()
        Ls = list()
        Bs = list()
        if (self.B != None):
            for h in range(len(self.yhat[0])):
                v1 = np.concatenate([sum([dot(self.yhat[t][h],zt.T)
                                          for i,zt in enumerate(Z_T_star)]),
                                     sum([dot(self.yhat[t][h],xt.T)
                                          for i,xt in enumerate(self.x)])], axis = 1)
                LB = dot(v1, v2)
                LBs.append(LB)
                Ls.append(LB[:p])
                Bs.append(LB[p:])
        else:
            for h in range(len(self.yhat[0])):
                v1 = sum([dot(self.yhat[t][h],zt.T)
                          for t,zt in enumerate(Z_T_star)])
                LB = dot(v1,inv(v2))
                LBs.append(LB)
            Ls = LBs

        # Sigma_u (observation covariance)
        Sig_u = list()
        for h in range(len(self.yhat[0])):
            if self.B!= None:
                sig_u_h = 1/len(self.y)*sum([dot((self.yhat[i][h]-dot(Ls[h],zt)-dot(Bs[h],self.x[i])),
                                              (self.yhat[i][h]-dot(Ls[h],zt)-dot(Bs[h],self.x[i])).T)
                                          +dot(dot(dot(Ls[h],A_alphagamma),self.zvarsmooth[i+1]),
                                              dot(Ls[h],A_alphagamma).T)
                                          for i,zt in Z_T_star])
            else:
                sig_u_h = 1/len(self.y)*sum([dot(self.yhat[i][h]-dot(Ls[h],zt),
                                              (self.yhat[i][h]-dot(Ls[h],zt)).T)
                                          +dot(dot(dot(Ls[h],A_alphagamma),self.zvarsmooth[i+1]),
                                              dot(Ls[h],A_alphagamma).T)
                                          for i,zt in enumerate(Z_T_star)])
            Sig_u.append(sig_u_h)
        Sig_u = np.diag(Sig_u)

        # State space covariances
        Ibetas = np.eye(p,len(self.zsmooth[1]))
        Sigma_eta_betas = list()
        for j in range(p):
            Cbeta = C[p+j,:]
            Ibeta = Ibetas[j]
            Sigma_eta_beta = 1/len(self.y)*sum([dot((zT[p+j]-dot(Cbeta, self.zsmooth[i])),
                                                    (zT[p+j]-dot(Cbeta, self.zsmooth[i])).T)
                                                +dot(dot(Ibeta, self.zvarsmooth[i+1]), Ibeta.T)
                                                +dot(dot(Cbeta, self.zvarsmooth[i]), Cbeta.T)
                                                -dot(dot(Ibeta, self.zcovarsmooth[i]), Cbeta.T)
                                                -dot(dot(Cbeta, self.zcovarsmooth[i]), Ibeta.T)
                                                for i,zT in enumerate(self.zsmooth[1:])])
            Sigma_eta_betas.append(Sigma_eta_beta)
        sigma_eta=np.diag(Sigma_eta_betas)


        Igammas = np.eye(seasons-1,len(self.zsmooth[1]))
        Sigma_xi_gammas = list()
        for j in range(p):
            Cgamma = C[-(seasons-1)+j,:]
            Igamma = Igammas[j]
            Sigma_xi_gamma = 1/len(self.y)*sum([dot((zT[-(seasons-1)+j]-dot(Cgamma, self.zsmooth[i])),
                                                    (zT[-(seasons-1)+j]-dot(Cgamma, self.zsmooth[i])).T)
                                                +dot(dot(Igamma, self.zvarsmooth[i+1]), Igamma.T)
                                                +dot(dot(Cgamma, self.zvarsmooth[i]), Cgamma.T)
                                                -dot(dot(Igamma, self.zcovarsmooth[i]), Cgamma.T)
                                                -dot(dot(Cgamma, self.zcovarsmooth[i]), Igamma.T) for i,zT in enumerate(self.zsmooth[1:])])
            Sigma_xi_gammas.append(Sigma_xi_gamma)
        sigma_xi=np.diag(Sigma_xi_gammas)

        sigma_epsilon  = np.eye(p) # level state covariances are assumed identity(p)
        ##### Clean up ######
        # Construct state covariance, Sigma_v
        SIG = block_diag(*[sigma_epsilon, sigma_eta, sigma_xi, np.zeros([(seasons-2)*p,(seasons-2)*p])])
        # Construct measurement matrix, A
        L = varimax(np.array(Ls),q=100)
        A = L
        for m in [np.zeros([self.n,p]), L, np.zeros([self.n,(seasons-2)*p])]:
            A = np.concatenate([A, m], axis = 1)

        # update model parameters
        self.Sigma_v = SIG
        self.sigma_eta = sigma_eta
        self.sigma_xi = sigma_xi
        self.L = L
        self.A = A
        self.Sigma_u = Sig_u
        self.mu_0 = mu_0
        self.Sigma_0 = Sigma_0
        if self.B != None:
            B = np.array(Bs)
            self.B = B
        if self.D != None:
            D = np.array(Ds)
            self.D = D

        # compute log likellihood
        if compute_ll == True:
            if (model.B!=None) &(model.D!=None):
                ll = sum([np.log(norm(Sigma_0)),
                         np.trace(dot(inv(Sigma_0),dot((model.zsmooth[0]-mu_0),(model.zsmooth[0]-mu_0).T)
                                      +model.zvarsmooth[0])),
                          len(model.yhat)*np.log(norm(SIG)),
                          np.trace(dot(inv(SIG),
                                      sum([dot((zt-dot(C,model.zsmooth[i])-dot(D,model.w[i])),
                                            (zt-dot(C,model.zsmooth[i])-dot(D,model.w[i])).T)
                                        +model.zvarsmooth[i+1]
                                        +dot(dot(C, model.zvarsmooth[i]), C.T)
                                        -dot(model.zcovarsmooth[i],C.T)
                                        -dot(C,model.zcovarsmooth[i])
                                        for i,zt in enumerate(model.zsmooth[1:])]))),
                          len(model.yhat)*np.log(norm(Sig_u)),
                          np.trace(dot(inv(Sig_u),
                                      sum([dot((model.yhat[i]-dot(A,zt)-dot(B,model.x[i])),
                                            (model.yhat[i]-dot(A,zt)-dot(B,model.x[i])).T)
                                        +dot(dot(A, model.zvarsmooth[i+1]), A.T)
                                        for i,zt in enumerate(model.zsmooth[1:])])))
                        ])
            elif (model.D!=None):
                ll = sum([np.log(norm(Sigma_0)),
                         np.trace(dot(inv(Sigma_0),dot((model.zsmooth[0]-mu_0),(model.zsmooth[0]-mu_0).T)
                                      +model.zvarsmooth[0])),
                          len(model.yhat)*np.log(norm(SIG)),
                          np.trace(dot(inv(SIG),
                                      sum([dot((zt-dot(C,model.zsmooth[i])-dot(D,model.w[i])),
                                            (zt-dot(C,model.zsmooth[i])-dot(D,model.w[i])).T)
                                        +model.zvarsmooth[i+1]
                                        +dot(dot(C, model.zvarsmooth[i]), C.T)
                                        -dot(model.zcovarsmooth[i],C.T)
                                        -dot(C,model.zcovarsmooth[i])
                                        for i,zt in enumerate(model.zsmooth[1:])]))),
                          len(model.yhat)*np.log(norm(Sig_u)),
                          np.trace(dot(inv(Sig_u),
                                      sum([dot((model.yhat[i]-dot(A,zt)),
                                            (model.yhat[i]-dot(A,zt)).T)
                                        +dot(dot(A, model.zvarsmooth[i+1]), A.T)
                                        for i,zt in enumerate(model.zsmooth[1:])])))
                         ])
            elif (model.B!=None):
                ll = sum([np.log(norm(Sigma_0)),
                         np.trace(dot(inv(Sigma_0),dot((model.zsmooth[0]-mu_0),(model.zsmooth[0]-mu_0).T)
                                      +model.zvarsmooth[0])),
                          len(model.yhat)*np.log(norm(SIG)),
                          np.trace(dot(inv(SIG),
                                      sum([dot((zt-dot(C,model.zsmooth[i])),
                                            (zt-dot(C,model.zsmooth[i])).T)
                                        +model.zvarsmooth[i+1]
                                        +dot(dot(C, model.zvarsmooth[i]), C.T)
                                        -dot(model.zcovarsmooth[i],C.T)
                                        -dot(C,model.zcovarsmooth[i])
                                        for i,zt in enumerate(model.zsmooth[1:])]))),
                          len(model.yhat)*np.log(norm(Sig_u)),
                          np.trace(dot(inv(Sig_u),
                                      sum([dot((model.yhat[i]-dot(A,zt)-dot(B,model.x[i])),
                                            (model.yhat[i]-dot(A,zt)-dot(B,model.x[i])).T)
                                        +dot(dot(A, model.zvarsmooth[i+1]), A.T)
                                        for i,zt in enumerate(model.zsmooth[1:])])))
                        ])
            else:
                ll = sum([np.log(norm(Sigma_0)),
                         np.trace(dot(inv(Sigma_0),dot((model.zsmooth[0]-mu_0),(model.zsmooth[0]-mu_0).T)
                                      +model.zvarsmooth[0])),
                          len(model.yhat)*np.log(norm(SIG)),
                          np.trace(dot(inv(SIG),
                                      sum([dot((zt-dot(C,model.zsmooth[i])),
                                            (zt-dot(C,model.zsmooth[i])).T)
                                        +model.zvarsmooth[i+1]
                                        +dot(dot(C, model.zvarsmooth[i]), C.T)
                                        -dot(model.zcovarsmooth[i],C.T)
                                        -dot(C,model.zcovarsmooth[i])
                                        for i,zt in enumerate(model.zsmooth[1:])]))),
                          len(model.yhat)*np.log(norm(Sig_u)),
                          np.trace(dot(inv(Sig_u),
                                      sum([dot((model.yhat[i]-dot(A,zt)),
                                            (model.yhat[i]-dot(A,zt)).T)
                                        +dot(dot(A, model.zvarsmooth[i+1]), A.T)
                                        for i,zt in enumerate(model.zsmooth[1:])])))
                        ])

            if self.ll!= None:
                print 'Change in ll {}'.format(ll-self.ll)
            else:
                print 'll: {}'.format(ll)
            self.ll = ll
    def EM(self, iters = 10, compute_ll=False, IterPrint = True):
        for it in range(iters):
            self.kfilter()
            self.ksmoother()
            self.maximize(compute_ll=compute_ll)
            if IterPrint == True:
                print 'Iteration: {}'.format(it)
