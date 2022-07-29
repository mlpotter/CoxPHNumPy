import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

class CoxNonParametric(object):
    def __init__(self,F,max_events=4):
        self.max_events=max_events
        self.beta = np.random.randn(1,F)*0.01
        self.loglikelihood_ = -np.inf

    def score(self,X,y,time,recurrence):
        score = 0
        for k in range(1,self.max_events+1):
            # get each likelihood component of the cox-partial likelihood for recurrent event k
            Xk = X[recurrence==k]; yk = y[recurrence==k]; timek = time[recurrence==k];
            N, F = Xk.shape
            for i in np.nonzero(yk)[0]:
                # if the individual is censored then he is not part of the cox-partial likelihood

                # individual i being considered for cox-partial likelihood term Li
                x = Xk[[i],:];

                # risk set includes all individuals alive up to time end AND all individuals who die at time end
                risk_set = (timek >=  timek[i])

                # the features of the people in risk set
                X_risk = Xk[risk_set]

                # the beta x X' term in the exponential term
                betaX = np.exp(np.matmul(self.beta, X_risk.T).T)

                # the score (which is the derivative of the likelihood for individual i
                score += (x - (X_risk * betaX / betaX.sum()).sum(axis=0))

        return score

    def loglikelihood(self,X,y,time,recurrence):
        loglikelihood = 0
        for k in range(1,self.max_events+1):
            # get the likelihood for the recurrent event k
            Xk = X[recurrence==k]; yk = y[recurrence==k]; timek = time[recurrence==k];

            N,F = Xk.shape
            for i in np.nonzero(yk)[0]:
                x = Xk[[i],:];
                # risk set includes all individuals alive up to time end AND all individuals who die at time end
                risk_set = (timek >= timek[i])
                X_risk = Xk[risk_set]

                loglikelihood += (np.matmul(self.beta,x.T)) - np.log(np.exp(np.matmul(self.beta, X_risk.T)).sum())

        return loglikelihood

    def fit(self,X,y,time,recurrence,lr=0.1,iterations=1000):
        likelihoods = []
        for iter in tqdm(range(iterations)):
            self.beta += lr*self.score(X,y,time,recurrence)/(y.sum())
            likelihoods.append(self.loglikelihood(X,y,time,recurrence).item())
        self.loglikelihood_ = likelihoods[-1]
        return likelihoods

    def se(self,X,y,time,recurrence):
        cov = 0
        for k in range(1, self.max_events + 1):
            # get each likelihood component of the cox-partial likelihood for recurrent event k
            Xk = X[recurrence == k];
            yk = y[recurrence == k];
            timek = time[recurrence == k];
            N, F = Xk.shape
            for i in np.nonzero(yk)[0]:
                # if the individual is censored then he is not part of the cox-partial likelihood

                # individual i being considered for cox-partial likelihood term Li
                x = Xk[[i], :];

                # risk set includes all individuals alive up to time end AND all individuals who die at time end
                risk_set = (timek >= timek[i]);

                # the features of the people in risk set
                X_risk = Xk[risk_set]

                # the beta x X' term in the exponential term
                betaX = np.exp(np.matmul(self.beta, X_risk.T).T)

                # the score (which is the derivative of the likelihood for individual i
                scorei = (x - (X_risk * betaX / betaX.sum()).sum(axis=0))
                cov += scorei.T@scorei
        return np.sqrt(1/np.diag(cov))

    def fit_base(self,X,y,time,recurrence):
        # we use the Kaplan-Meir Curve for each stratified event to get base survival function
        self.base_survival = {}
        self.base_hazard = {}
        for k in range(1, self.max_events + 1):
            # get each likelihood component of the cox-partial likelihood for recurrent event k
            Xk = X[recurrence == k]; yk = y[recurrence == k]; timek = time[recurrence == k];
            N, F = Xk.shape

            sort_idx = np.argsort(timek)
            Xk = Xk[sort_idx]; yk = yk[sort_idx]; timek = timek[sort_idx]

            old_bucket = pd.Interval(left=0,right=timek[0],closed="left") #f"({0},{timek[0]}]"
            self.base_survival[k] = {old_bucket:1}
            unique_times = np.unique(timek[yk==1])
            for (t1,t2) in zip(unique_times[:-1],unique_times[1:]):
                bucket = pd.Interval(left=t1,right=t2,closed="left") #f"({t1},{t2}]"
                risk_set = (timek >= t1);
                dead_now = ((timek == t1) & (yk==1)).sum(); lived = risk_set.sum();
                self.base_survival[k][bucket] = (1 - dead_now/lived) * self.base_survival[k][old_bucket]
                old_bucket = bucket

            intervals, values = zip(*self.base_survival[k].items())

            # fit the base hazard function from the base survival function using relationship of s(t) = e^(-CUMHAZARD(t))
            # and taking differences between intervals
            self.base_hazard[k] = dict(zip(intervals,np.diff(-np.log((1,)+values))))

if __name__ == "__main__":
    from data_preprocessing import get_bladder
    X,y,time,recurrence = get_bladder()
    N,F = X.shape

    print("Testing Model Functions")
    model = CoxNonParametric(F)
    model.fit_base(X,y,time,recurrence)
    model.fit(X,y,time,recurrence,iterations=10)
    model.se(X,y,time,recurrence)
    model.loglikelihood(X,y,time,recurrence).item()
