# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import matplotlib.pyplot as plt

from data_preprocessing import get_bladder
from models import CoxNonParametric

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X,y,time,recurrence = get_bladder()
    N,F = X.shape

    model = CoxNonParametric(F)
    model.fit_base(X,y,time,recurrence)

    likelihoods = model.fit(X,y,time,recurrence,iterations=1400)
    se = model.se(X,y,time,recurrence).tolist()
    coef = model.beta[0].tolist()
    summary_df = pd.DataFrame({"coef":coef,"SE":se})
    print(summary_df)

    print("-2*LOG L ={:.4f}".format(-2*model.loglikelihood_))
    plt.plot(range(len(likelihoods)),likelihoods)
    plt.xlabel("Iterations"); plt.ylabel("-LL(X)")
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
