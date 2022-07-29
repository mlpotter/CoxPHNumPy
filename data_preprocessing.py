import pandas as pd
import numpy as np
import os

def get_bladder(dataframe=False):
    """
    replicate dataset from Survival Analysis A Self-Learning Text 3rd Edition Springer
    num: initial number of tumors
    size: initial size of tumors in cm
    tx: 0=placebo, 1=thiotepa
    int: the kth event
    id: patient id
    start: survived up to t=start
    end: event happend at t=end
    :return: counting process format for bladder dataframe
    """
    folder = r"C:\Users\lpott\Desktop\recurrent_event\data"
    filename = "bladder.xlsx"
    filepath = os.path.join(folder,filename)
    bladder = pd.read_excel(filepath)
    # bladder = bladder[bladder.id.isin(np.arange(27))]
    bladder = bladder[bladder.treatment.isin(["placebo","thiotepa"])]
    bladder = bladder[bladder.enum.isin([1,2,3,4])]

    bladder.status = bladder.status.replace({3:0,2:0}) #bladder[bladder.status.isin([0,1])]
    bladder.treatment = bladder.treatment.replace({"placebo":0.0,"thiotepa":1.0},inplace=False).astype(float)

    bladder.drop(["recur","rtumor","rsize"],axis=1,inplace=True)

    bladder.rename(columns={"number":"num","treatment":"tx","enum":"int"},inplace=True)

    if dataframe:
        return bladder.loc[:, ["tx", "num", "size", "stop", "int", "status"]]

    X = bladder.loc[:,["tx","num","size"]]
    y = bladder.status
    time = bladder.stop
    recurrence = bladder.int
    return (X.values,y.values,time.values,recurrence.values)


if __name__ == "__main__":
    X,y,time,recurrence = get_bladder()


    bladder = get_bladder_lifelines()
    from lifelines import CoxPHFitter
    from lifelines import KaplanMeierFitter
    cph = CoxPHFitter()
    cph.fit(bladder,'stop',event_col='status',strata='int')
    cph.print_summary()

    print("Running data_preprocessing.py")