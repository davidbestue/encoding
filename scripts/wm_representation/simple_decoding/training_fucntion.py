
import numpy as np
import statsmodels.api as sm


def train_each_vxl( training_dataset, training_targets ):
    #
    ### X Training
    ## X matrix (intercept and spikes)
    X = np.column_stack([np.ones(np.shape(training_dataset)[0]),  training_dataset])
    ## Y (sinus and cos of the target)
    sinus =np.sin([np.radians(np.array(training_targets)[i]) for i in range(0, len(training_targets))])
    cosinus = np.cos([np.radians(np.array(training_targets)[i]) for i in range(0, len(training_targets))])
    Y = np.column_stack([cosinus, sinus])
    Y = Y.astype(float) #to make it work in the cluster
    X = X.astype(float)
    model = sm.OLS(Y, X)
    ##train the model
    weights = model.fit()  ## training_weights.params
    return weights



