
import numpy as np
import pandas as pd


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    #return np.rad2deg((ang1 - ang2) % (2 * np.pi))
    return abs( np.rad2deg(ang1-ang2))



def test_wm(testing_activity, testing_behaviour, weights, nscans_wm, TR, Subject, Brain_region, condition ):
    df=[]
    testing_angles = np.array(testing_behaviour['T'])
    for scan_s in range(nscans_wm):
        for trial_n in range(len(testing_angles)):
            test_interc = [1] + list(testing_activity[trial_n, scan_s, :])
            x,y = weights.predict(test_interc)[0]
            y_real =np.sin(np.radians(testing_angles[trial_n]) )
            x_real = np.cos(np.radians(testing_angles[trial_n]) )
            error = angle_between( (x,y), (x_real, y_real))
            time = scan_s * TR
            target = testing_behaviour['T'].iloc[trial_n]
            response= testing_behaviour['A_R'].iloc[trial_n]
            df.append( [ error, Subject, Brain_region, time, trial_n, condition, target, response ])
    #
    df=pd.DataFrame(df)
    df.columns=['error', 'Subject', 'Brain_region', 'time', 'trial', 'condition', 'target', 'response']
    return df

