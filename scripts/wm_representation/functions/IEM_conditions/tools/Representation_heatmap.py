import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def Representation_heatmap(df, ref_angle=180, condition='1_7'):
    plt.figure()
    plt.title('Heatmap decoding')
    ######midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
    ax = sns.heatmap(df, yticklabels=list(df.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
    ax.plot([0.25, np.shape(df)[1]-0.25], [posch1_to_posch2(18),posch1_to_posch2(18)], 'k--')
    plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
    plt.ylabel('Angle')
    plt.xlabel('time (s)')
    plt.show(block=False)
    
    