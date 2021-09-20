import matplotlib.pyplot as plt
import numpy as np
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    # figure, axes = plt.subplots()
    display.display(plt.gcf())
    plt.clf()
    plt.title('Model Training...')
    plt.xlabel('No of Games')
    plt.ylabel('Score')
    # Plot scores and mean_scores
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    # axes.plot(np.arange(-4, 5) , scores, 'rx', np.arange(-4, 5) , mean_scores, 'b+',  linestyle='solid') 
    # axes.fill_between(np.arange(-4, 5),  
    #               scores,  
    #               mean_scores, 
    #               where=mean_scores>scores,  
    #               interpolate=True, 
    #               color='green', alpha=0.3) 
    # lgnd = axes.legend(['scores', 'mean_scores'],  loc='upper center',  shadow=True) 
    # lgnd.get_frame().set_facecolor('#ffb19a') 
    # plt.legend(loc='upper center', frameon=False, ncol=1)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
