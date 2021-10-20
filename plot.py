import pylab
from datetime import datetime

def plot_data(episodes, scores, max_q_mean, title):
        # datetime object containing current date and time
        now = datetime.now()
         
        print("now =", now)
        
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d_%m_%Y__%H_%M_%S_")
        print("date and time =", dt_string)	
        pylab.figure(0)
        pylab.plot(episodes, max_q_mean, 'b')
        pylab.xlabel("Episodes")
        pylab.ylabel("Average Q Value")
        pylab.savefig("{}_{}qvalues.png".format(title,dt_string))

        pylab.figure(1)
        pylab.plot(episodes, scores, 'b',linestyle="",marker="o")
        pylab.xlabel("Episodes")
        pylab.ylabel("Score")
        pylab.savefig("{}_{}scores.png".format(title,dt_string))
