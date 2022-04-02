import os
import matplotlib
from matplotlib import pyplot as plt 
import numpy as np 



def graphs(y1,y2,y3,y4,y5,y6):

    x=["None", "n/25","n/50" ]

    # Initialise the figure and axes.
    fig, ax = plt.subplots(1, figsize=(8, 6))

    # Set the title for the figure, Change this accordingly 
    fig.suptitle('Item_based_recommendation_MAE', fontsize=15)

    # Draw all the lines in the same plot, assigning a label for each one to be
    # shown in the legend.
    ax.plot(x, y1, color="red", label="Pearson, >0 sim")
    ax.plot(x, y2, color="brown", label="Pearson, 0.3 sim")
    ax.plot(x, y3, color="blue", label="Pearson, 0.5 sim")
    ax.plot(x, y4, color="cyan", label="Distance, >0 sim")
    ax.plot(x, y5, color="Orange", label="Distance, 0.3 sim")
    ax.plot(x, y6, color="Purple", label="Distance, 0.5sim")

    # Add a legend, and position it on the lower right (with no box)
    plt.legend(loc="best", title="Legend Title", frameon=False)
    #change this accordingly
    plt.xlabel("Sim_Weight")
    plt.ylabel("MAE")
    plt.savefig("graphs/Item_based_MAE")
    
    plt.show()

def charts():

    labels = ['UD', 'UP', 'ID', 'IP']
    men_means = [20, 34, 30, 35, 27]
    y1=[49659,87497,24176,57540]
    y2=[33774,87459,16093,57236]
    y3=[8780,77701,21945,55508]
    y4=[90645,98124,24716,97504]
    y5=[158,96646,226,85665]
    y6=[0,64202,0,69868]	
    y7=[94708,98779,97029,97015]
    y8=[98,87781,0,87305]
    y9=[0,37949,0,36300]




    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, y1, width )
    rects2 = ax.bar(x + width/2, y2, width)
    rects3 = ax.bar(x - width/2, y3, width )
    rects4 = ax.bar(x + width/2, y2, width)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()
    

def main():

    # #User Based MSE
    # y1= [ 0.7473898938, 0.7710908531, 0.8030167186] #curve1: Pearson, >0 sim
    # y2=[0.7470730795, 0.7692387843, 0.8316624451] #Curve2: Pearson, 0.3 sim
    # y3=[0.7314830318, 0.8408836233, 0.907656] #Curve3: Pearson, 0.5 sim
    # y4=[0.5240260147, 0.7843381846, 0.825747435] #Curve4: Distance, >0 sim
    # y5=[0.3514603334, 0.0506329114, 0.0204081633] #Curve5: Distance, 0.3sim
    # y6=[0,0,0] #Curve6: Distance, 0.5sim

    # #User Based RMSE
    # y1= [0.8645171449, 0.87811779, 0.8961120012] #curve1: Pearson, >0 sim
    # y2=[0.8643338935, 0.8770625886, 0.9119552868]#Curve2: Pearson, 0.3 sim
    # y3=[0.8552678129, 0.9169970683, 0.95271] #Curve3: Pearson, 0.5 sim
    # y4=[0.7238964116, 0.8856286946, 0.9087064625 ] #Curve4: Distance, >0 sim
    # y5=[0.5928409006, 0.2250175802, 0.1428571429] #Curve5: Distance, 0.3sim
    # y6=[0,0,0] #Curve6: Distance, 0.5sim

    # #User Based MAE
    # y1= [0.6612182019, 0.6911957444, 0.7086043915] #curve1: Pearson, >0 sim
    # y2=[0.6610176069, 0.6877784088, 0.7119841439] #Curve2: Pearson, 0.3 sim
    # y3=[0.6490103791, 0.6886860195,  0.708838] #Curve3: Pearson, 0.5 sim
    # y4=[0.5148981729, 0.6856509925, 0.7107451857] #Curve4: Distance, >0 sim
    # y5=[0.4121893461, 0.0506329114, 0.0204081633] #Curve5: Distance, 0.3sim
    # y6=[0,0,0] #Curve6: Distance, 0.5sim

    #Item Based MSE
    # y1= [1.055315854,0.6909226879,0.6765823546] #curve1: Pearson, >0 sim
    # y2=[1.041860603,0.7727069532,0.7536180275] #Curve2: Pearson, 0.3 sim
    # y3=[1.027870342,0.7936596326,0.846440929] #Curve3: Pearson, 0.5 sim
    # y4=[0.158828, 0.1588281804, 0.7033940428] #Curve4: Distance, >0 sim
    # y5=[0.1074731007, 0.530973451, 0] #Curve5: Distance, 0.3sim
    # y6=[0,0,0] #Curve6: Distance, 0.5sim

    # #Item Based RMSE
    # y1= [0.7714179033,0.6597354074,0.6505059735] #curve1: Pearson, >0 sim
    # y2=[0.7672174266,0.679348491,0.6754549041] #Curve2: Pearson, 0.3 sim
    # y3=[0.7627780253,0.6647569185,0.6541721769] #Curve3: Pearson, 0.5 sim
    # y4=[0.398533,0.3985325337, 0.8386859023] #Curve4: Distance, >0 sim
    # y5=[0.3278309025, 0.2304286118,0] #Curve5: Distance, 0.3sim
    # y6=[0, 0, 0] #Curve6: Distance, 0.5sim

    # #Item Based MAE
    # y1=[1.027285673,0.8312175936,0.8225462629] #curve1: Pearson, >0 sim
    # y2=[1.020715731,0.8790375152,0.8681117598] #Curve2: Pearson, 0.3 sim
    # y3=[1.013839406,0.8908757672,0.9200222437] #Curve3: Pearson, 0.5 sim
    # y4=[0.124876,0.1248760397, 0.6450863474] #Curve4: Distance, >0 sim
    # y5=[0.1090802917, 0.0530973451,0 ] #Curve5: Distance, 0.3sim
    # y6=[0, 0, 0] #Curve6: Distance, 0.5sim

    graphs(y1,y2,y3,y4,y5,y6)
    

if __name__ == '__main__':
    main()