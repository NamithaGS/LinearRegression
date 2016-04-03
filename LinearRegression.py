__author__ = 'namitha'

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import os
import sys
# Specify the data file to open
filetotopen = sys.argv[1]
fname = os.path.expanduser(filetotopen)
# Load the dataset
data = np.genfromtxt(
    fname,
    names = True, #  If `names` is True, the field names are read from the first valid line
    delimiter = '\t', # tab separated values
    dtype = None)  # guess the dtype of each column

totalnumberofrecords = data.size
acolor=['Blue','Red','Orange','black','Yellow','Purple','Green','Brown']
colorindex=0
serieslabel = np.unique(data['SERIES'])
minx=[];miny=[];maxx=[];maxy=[];

## For every disctinct series
for series in serieslabel:
    seriesdatax0 = []
    seriesdatay0 = []
    for x in range(0,totalnumberofrecords):  ## numpy array cannot take mixed data types
        if (data['SERIES'][x]== series):     ## Seperate the tuples to arrays for the specific series
            seriesdatax0.append(data['X'][x])
            seriesdatay0.append(data['Q'][x])

    seriesdatax1=np.array(seriesdatax0)    ## Convert to numpy arrays for easier manipulation
    seriesdatay1=np.array(seriesdatay0)
    splitnumber = -(seriesdatax1.size/2)

    diabetes_X_train = seriesdatax1[:splitnumber]  # Split the data into training/testing sets
    diabetes_X_test = seriesdatax1[splitnumber:]

    diabetes_y_train =  seriesdatay1[:splitnumber]  # Split the targets into training/testing sets
    diabetes_y_test = seriesdatay1[splitnumber:]

    regr = linear_model.LinearRegression() # Create linear regression object (fit_intercept=True,normalize=True)

    regr.fit(diabetes_X_train.reshape(diabetes_X_train.shape[0],1) , diabetes_y_train) # Train the model using the training sets

    # Print the coefficients
    print (series)
    print "Coefficients: ", float(regr.coef_)
    # The mean square error
    print ("Residual sum of squares: %.2f"
          % np.mean((regr.predict(diabetes_X_test.reshape(diabetes_X_test.shape[0],1)) - diabetes_y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(diabetes_X_test.reshape(diabetes_X_test.shape[0],1), diabetes_y_test))

    # Plot outputs for test set
    plt.plot(diabetes_X_test, regr.predict(diabetes_X_test.reshape(diabetes_X_test.shape[0],1)), color=acolor[colorindex],   linewidth=2,label = series)
    plt.scatter(diabetes_X_test, diabetes_y_test,  color=acolor[colorindex])
    plt.grid()
    minx.append(diabetes_X_test.min())   ## get max and min values for axis
    miny.append(diabetes_y_test.min())
    maxx.append(diabetes_X_test.max())
    maxy.append(diabetes_y_test.max())
    if(colorindex<=7): colorindex+=1
    else:
        colorindex=0
    plt.draw()
    plt.legend(loc='upper left')

plt.title('Single-Variable Linear Regression')
plt.xlabel('X values')
plt.ylabel('Q values')
plt.xlim(min(minx)-10,max(maxx)+10)
plt.ylim(min(miny)-10,max(maxy)+10)
plt.savefig(filetotopen+".png")
plt.show()   ##Show the plots




