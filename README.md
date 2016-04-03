# LinearRegression

Introduction
------------
Linear regression analysis fits a straight line to some data in order to capture the linear relationship between that data
The regression line is constructed by optimizing the parameters of the straight line function such that the line best fits a sample of (x, y) observations where y is a variable dependent on the value of x.<br />
Linear regression a high bias/low variance model.<br />
The objective function used in this program is the least squares method .<br />
Tools used: scikit-learn ( linear_model), numpy, matlabplotlib

Goal
-----
Perform a simple single-variable linear regression of X predicting Q for each distinct series found. 

Input data ( dataSimple.txt, dataMore.txt, datafloat.txt)
------------------------------------
Tab seperated data for series. Example:<br />
SERIES  Q   X <br />
SERIESA	13	1 <br />
SERIESA	14	2 <br /> 
SERIESB	15	3 <br />
SERIESB	16	4 <br />

Running the program ( LinearRegression.py )
------------------------------------------
python LinearRegression.py filename.tx

Output
-------
filename.png which contains the Linear Regression plot <br />
Additional print with model parameters : <br /> <br />
            &nbsp;&nbsp;&nbsp;SERIESA<br />
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Coefficients:  2.4<br />
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Residual sum of squares: 6.00<br />
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Variance score: 0.91<br />
		<p align="center">
  <img src="https://github.com/NamithaGS/LinearRegression/blob/master/dataSimple.txt.png" width="350"/>

</p>

