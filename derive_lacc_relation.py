import argparse
import numpy as np 
import pylab

from numpy import *
from numpy.fft import *
from pylab import *

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#import np.multiply as m_array

#multiply

from scipy import optimize
from string import Template
from scipy import stats



from matplotlib import rc, rcParams 
from matplotlib import rc

DESCRIPTION = 'Derive accretion luminosity to accretion rate relation using sample from Mendigutia and Herczeg&Hillenbrand.'

parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)

flux_conv_fact = lambda R: 4*np.pi*np.multiply(R,R); # defining flux conversion factor

convert_flux = lambda flux, conv_fact: np.multiply(conv_fact,flux) # converting flux to luminosity


matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

def confband(xd,yd,a,b,conf=0.95,x=None):
	"""
	Calculates the confidence band of the linear regression model at the desired confidence level.
	The 2sigma confidence interval is 95% sure to contain the best-fit regression line. 
	This is not the same as saying it will contain 95% of the data points.


	Arguments: 
	- conf: desired confidence level, by default 0.95 (2 sigma) 
	- xd,yd: data arrays - a,b: linear fit parameters as in y=ax+b 
	- x: (optional) array with x values to calculate the confidence band. 
	If none is provided, will by default generate 100 points in the original x-range of the data. 

	Returns: Sequence (lcb,ucb,x) with the arrays holding the lower and upper confidence bands corresponding to the [input] x array. 

	Usage: 
	>>> lcb,ucb,x=nemmen.confband(all.kp,all.lg,a,b,conf=0.95) calculates the confidence bands for the given input arrays
	>>> pylab.fill_between(x, lcb, ucb, alpha=0.3, facecolor='gray') plots a shaded area containing the confidence band

	References: 
	1. http://en.wikipedia.org/wiki/Simple_linear_regression, see Section Confidence intervals 
	2. http://www.weibull.com/DOEWeb/confidence_intervals_in_simple_linear_regression.htm 

	Rodrigo Nemmen v1 Dec. 2011 v2 Jun. 2012: 

	Costigan 22 Dec 2013, made adjustments to clarify what this piece of code does. 
	see confidence intervals here http://en.wikipedia.org/wiki/Simple_linear_regression#Confidence_intervals
	"""
	
	

	alpha=1.-conf # significance
	n=len(xd)  # data sample size 
	print "length of data"
	print n

	if x==None: x=np.linspace(min(xd),max(xd),100) # Predicted values (best-fit model) 
	y=np.multiply(a,x) + b 
	# Auxiliary definitions 
	print "Slope----------------"
	print a

	N=np.size(xd)  #length of data
	mean_sq_er = 1./(N-2.)* np.sum((yd-a*xd-b)**2) #mean sqaure error, sum of residuals divided by dof
	print "Mean square error"
	print mean_sq_er
	mean_sq_er = sqrt(mean_sq_er)
	
	sxd=np.sum((xd-np.mean(xd))**2) 
	sx=(x-np.mean(xd))**2 # array 

	# Quantile of Student's t distribution for 
	p=1-alpha/2 
	q=stats.t.ppf(1.-alpha/2.,n-2) 

	# Confidence band
	dy=q*mean_sq_er*np.sqrt( 1./n + sx/sxd)
	ucb=y+dy # Upper confidence band
	lcb=y-dy # Lower confidence band
	

	return lcb,ucb,x,y


def error_calculation(xdata,ydata,fitted_slope,fitted_intercept):

	sample_x = np.linspace(min(xdata),max(xdata),100)
	model_y = fitted_intercept + np.multiply(sample_x,fitted_slope)

	N=np.size(xdata)  #length of data
	mean_sq_er = 1./(N-2.)*np.sum((ydata-fitted_slope*xdata-fitted_intercept)**2) #mean sqaure error, sum of residuals divided by dof

	mean_x = np.mean(xdata)
	mean_x2 = mean_x**2
	mean_diff = np.sum(np.subtract(xdata,mean_x)**2)
	sigma_intercept = mean_sq_er*np.sqrt(1./N + mean_x2/mean_diff) 

	sigma_slope = mean_sq_er*np.sqrt(1./mean_diff)

	return sigma_slope,sigma_intercept

file_HH='HHnonzero_accretion.lst'
data_HH = np.genfromtxt(file_HH, dtype=np.float64)
name_HH,Lacc_HH, Mdot_HH, Mass_HH, FluxHa_HH, R_HH = data_HH.T

file_M11='Mend_accretion.lst'
data_M11 = np.genfromtxt(file_M11, dtype=np.float64)
name_M11, LogMdot_M11, delta_Mdot_M11, logLacc_M11, delta_Lacc_M11, upperlimit, Mass_M11, LumHa = data_M11.T

file_HK='HK_accretion.lst'
data_HK = np.genfromtxt(file_HK, dtype=np.float64)
name_HK, LumHa_HK, Lacc_HK = data_HK.T

file_A='Alcala_Accretion.lst'
data_A = np.genfromtxt(file_A, dtype=np.float64)
name_A, flux_A, error,LogLacc_A, distance_A, mass_A = data_A.T

file_WHT='WHT_LHa_Lacc_withremoval.txt'
data_WHT = np.genfromtxt(file_WHT, dtype=np.float64)
LogLumHa_WHT, LogLumAcc_WHT, Mdot_WHT, Mass_WHT, Name_WHT,Year_WHT = data_WHT.T


n_WHT=len(LogLumHa_WHT)  # data sample size 
print "Log LumHa",LogLumHa_WHT
print "Log LumAcc", LogLumAcc_WHT



#some neccessary constants 
Rsol = 6.955*(10**10) #Solar Radius in cm
Lsol = 3.8839e33 # Solar Luminosity in Ergs/s
distance_Taurus = 140.*3.086e18 #4.31998*(10**20) # 140pc in cm

Rcm_HH = R_HH*Rsol  # conversion from solar radii


#Convert HH flux in line luminsity
conversion_factor = 4.*np.pi*distance_Taurus**2
FluxHa_HH_True = FluxHa_HH*(1e-14) # to convert to real units ergs/cm2/s
print "Flux", FluxHa_HH
LumHa_HH = FluxHa_HH_True*conversion_factor #Halpha flux here is 10^-14 ergs/cm2/s
LumHasol_HH = LumHa_HH/Lsol #observed luminosity
logLumHa_HH = np.log10(LumHasol_HH, dtype='float64')

#converting accretion luminosity into real units
Lacc_Lsol_HH = Lacc_HH*(10**-4)	
logLacc_HH = np.log10(Lacc_Lsol_HH, dtype='float64')

#converting HK accretion lum into the same
LogLacc_HK = np.log10(Lacc_HK)

#putting both samples together.
logLacc_HH_HK = np.concatenate((logLacc_HH,LogLacc_HK))
logLumHa_HH_HK = np.concatenate((logLumHa_HH,LumHa_HK))



#Converting Mendigutia Luminosity to real untils. 
LumHa_true = LumHa*(10**31)
LumHa_sol = LumHa_true/Lsol
logLumHa_M11 = np.log10(LumHa_sol, dtype='float64' )

#Convert Alcala flux in line luminsity
distanceA_cm = distance_A*3.086e18
conversion_factor = 4.*np.pi*distanceA_cm**2
print "Conversion Factor:", conversion_factor
LumHa_A = flux_A*conversion_factor #Halpha flux here is ergs/cm2/s
LumHasol_A = LumHa_A/Lsol #in units of solar luminosity
logLumHa_A = np.log10(LumHasol_A, dtype='float64')



sampleLHa = [-6,-5,-4,-3,-2,-1,0,1]		


#Fit to Mendigutia 
slope_Mend, intercept_Mend, r_value, p_value, std_err = stats.linregress(logLumHa_M11,logLacc_M11)
print "-----Mendigutia-----"
print "Fit Slope M11:", slope_Mend
print "Fit Intercept M11:", intercept_Mend 
Mend_lumfit = np.multiply(slope_Mend,sampleLHa) + intercept_Mend

#finding errors to fit.
er_slopeM11, er_interM11 = error_calculation(logLumHa_M11,logLacc_M11,slope_Mend,intercept_Mend)
print "Error on slope M11:", er_slopeM11
print "Error on intercept M11:", er_interM11

""" This data is wrong so fit data removed """
# Fit to Herczeg and Hillenbrand
#slope_HH_HK, intercept_HH_HK, r_value, p_value, std_err = stats.linregress(logLumHa_HH_HK,logLacc_HH_HK)
#print "Fit Slope HH"
#print slope_HH_HK
#print "Fit Intercept HH"
#print intercept_HH_HK 
#HH_HK_lumfit = np.multiply(slope_HH_HK,sampleLHa)+intercept_HH_HK
#finding errors to fit
#er_slopeHH_HK, er_interHH_HK = error_calculation(logLumHa_HH_HK,logLacc_HH_HK,slope_HH_HK,intercept_HH_HK)
#print "Error on slope to HH and HK "
#print er_slope_HH_HK
#print "Error on intercept to HH and HK"
#print er_interHH_HK


# Fit to Alcala
slope_A, intercept_A, r_value, p_value, std_err = stats.linregress(logLumHa_A,LogLacc_A)
print "----Alcala----"
print "Fit Slope Alcala:", slope_A
print "Fit Intercept Alcala:", intercept_A 
A_lumfit = np.multiply(slope_A,sampleLHa) + intercept_A

#finding errors to fit.
er_slopeA, er_interA = error_calculation(logLumHa_A,LogLacc_A,slope_A,intercept_A)
print "Error on slope Alcala:", er_slopeA
print "Error on intercept Alcala:", er_interA



#Fit to entire sample
All_LumHa = np.concatenate((logLumHa_A,logLumHa_M11))
All_Lumacc = np.concatenate((LogLacc_A,logLacc_M11))

slope_all, intercept_all, r_value, p_value, std_err = stats.linregress(All_LumHa,All_Lumacc)
All_lumfit = slope_all*All_LumHa+intercept_all


#Showing WHT Lacc with fit. 
WHT_Lacc = slope_all*LogLumHa_WHT+intercept_all
print "Fit Slope all:", slope_all
print "Fit Intercept:", intercept_all 


if raw_input('Show plot? ') in ('y', 'Y'):

	ax = plt.subplots()

#	ax.plot(logLumHa_HH_HK,logLacc_HH_HK, 'ro')
#	ax.plot(LumHa_HK,LogLacc_HK, 'co')
#fig, ax = plt.subplots()
#	ax.plot(logLumHa_HH_HK,logLacc_HH_HK, 'ro')
#	ax.plot(LumHa_HK,LogLacc_HK, 'co')
#	ax.plot(logLumHa_HH_HK,HH_HK_lumfit,'r')
	ax.plot(logLumHa_M11,logLacc_M11, 'bo')
	ax.plot(sampleLHa,Mend_lumfit,'b--')

#	ax.plot(sampleLHa,All_lumfit,'g-')

	ax.plot(logLumHa_A,LogLacc_A, 'y^')
	ax.plot(sampleLHa,A_lumfit, 'y--')

	ax.set_xlabel('Log(L Halpha) [L$_{\odot}$]',fontsize=18)
	ax.set_ylabel('Log(Lacc) [L$_{\odot}$]',fontsize=18)
	pylab.ylim([-6,4])
	pylab.xlim([-7,1])
	plt.show()    

if raw_input('Show Plot with all samples on it? ') in ('y', 'Y'):

	
	file_HHtest='test.lst' # Reading in HH data pulled from plot
	data_HHtest = np.genfromtxt(file_HHtest,dtype=np.float64)	
	LogACC_HHtest, LogHA_HHtest = data_HHtest.T

	#putting both samples together.
	logHa_all = np.concatenate((logLumHa_M11,logLumHa_A))#,LogHA_HHtest))
	logLacc_all = np.concatenate((logLacc_M11,LogLacc_A))#,LogACC_HHtest))


	fig, ax = plt.subplots()
	ax.plot(logLumHa_M11,logLacc_M11, 'bo',label='Mendigutia+11')
	ax.plot(logLumHa_A,LogLacc_A,'ys',label='Alcala+13')
	ax.plot(LogLumHa_WHT,LogLumAcc_WHT, 'ms',label='This Sample')

	l = ax.legend(loc=4,numpoints=1)

	slope_HHtest, intercept_HHtest, r_value, p_value, std_err = stats.linregress(LogHA_HHtest,LogACC_HHtest)


	slope_all, intercept_all, r_value, p_value, std_err = stats.linregress(logHa_all,logLacc_all)
	print "-------All data fit-----------"
	print "Fit Slope All enteries:", slope_all
	print "Fit Intercept All enteries:", intercept_all
	print "Standard Error for fit to all enteries:", std_err
	correlation = r_value**2
	print "Correlation coefficent for fit:", correlation
	print "P value:", p_value
	all_lumfit =  np.multiply(slope_all,sampleLHa) + intercept_all

	er_slopeALL, er_interALL = error_calculation(logHa_all,logLacc_all,slope_all,intercept_all)
	print "Error on slope for all data points:", er_slopeALL
	print "Error on intercept for all data points:",er_interALL

	ax.plot(sampleLHa,all_lumfit,'m--')
	ax.plot(sampleLHa,Mend_lumfit,'b--')
	ax.plot(sampleLHa,A_lumfit, 'y--')

	#plotting the 2 sigma shadding either side of fit to data
	slope, intercept, r_value, p_value, std_err = stats.linregress(logHa_all,logLacc_all)

	lcb,ucb,xcb,ycb = confband(logHa_all,logLacc_all,slope,intercept,conf=0.95,x=None)

	pylab.fill_between(xcb, lcb, ucb, alpha=0.3, facecolor='gray',linewidth=0.0) 


	for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
		item.set_fontsize(14)


	ax.set_xlabel('Log(Lum. $H\\alpha$) [L$_{\odot}$]',fontsize=16)
	ax.set_ylabel('Log(Lum. Acc.) [L$_{\odot}$]',fontsize=16)

	sample_Lacc =  np.multiply(sampleLHa,slope_all) + intercept_all

	ax.plot(sampleLHa,sample_Lacc,'k-')

	pylab.xlim([-5.6,0.2])
	pylab.ylim([-5.5,4])
	plt.show()  


if raw_input('Show Mdot-Mass Plot?') in ('y', 'Y'):
	print "Accretion Rate",Mdot_WHT
	print "Mass",Mass_WHT
	print "LogLum Acc",LogLumAcc_WHT

	fig, ax = plt.subplots()
	ax.plot(Mass_WHT, Mdot_WHT, 'r^',label='ISIS')
	ax.plot(Mass_M11, LogMdot_M11, 'bo',label='Mendigutia 2011')
        l = ax.legend(loc=4,numpoints=1)	
	plt.xlabel("Mass [Msol]")
	plt.ylabel("Log($\dot{M}$) [L$_{\odot}$] ")
        pylab.xlim([0,7])
        pylab.ylim([-9,-3])
	plt.show()


if raw_input('Show Mass Comparison?') in ('y', 'Y'):


	
	plt.hist(Mass_HH, bins=10, histtype='stepfilled',alpha=0.5, color='g',label='ISIS')
	plt.hist(Mass_M11, bins=20, histtype='stepfilled',alpha=0.5, color='b',label='LAMP')
	plt.hist(Mass_WHT, bins=20, histtype='stepfilled',alpha=0.5, color='r',label='LAMP')

	#pylab.ylim([0,4])
	plt.title("Comparison of Mass ranges")
	plt.xlabel("Mass [Msol]")
	plt.ylabel("Number")
	plt.show()






