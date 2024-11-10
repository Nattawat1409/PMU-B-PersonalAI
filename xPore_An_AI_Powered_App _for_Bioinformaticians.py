#Nattawat Ruensumrit#
#ETPCA-S0237#

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
RAND_STATE = 123
np.random.seed(RAND_STATE)
## GMM

# Scatter plots with histograms and normal distributions
def viz(data,mu,sd):
  green = '#40909A'
  orange = '#C76C2B'
  alpha = 0.5
  plt.figure()

  data_min = min(min(data[0]),min(data[1]))
  data_max = max(max(data[0]),max(data[1]))

  # Plot the 1st normal
  x = np.linspace(data_min, data_max, 100)
  p = norm.pdf(x, mu[0], sd[0])
  plt.plot(x, p, linewidth=2,color=green)

  # Plot the 2nd normal
  x = np.linspace(data_min, data_max, 100)
  p = norm.pdf(x, mu[1], sd[1])
  plt.plot(x, p, linewidth=2,color=orange)

  # Plot the histograms
  _ = plt.hist(data[0],bins=10,density=True,color=green,alpha=alpha)
  _ = plt.hist(data[1],bins=10,density=True,color=orange,alpha=alpha)

  # Plot the scatters
  plt.plot(data1,np.zeros(len(data[0])),linestyle='None', marker='o',markersize=10,alpha=alpha,c=green)
  plt.plot(data2,np.zeros(len(data[1])),linestyle='None', marker='o',markersize=10,alpha=alpha,c=orange)

  # Remove the frame (borders)
  ax = plt.gca()
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)


data = np.expand_dims(list(data1)+list(data2), axis=1)
gmm = GaussianMixture(n_components=2, init_params='random_from_data')
gmm.fit(data)

print('GMM >>>')
### HW: Complete the code below ###
w1,w2 = gmm.weights_ # Q1 
mu1,mu2 = gmm.means_.flatten() # Q2
sd1,sd2 = np.sqrt(gmm.covariances_.flatten())# Q3

mu = [mu1 , mu2] #convert into list format#
sd = [sd1 , sd2] #convert into list format#

print('p(x) = %.2fNormal(%.2f,%.2f)+%.2fNormal(%.2f,%.2f)' %(w1,mu1,sd1,w2,mu2,sd2))
## p_x = w1 * norm.pdf(w1, mu1, sd1) + w2 * norm.pdf(w2, mu2, sd2) ##
## print("p(x) = ",p_x) ##
data1 = np.random.normal(mu1, sd1, size=100)
data2 = np.random.normal(mu2, sd2, size=50)
# Visualize the data and the inferred norms.
viz([list(data1)]+[list(data2)],mu,sd) # Q4
### End of HW ###
