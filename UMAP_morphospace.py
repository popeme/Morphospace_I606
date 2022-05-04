#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:54:45 2022

@author: Maria
"""

import umap
from matplotlib.pyplot import plot, show, draw, figure, cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io 
import numpy as np
import seaborn as sns


#%%
%matplotlib inline

in_dir = "/home/spornslab/Documents/Maria/morphospace/Data/"
out_dir = "/home/spornslab/Documents/Maria/morphospace/Results/"

matrices = scipy.io.loadmat(in_dir + 'morpho.mat')
phenotypes = scipy.io.loadmat(in_dir + 'phenoVars.mat')

networks = np.array(matrices['morpho'])
N1, N2 = networks.shape

triu = np.triu_indices(n=N1,k=1)

upper_triangles = np.zeros((N1,79800))

# for i in range(N3):
#     upper_triangles[i,:] = networks[i,:][triu[0], triu[1]]
#     print(i)

#vec = np.array(upper_triangles) # vec is of size (N,M) - N = number of data points; M = dimension size of data point
vec = networks



for n_neighbors in np.arange(10,95,10):
    for dist in np.arange(0.1,1.1,0.1):
        plt.figure()
        counter = 1
        for func in ["euclidean", "chebyshev", "cosine"]:
            umap_output = umap.UMAP(n_components=2,
                                    metric=func,
                                    min_dist = dist,
                                    n_neighbors=n_neighbors).fit_transform(vec) # n_components = 3 gives you 3-d representation ; 2 would give you 2d
            
            
            plt.subplot(1,3,counter)
            plt.title("{0} | {1} | {2}".format(func, round(dist,2), n_neighbors))
            plt.scatter(umap_output[:,0], 
                        umap_output[:,1], 
                        s=10)
            counter += 1
            
        plt.tight_layout()
        plt.savefig(out_dir+'umap'+ str(round(dist,2)) +'_'+ str(n_neighbors)+'.png')
        plt.show()
        

#%%

%matplotlib inline

in_dir = "/home/spornslab/Documents/Maria/morphospace/Data/"
out_dir = "/home/spornslab/Documents/Maria/morphospace/Results/"

matrices = scipy.io.loadmat(in_dir + 'morpho.mat')
phenotype = scipy.io.loadmat(in_dir + 'phenoVars.mat')

phenotypes = np.array(phenotype['vars_morpho'])
networks = np.array(matrices['morpho'])
N1, N2 = networks.shape

triu = np.triu_indices(n=N1,k=1)

upper_triangles = np.zeros((N1,79800))

# for i in range(N3):
#     upper_triangles[i,:] = networks[i,:][triu[0], triu[1]]
#     print(i)

#vec = np.array(upper_triangles) # vec is of size (N,M) - N = number of data points; M = dimension size of data point
vec = networks

measures = ['perceived stress','anxiety','depression','age-adjusted fluid intelligence','age','years of education']

umap_outputs_all = np.zeros((92,2,3))

plt.figure()
sns.color_palette("viridis", as_cmap=True)
counter = 1
for func in ["euclidean", "chebyshev", "cosine"]:
    measure = measures[i]
    umap_output = umap.UMAP(n_components=2,
                            metric=func,
                            min_dist = 0.3,
                            n_neighbors=30).fit_transform(vec) # n_components = 3 gives you 3-d representation ; 2 would give you 2d
            
    umap_outputs_all[:,:,counter-1] = umap_output        
    plt.subplot(1,3,counter)
    plt.title("{0}".format(func))
    plt.scatter(x=umap_output[:,0], 
                y=umap_output[:,1], 
                s=10)
    counter += 1       
plt.tight_layout()
plt.savefig(out_dir+'umap_'+measure+ str(0.3)+'_'+ str(30)+'.png')
plt.show()

plt.figure()

for i in range(0,6):
    counter = 1
    for func in ["euclidean", "chebyshev", "cosine"]:
        measure = measures[i]
        plt.subplot(1,3,counter)
        plt.title("{0}".format(func))
        #sns.color_palette("viridis", as_cmap=True)
        plt.scatter(x=umap_outputs_all[:,0,counter-1], 
                        y=umap_outputs_all[:,1,counter-1], 
                        s=10, 
                        c = phenotypes[:,i],
                        cmap="viridis")
        
        counter += 1
    plt.suptitle("{0}".format(measure))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir + 'umap' + str(0.3) + '_' + str(30) +'_' + measure + '.png')
    plt.show()
    
  