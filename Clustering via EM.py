#!/usr/bin/env python
# -*- coding: utf-8 -*-
###########################
# CSCI 573 Homework2 
# Author: Chu-An Tsai
# 10/16/2019
###########################

import numpy as np
import sys
import collections

# Get arguments from user input
script = sys.argv[0]
filename = sys.argv[1]
k = 3
dataset = np.loadtxt(filename, delimiter=",", usecols=(0,1,2,3))
n, d = dataset.shape
# Get original data labels and transform it for purity comparing
datalabel = np.loadtxt(filename, delimiter=",", dtype=np.str, usecols=(4))
for i in range (len(datalabel)):
    if (datalabel[i] == 'Iris-setosa'):
        datalabel[i] = 0
    elif (datalabel[i] == 'Iris-versicolor'):
        datalabel[i] = 1 
    else:
        datalabel[i] = 2         
data_label = datalabel.astype(np.float)

# Expectation-Maximization Algorithm
def EM(dataset, k, means, covs, priors, tolarence=0.00001):
    n, d = dataset.shape    
    old_means = np.zeros((k,d))    
    iter_num = 1

    while True:
        # E-step
        wij = np.zeros((k,n))
        for i in range(k):
            for j in range(n):
                temp = 0
                for m in range(k):
                    temp = temp + prob_part(dataset[j,:],means[m,:],covs[m][0]) * priors[m]
                wij[i,j] = prob_part(dataset[j,:],means[i,:],covs[i][0]) * priors[i] / temp
        # M-step
        for i in range(k):
            means[i,:] = 0
            for j in range(n):
                means[i,:] = means[i,:] + wij[i,j] * dataset[j,:] 
            means[i,:] = means[i,:] / np.sum(wij[i,:],axis=0)
            covs[i][0] = np.zeros((d,d))
            for m in range(d):
                for h in range(d):
                    for j in range(n):
                        covs[i][0][m][h] = covs[i][0][m][h] + wij[i,j] * (dataset[j,m] - means[i,m]) * (dataset[j,h] - means[i,h])
            covs[i][0] = covs[i][0] / np.sum(wij[i,:],axis=0)
            priors[i] = np.sum(wij[i,:],axis=0) / n
        # Iteration for convergance
        if np.linalg.norm(means - old_means) ** 2 <= tolarence:
            clusters = [[] for i in range(k)]
            labels = [[] for i in range(k)]
            cluster_label = np.zeros((n))
            for i in range(n):
                max_prob = sys.float_info.min
                max_idx = -1
                for j in range(k):
                    prob = prob_part(dataset[i,:],means[j,:],covs[j][0])
                    if prob > max_prob:
                        max_prob = prob
                        max_idx = j
                        cluster_label[i] = max_idx
                # Assign data point to cluster
                clusters[max_idx].append(dataset[i,:])
                labels[max_idx].append(i+1)
                
            return clusters, labels, cluster_label, means, covs, priors, iter_num
        iter_num = iter_num + 1
        old_means = np.copy(means)

# Partial multivariate normal distribution function
def prob_part(x, means, covs):
    return 1. / (((2*np.pi) ** (float(covs.shape[0])/2)) * (np.linalg.det(covs) ** (1./2))) * np.exp(-(1./2) * (x-means).T @ np.linalg.inv(covs) @ (x-means))

#purity calculation
def purity_cal(data_label, cluster_label):
       
    total_num = len(data_label)
    cluster_counter = collections.Counter(cluster_label)
    original_counter = collections.Counter(data_label)

    t = []
    for k in cluster_counter:
        p_k = []
        for j in original_counter:
            count = 0
            for i in range(len(cluster_label)):
                if cluster_label[i] == k and data_label[i] == j:
                    count += 1
            p_k.append(count)
        temp_t = max(p_k)
        t.append(temp_t)
    
    return sum(t)/total_num

# Initialization (mean, covariance, prior probibility)
# Assign the first, second, and third n/k data points as cluster#1, #2, and #3 
# Set up initial means as the instruction requested
cluster_1 = dataset[0:50]
mu_c1 = cluster_1.mean(axis=0)
cluster_2 = dataset[50:100]
mu_c2 = cluster_2.mean(axis=0)
cluster_3 = dataset[100:150]
mu_c3 = cluster_3.mean(axis=0)
means = np.vstack((mu_c1,mu_c2,mu_c3))
# initial identity matrix
covs = [[] for i in range(k)]
for i in range(k):
    covs[i].append(np.identity(d))
# Initial prior probabilities P(Ci) = 1/k        
priors = np.ones((k,1)) * (1./k)

# Call functions
clusters, labels, cluster_label, final_means, final_covs, final_priors, iter_num = EM(dataset, k, means, covs, priors)
cluster_label = cluster_label.astype(np.float)
purity_score = purity_cal(data_label, cluster_label)

# Print output (rounded with 3 digits after the decimal point as requested)
print("\nMean:\n", final_means[0].round(3), "\n" , final_means[1].round(3), "\n" , final_means[2].round(3))
print("\nCovariance Matrices:\n", final_covs[0][0].round(3), "\n", "\n", final_covs[1][0].round(3), "\n", "\n", final_covs[2][0].round(3))
print("\nIteration count = ", iter_num)
print("\nCluster Membership:")
print("Cluster #1:\n",labels[0])
print("Cluster #2:\n",labels[1])
print("Cluster #3:\n",labels[2])
print("\nSize:",len(labels[0]), ",",len(labels[1]), ",",len(labels[2]))
print("\nPurity:", round(purity_score, 3),"\n")
