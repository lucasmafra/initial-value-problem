#!/usr/bin/env python
# coding: utf-8

# In[2]:


def gauss_seidel(A, b, x):
    tolerance = 1e-16
    max_iterations = 1000000
    n = len(A)
    xprev = [0.0 for i in range(n)]
    for i in range(max_iterations):
        for j in range(n):
            xprev[j] = x[j]
        for j in range(n):
            summ = 0.0
            for k in range(n):
                if (k != j):
                    summ = summ + A[j][k] * x[k]
            x[j] = (b[j] - summ) / A[j][j]
        diff1norm = 0.0
        oldnorm = 0.0
        for j in range(n):
            diff1norm = diff1norm + abs(x[j] - xprev[j])
            oldnorm = oldnorm + abs(xprev[j])  
        if oldnorm == 0.0:
            oldnorm = 1.0
        norm = diff1norm / oldnorm
        if (norm < tolerance) and i != 0:            
            return x

