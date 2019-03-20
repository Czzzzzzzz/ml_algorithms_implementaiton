
# coding: utf-8

# In[117]:


import numpy as np

from sklearn import datasets
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


# # 0. PCA implementation

# In[108]:


class My_PCA:
    
    def __init__(self, n_component):
        '''
        Parameters:
        -----------
        n_component: int, float
            Number of components to keep
            if n_component > 1, n_component principle components are kept.
            
            if 0 < n_component < 1, select the number of components such that the 
            amount of variance that needs to be explained is greater than the percentage 
            specified by n_components.
        '''
        self.n_component = n_component
        self.components = None
        self.mean = None
        
        self.dims = None
    
    def __explained_variance_gt(self, E, percentage):
        '''
        select the number of components such that the 
        amount of variance that needs to be explained is greater than the percentage 
        specified by n_components.
        
        Parameters:
        -----------
        E: array-like, shape (dims,)
            diagnal matrix containing eigenvalues
        
        Returns:
        --------
        num: int
            Returns the number of components. 
        '''
        explained_variance = (E ** 2) 
        total_var = np.sum(explained_variance)
        explained_variance_ratio = explained_variance / total_var
        ratio_cumsum = np.cumsum(explained_variance_ratio)
        num = np.searchsorted(ratio_cumsum, percentage) + 1
        
        return num
        
    
    def fit(self, X):
        '''
        Step 1: the input data X is centered. 
        Step 2: compute SVD of X
        Step 3: select the number of components
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        
        Returns:
        --------
        self: Object
            Returns the instance itself.
        '''
        
        self.dims = X.shape[1]
        
        '''
        step 1: 
        center X.
        '''
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        
        '''
        step 2:
        compute SVD of X and keep the eigenvectors of right singular matrix which is 
        exactly the eigenvectors of covariance matrix X^T*X
        '''        
        U, E, VT = np.linalg.svd(X)
    
        '''
        step 3: determine the number of eigenvectors         
        '''
        if self.n_component < 1:            
            num = self.__explained_variance_gt(E, self.n_component)
        else:
            num = self.n_component
            
        self.components = VT[:num, :]
        
        return self
        
    
    def transform(self, X):
        
        assert X.shape[1] == self.dims, "Shape(%d, %d) of X doesn't match the dimenstions %d" % (X.shape[0], X.shape[1], self.dims)
        
        X = X - self.mean
        X_ = np.matmul(X, self.components.T)
        
        return X_
    


# # 1. Comparison between sklearn.PCA and my PC.
# 
# * In the source code of sklearn.PCA, it executes the function svd_flip() to correct the signs. That's why the sign of some columns is the opposite of my result. In fact, I have no idea about the motivation of svd_flip().

# In[103]:


get_ipython().run_cell_magic('time', '', "\niris = datasets.load_iris()\nX = iris.data\n\n# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])\npca = PCA(n_components=3)\npca.fit(X)\nX = pca.transform(X)\n\nprint('show the first 10 rows:\\n', X[:10])")


# In[107]:


get_ipython().run_cell_magic('time', '', "\nX = iris.data\n\npca = My_PCA(3)\npca.fit(X)\nX = pca.transform(X)\n\nprint('show the first 10 rows:\\n', X[:10])")


# # 2. Decompress data when percentage varies
# 
# * The larger percentage, the more eigenvectors are selected, and the less reconstruction error.

# In[109]:


def decompress(compressed_data, eigenvectors):
    X_hat = np.matmul(compressed_data, eigenvectors)
    return X_hat

def reconstruction_error(orig, decompressed):
    error = np.mean((orig - decompressed)**2)
    return error


# In[121]:


get_ipython().run_cell_magic('time', '', "\nX = iris.data\nns = [1, 2, 3, 4]\nerrors = []\nfor n in ns:\n    pca = My_PCA(n)\n    pca.fit(X)\n    X_ = pca.transform(X)\n    decompress_X = decompress(X_, pca.components)\n    errors.append(reconstruction_error(X, decompress_X))\n\nplt.figure(figsize=(8, 8))\nplt.plot(ns, errors)\nplt.xlabel('n_components')\nplt.ylabel('reconstruction error')\nplt.show()")


# # 3. Reference
# 
# [1] https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# 
# [2] https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/pca.py
