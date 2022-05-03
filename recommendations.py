'''
CSC381: Building a simple Recommender System
The final code package is a collaborative programming effort between the
CSC381 student(s) named below, the class instructor (Carlos Seminario), and
source code from Programming Collective Intelligence, Segaran 2007.
CSC381 Programmer/Researcher: <<Anh Hoang, Mike Remezo, Malavika Kalani>>
'''

import os
import time
import matplotlib
from matplotlib import pyplot as plt 
import numpy as np 
import math
from math import sqrt 
import copy as cp
import pickle 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from os import getcwd
from numpy.linalg import solve 
from sklearn.metrics import mean_squared_error
from copy import deepcopy
import statistics
from scipy import stats # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html#scipy.stats.ttest_ind


#ALS AND SGD 
class ExplicitMF():
    def __init__(self, 
                 ratings,
                 n_factors=40,
                 learning='sgd',
                 sgd_alpha = 0.1,
                 sgd_beta = 0.1,
                 sgd_random = False,
                 item_fact_reg=0.0, 
                 user_fact_reg=0.0,
                 item_bias_reg=0.0,
                 user_bias_reg=0.0,
                 max_iters = 20,
                 verbose=True):
        """
        Train a matrix factorization model to predict empty 
        entries in a matrix. The terminology assumes a 
        ratings matrix which is ~ user x item
        
        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings
            Note: can be full ratings matrix or train matrix
        
        n_factors : (int)
            Number of latent factors to use in matrix factorization model
            
        learning : (str)
            Method of optimization. Options include 'sgd' or 'als'.
        
        sgd_alpha: (float)
            learning rate for sgd
            
        sgd_beta:  (float)
            regularization for sgd
            
        sgd_random: (boolean)
            False makes use of random.seed(0)
            False means don't make it random (ie, make it predictable)
            True means make it random (ie, changee everytime code is run)
        
        item_fact_reg : (float)
            Regularization term for item latent factors
            Note: currently, same value as user_fact_reg
        
        user_fact_reg : (float)
            Regularization term for user latent factors
            Note: currently, same value as item_fact_reg
            
        item_bias_reg : (float)
            Regularization term for item biases
            Note: for later use, not used currently
        
        user_bias_reg : (float)
            Regularization term for user biases
            Note: for later use, not used currently
            
        max_iters : (integer)
            maximum number of iterations
        
        verbose : (bool)
            Whether or not to printout training progress
            
            
        Original Source info: 
            https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea#introsgd
            https://gist.github.com/EthanRosenthal/a293bfe8bbe40d5d0995#file-explicitmf-py
        """
        
        self.ratings = ratings 
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_fact_reg = item_fact_reg
        self.user_fact_reg = user_fact_reg
        self.item_bias_reg = item_bias_reg 
        self.user_bias_reg = user_bias_reg 
        self.learning = learning
        if self.learning == 'als':
            np.random.seed(0)
        if self.learning == 'sgd':
            self.sample_row, self.sample_col = self.ratings.nonzero()
            self.n_samples = len(self.sample_row)
            self.sgd_alpha = sgd_alpha # sgd learning rate, alpha
            self.sgd_beta = sgd_beta # sgd regularization, beta
            self.sgd_random = sgd_random # randomize
            if self.sgd_random == False:
                np.random.seed(0) # do not randomize
        self._v = verbose
        self.max_iters = max_iters
        self.nonZero = ratings > 0 # actual values
        
        print()
        if self.learning == 'als':
            print('ALS instance parameters:\nn_factors=%d, user_reg=%.5f,  item_reg=%.5f, num_iters=%d' %\
              (self.n_factors, self.user_fact_reg, self.item_fact_reg, self.max_iters))
        
        elif self.learning == 'sgd':
            print('SGD instance parameters:\nnum_factors K=%d, learn_rate alpha=%.5f, reg beta=%.5f, num_iters=%d, sgd_random=%s' %\
              (self.n_factors, self.sgd_alpha, self.sgd_beta, self.max_iters, self.sgd_random ) )
        print()

    def train(self, n_iter=10): 
        """ Train model for n_iter iterations from scratch."""
        
        def normalize_row(x):
            norm_row =  x / sum(x) # weighted values: each row adds up to 1
            return norm_row

        # initialize latent vectors        
        self.user_vecs = np.random.normal(scale=1./self.n_factors,\
                                          size=(self.n_users, self.n_factors))
        self.item_vecs = np.random.normal(scale=1./self.n_factors,
                                          size=(self.n_items, self.n_factors))
        
        if self.learning == 'als':
            ## Try one of these. apply_long_axis came from Explicit_RS_MF_ALS()
            ##                                             Daniel Nee code
            
            self.user_vecs = abs(np.random.randn(self.n_users, self.n_factors))
            self.item_vecs = abs(np.random.randn(self.n_items, self.n_factors))
            
            #self.user_vecs = np.apply_along_axis(normalize_row, 1, self.user_vecs) # axis=1, across rows
            #self.item_vecs = np.apply_along_axis(normalize_row, 1, self.item_vecs) # axis=1, across rows
            
            self.partial_train(n_iter)
            
        elif self.learning == 'sgd':
            self.user_bias = np.zeros(self.n_users)
            self.item_bias = np.zeros(self.n_items)
            self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
            self.partial_train(n_iter)
    
    
    def partial_train(self, n_iter):
        """ 
        Train model for n_iter iterations. 
        Can be called multiple times for further training.
        Remains in the while loop for a number of iterations, calculated from
        the contents of the iter_array in calculate_learning_curve()
        """
        
        ctr = 1
        while ctr <= n_iter:

            if self.learning == 'als':
                self.user_vecs = self.als_step(self.user_vecs, 
                                               self.item_vecs, 
                                               self.ratings, 
                                               self.user_fact_reg, 
                                               type='user')
                self.item_vecs = self.als_step(self.item_vecs, 
                                               self.user_vecs, 
                                               self.ratings, 
                                               self.item_fact_reg, 
                                               type='item')
                
            elif self.learning == 'sgd':
                self.training_indices = np.arange(self.n_samples)
                np.random.shuffle(self.training_indices)
                self.sgd()
            ctr += 1

    def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 ratings,
                 _lambda,
                 type='user'):
        """
        ALS algo step.
        Solve for the latent vectors specified by type parameter: user or item
        """
        
        #lv_shape = latent_vectors.shape[0] ## debug
        
        if type == 'user':

            for u in range(latent_vectors.shape[0]): # latent_vecs ==> user_vecs
                #r_u = ratings[u, :] ## debug
                #fvT = fixed_vecs.T ## debug
                idx = self.nonZero[u,:] # get the uth user profile with booleans 
                                        # (True when there are ratings) based on 
                                        # ratingsMatrix, n x 1
                nz_fixed_vecs = fixed_vecs[idx,] # get the item vector entries, non-zero's x f
                YTY = nz_fixed_vecs.T.dot(nz_fixed_vecs) # fixed_vecs are item_vecs
                lambdaI = np.eye(YTY.shape[0]) * _lambda
                
                latent_vectors[u, :] = \
                    solve( (YTY + lambdaI) , nz_fixed_vecs.T.dot (ratings[u, idx] ) )

                '''
                ## debug
                if u <= 10: 
                    print('user vecs1', nz_fixed_vecs)
                    print('user vecs1', fixed_vecs, '\n', ratings[u, :] )
                    print('user vecs2', fixed_vecs.T.dot (ratings[u, :] ))
                    print('reg', YTY, '\n', lambdaI)
                    print('new user vecs:\n', latent_vectors[u, :])
                ## debug
                '''
                    
        elif type == 'item':
            
            for i in range(latent_vectors.shape[0]): #latent_vecs ==> item_vecs
                idx = self.nonZero[:,i] # get the ith item "profile" with booleans 
                                        # (True when there are ratings) based on 
                                        # ratingsMatrix, n x 1
                nz_fixed_vecs = fixed_vecs[idx,] # get the item vector entries, non-zero's x f
                XTX = nz_fixed_vecs.T.dot(nz_fixed_vecs) # fixed_vecs are user_vecs
                lambdaI = np.eye(XTX.shape[0]) * _lambda
                latent_vectors[i, :] = \
                    solve( (XTX + lambdaI) , nz_fixed_vecs.T.dot (ratings[idx, i] ) )

        return latent_vectors

    def sgd(self):
        ''' run sgd algo '''
        
        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            prediction = self.predict(u, i)
            e = (self.ratings[u,i] - prediction) # error
            
            # Update biases
            self.user_bias[u] += self.sgd_alpha * \
                                (e - self.sgd_beta * self.user_bias[u])
            self.item_bias[i] += self.sgd_alpha * \
                                (e - self.sgd_beta * self.item_bias[i])
            
            # Create copy of row of user_vecs since we need to update it but
            #    use older values for update on item_vecs, 
            #    so make a deepcopy of previous user_vecs
            previous_user_vecs = deepcopy(self.user_vecs[u, :])
            
            # Update latent factors
            self.user_vecs[u, :] += self.sgd_alpha * \
                                    (e * self.item_vecs[i, :] - \
                                     self.sgd_beta * self.user_vecs[u,:])
            self.item_vecs[i, :] += self.sgd_alpha * \
                                    (e * previous_user_vecs - \
                                     self.sgd_beta * self.item_vecs[i,:])           
    
    def calculate_learning_curve(self, iter_array, test):
        """
        Keep track of MSE as a function of training iterations.
        
        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of 
            the learning curve. e.g. [1, 5, 10, 20]
        test : (2D ndarray)
            Testing dataset (assumed to be user x item)
        
        
        
        This function creates two new class attributes:
        
        train_mse : (list)
            Training data MSE values for each value of iter_array
        test_mse : (list)
            Test data MSE values for each value of iter_array
        """
        
        print()
        if self.learning == 'als':
            print('Runtime parameters:\nn_factors=%d, user_reg=%.5f, item_reg=%.5f,'
                  ' max_iters=%d,'
                  ' \nratings matrix: %d users X %d items' %\
                  (self.n_factors, self.user_fact_reg, self.item_fact_reg, 
                   self.max_iters, self.n_users, self.n_items))
        if self.learning == 'sgd':
            print('Runtime parameters:\nn_factors=%d, learning_rate alpha=%.3f,'
                  ' reg beta=%.5f, max_iters=%d, sgd_random=%s'
                  ' \nratings matrix: %d users X %d items' %\
                  (self.n_factors, self.sgd_alpha, self.sgd_beta, 
                   self.max_iters, self.sgd_random, self.n_users, self.n_items))
        print()       
        
        iter_array.sort()
        self.train_mse =[]
        self.test_mse = []
        iter_diff = 0
        
        start_time = time.time()
        stop_time = time.time()
        elapsed_time = (stop_time-start_time) #/60
        print ( 'Elapsed train/test time %.2f secs' % elapsed_time )
        error_list=[]        
        
        # Loop through number of iterations
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print ('Iteration: {}'.format(n_iter))
            if i == 0:
                self.train(n_iter - iter_diff) # init training, run first iter
            else:
                self.partial_train(n_iter - iter_diff) # run more iterations
                    # .. as you go from one element of iter_array to another

            predictions = self.predict_all() # calc dot product of p and qT
            # calc train  errors -- predicted vs actual
            error_training=[self.get_mse(predictions, self.ratings)]
            #if self.ratings>0:
                #error_list.append(predictions-self.ratings)
            self.train_mse += error_training
            if test.any() > 0: # check if test matrix is all zeroes ==> Train Only
                               # If so, do not calc mse and avoid runtime error   
                # calc test errors -- predicted vs actual
                error_testing=[self.get_mse(predictions, test)] 
                #if test>0:
                    #error_list.append(predictions-test)
                self.test_mse += error_testing
            else:
                self.test_mse = ['n/a']
            if self._v:
                print ('Train mse: ' + str(self.train_mse[-1]))
                if self.test_mse != ['n/a']:
                    print ('Test mse: ' + str(self.test_mse[-1]))
            iter_diff = n_iter
            
            stop_time = time.time()
            elapsed_time = (stop_time-start_time) #/60
            print ( 'Elapsed train/test time %.2f secs' % elapsed_time ) 
        if self.learning == 'als':  
            pickle.dump(error_list, open( "data_2/sq_error_ALS_2_0.02_0.02.p", "wb" ))  
        if self.learning == 'sgd':
            pickle.dump(error_list, open( "data_2/sq_error_SGD_2_1.p", "wb" ))

    def predict(self, u, i):
        """ Single user and item prediction """
        
        if self.learning == 'als':
            return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        elif self.learning == 'sgd':
            prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
            prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
            return prediction
    
    def predict_all(self):
        """ Predict ratings for every user and item """
        
        predictions = np.zeros((self.user_vecs.shape[0], 
                                self.item_vecs.shape[0]))
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)
        return predictions    

    def get_mse(self, pred, actual):
        ''' Calc MSE between predicted and actual values '''
        
        # Ignore nonzero terms.
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return mean_squared_error(pred, actual)

def ratings_to_2D_matrix(ratings, m, n):
    '''
    creates a U-I matrix from the data
    ==>>  eliminates movies (items) that have no ratings!
    '''
    print('Summary Stats:')
    print()
    print(ratings.describe())
    ratingsMatrix = ratings.pivot_table(columns=['item_id'], index =['user_id'],
        values='rating', dropna = False) # convert to a U-I matrix format from file input
    ratingsMatrix = ratingsMatrix.fillna(0).values # replace nan's with zeroes
    ratingsMatrix = ratingsMatrix[0:m,0:n] # get rid of any users/items that have no ratings
    print()
    print('2D_matrix shape', ratingsMatrix.shape) # debug
    
    return ratingsMatrix

def file_info(df):
    ''' print file info/stats  '''
    print()
    print (df.head())
    
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    
    ratings = ratings_to_2D_matrix(df, n_users, n_items)
    
    print()
    print (ratings)
    print()
    print (str(n_users) + ' users')
    print (str(n_items) + ' items')
    
    sparsity = float(len(ratings.nonzero()[0]))
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    sparsity = 100 - sparsity
    print ('Sparsity: {:4.2f}%'.format(sparsity))
    return ratings

def train_test_split(ratings, TRAIN_ONLY):
    ''' split the data into train and test '''
    test = np.zeros(ratings.shape)
    train = deepcopy(ratings) # instead of copy()
    
    ## setting the size parameter for random.choice() based on dataset size
    if len(ratings) < 10: # critics
        size = 1
    elif len(ratings) < 1000: # ml-100k
        size = 20
    else:
        size = 40 # ml-1m
        
    #print('size =', size) ## debug
    
    if TRAIN_ONLY == False:
        np.random.seed(0) # do not randomize the random.choice() in this function,
                          # let ALS or SGD make the decision to randomize
                          # Note: this decision can be reset with np.random.seed()
                          # .. see code at the end of this for loop
        for user in range(ratings.shape[0]): ## CES changed all xrange to range for Python v3
            test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                            size=size, 
                                            replace=True) #False)
            # When replace=False, size for ml-100k = 20, for critics = 1,2, or 3
            # Use replace=True for "better" results
            
            '''
            np.random.choice() info ..
            https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
            
            random.choice(a, size=None, replace=True, p=None)
            
            Parameters --
            a:         1-D array-like or int
            If an ndarray, a random sample is generated from its elements. 
            If an int, the random sample is generated as if it were np.arange(a)
            
            size:      int or tuple of ints, optional
            Output shape. If the given shape is, e.g., (m, n, k), 
            then m * n * k samples are drawn. 
            Default is None, in which case a single value is returned.
        
            replace:   boolean, optional
            Whether the sample is with or without replacement. 
            Default is True, meaning that a value of a can be selected multiple times.
        
            p:        1-D array-like, optional
            The probabilities associated with each entry in a. If not given, 
            the sample assumes a uniform distribution over all entries in a.
    
            Returns
            samples:   single item or ndarray
            The generated random samples
            
            '''
            
            train[user, test_ratings] = 0.
            test[user, test_ratings] = ratings[user, test_ratings]
            
        # Test and training are truly disjoint
        assert(np.all((train * test) == 0)) 
        np.random.seed() # allow other functions to randomize
    
    #print('TRAIN_ONLY (in split) =', TRAIN_ONLY) ##debug
    
    return train, test
    

def test_train_info(test, train):
    ''' print test/train info   '''

    print()
    print ('Train info: %d rows, %d cols' % (len(train), len(train[0])))
    print ('Test info: %d rows, %d cols' % (len(test), len(test[0])))
    
    test_count = 0
    for i in range(len(test)):
        for j in range(len(test[0])):
            if test[i][j] !=0:
                test_count += 1
                #print (i,j,test[i][j]) # debug
    print('test ratings count =', test_count)
    
    train_count = 0
    for i in range(len(train)):
        for j in range(len(train[0])):
            if train[i][j] !=0:
                train_count += 1
                #print (i,j,train[i][j]) # debug
    
    total_count = test_count + train_count
    print('train ratings count =', train_count)
    print('test + train count', total_count)
    print('test/train percentages: %0.2f / %0.2f' 
          % ( (test_count/total_count)*100, (train_count/total_count)*100 ))
    print()


def plot_learning_curve(iter_array, model):
    ''' plot the error curve '''
    
    ## Note: the iter_array can cause plots to NOT 
    ##    be smooth! If matplotlib can't smooth, 
    ##    then print/plot results every 
    ##    max_num_iterations/10 (rounded up)
    ##    instead of using an iter_array list
    
    #print('model.test_mse', model.test_mse) # debug
    if model.test_mse != ['n/a']:
        plt.plot(iter_array, model.test_mse, label='Test', linewidth=3)
    plt.plot(iter_array, model.train_mse, label='Train', linewidth=3)

    plt.xticks(fontsize=10); # 16
    plt.xticks(iter_array, iter_array)
    plt.yticks(fontsize=10);
    
    axes = plt.gca()
    axes.grid(True) # turns on grid
    
    if model.learning == 'als':
        runtime_parms = \
            'shape=%s, n_factors=%d, user_fact_reg=%.3f, item_fact_reg=%.3f'%\
            (model.ratings.shape, model.n_factors, model.user_fact_reg, model.item_fact_reg)
            #(train.shape, model.n_factors, model.user_fact_reg, model.item_fact_reg)
        plt.title("ALS Model Evaluation\n%s" % runtime_parms , fontsize=10) 
    elif model.learning == 'sgd':
        runtime_parms = \
            'shape=%s, num_factors K=%d, alpha=%.3f, beta=%.3f'%\
            (model.ratings.shape, model.n_factors, model.sgd_alpha, model.sgd_beta)
            #(train.shape, model.n_factors, model.learning_rate, model.user_fact_reg)
        plt.title("SGD Model Evaluation\n%s" % runtime_parms , fontsize=10)         
    
    plt.xlabel('Iterations', fontsize=15);
    plt.ylabel('Mean Squared Error', fontsize=15);
    plt.legend(loc='best', fontsize=15, shadow=True) # 'best', 'center right' 20
    
    plt.show()
    
def from_file_to_dict(path, datafile, itemfile):
    ''' Load user-item matrix from specified file 
        
        Parameters:
        -- path: directory path to datafile and itemfile
        -- datafile: delimited file containing userid, itemid, rating
        -- itemfile: delimited file that maps itemid to item name
        
        Returns:
        -- prefs: a nested dictionary containing item ratings for each user
    
    '''
    
    # Get movie titles, place into movies dictionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile, encoding = 'iso8859') as myfile: 
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2]
                movies[id]=title.strip()
    
    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(movies), line, id, title)
        return {}
    except Exception as ex:
        print (ex)
        print (len(movies))
        return {}
    
    # Load data into a nested dictionary
    prefs={}
    for line in open(path+'/'+ datafile):
        #print(line, line.split('\t')) #debug
        (user,movieid,rating,ts)=line.split('\t')
        user = user.strip() # remove spaces
        movieid = movieid.strip() # remove spaces
        prefs.setdefault(user,{}) # make it a nested dicitonary
        prefs[user][movies[movieid]]=float(rating)
    
    #return a dictionary of preferences
    return prefs

def from_file_to_2D(path, genrefile, itemfile):
    ''' Load feature matrix from specified file 
        Parameters:
        -- path: directory path to datafile and itemfile
        -- genrefile: delimited file that maps genre to genre index
        -- itemfile: delimited file that maps itemid to item name and genre
        
        Returns:
        -- movies: a dictionary containing movie titles (value) for a given movieID (key)
        -- genres: dictionary, key is genre, value is index into row of features array
        -- features: a 2D list of features by item, values are 1 and 0;
                     rows map to items and columns map to genre
                     returns as np.array()
    
    '''
    # Get movie titles, place into movies dictionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile, encoding='iso8859') as myfile: 
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2]
                movies[id]=title.strip()
    
    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(movies), line, id, title)
        return {}
    except ValueError as ex:
        print ('ValueError', ex)
        print (len(movies), line, id, title)
    except Exception as ex:
        print (ex)
        print (len(movies))
        return {}
    
    ##
    # Get movie genre from the genre file, place into genre dictionary indexed by genre index
    genres={} # key is genre index, value is the genre string
    ##
    ## Your code here!!
    try:
        with open (path + '/' + genrefile, encoding='iso8859') as myfile: 
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2]
                genres[int(title.strip())]= id
    except UnicodeDecodeError as ex:
         print (ex)
         print (len(genres), line, id, title)
         return {}
    except ValueError as ex:
         print ('ValueError', ex)
         print (len(genres), line, id, title)
    except Exception as ex:
         print (ex)
         print (len(genres))
         return {}
    ##
    ##
    
    # Load data into a nested 2D list
    features = []
    start_feature_index = 5
    try: 
        for line in open(path+'/'+ itemfile, encoding='iso8859'):
            fields = line.split('|')[start_feature_index:]
            row = []
            for feature in fields:
                row.append(int(feature))
            features.append(row)
        features = np.array(features)
        print(features)
    except Exception as ex:
        print (ex)
        print ('Proceeding with len(features)', len(features))
        #return {}
    
    #return features matrix
    return movies, genres, features  

def to_array(prefs):
    ''' convert prefs dictionary into 2D list '''
    R = prefs_to_2D_list(prefs)
    R = np.array(R)
    print ('to_array -- height: %d, width: %d' % (len(R), len(R[0]) ) )
    return R

def to_string(features):
    ''' convert features np.array into list of feature strings '''
    
    feature_str = []
    for i in range(len(features)):
        row = ''
        for j in range(len (features[0])):
            row += (str(features[i][j]))
        feature_str.append(row)
    print ('to_string -- height: %d, width: %d' % (len(feature_str), len(feature_str[0]) ) )
    return feature_str

def to_docs(features_str, genres):
    ''' convert feature strings to a list of doc strings for TFIDF '''
    
    feature_docs = []
    for doc_str in features_str:
        row = ''
        for i in range(len(doc_str)):
            if doc_str[i] == '1':
                row += (genres[i] + ' ') # map the indices to the actual genre string
        feature_docs.append(row.strip()) # and remove that pesky space at the end
        
    print ('to_docs -- height: %d, width: varies' % (len(feature_docs) ) )
    return feature_docs

def cosine_sim(docs):
    ''' Perofmrs cosine sim calcs on features list, aka docs in TF-IDF world
    
        Parameters:
        -- docs: list of item features
     
        Returns:   
        -- list containing cosim_matrix: item_feature-item_feature cosine similarity matrix 
    
    
    '''
    
    #print()
    #print('## Cosine Similarity calc ##')
    #print()
    #print('Documents:', docs[:10])
    
    #print()
    #print ('## Count and Transform ##')
    #print()
    
    # get the TFIDF vectors
    tfidf_vectorizer = TfidfVectorizer() # orig
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    #print (tfidf_matrix.shape, type(tfidf_matrix)) # debug


    #print()
    #print('Document similarity matrix:')
    cosim_matrix = cosine_similarity(tfidf_matrix[0:], tfidf_matrix)
    #print (type(cosim_matrix), len(cosim_matrix))
    #print()
    #print(cosim_matrix[0:6])
    #print()
    
    
    #print('Examples of similarity angles')
    if tfidf_matrix.shape[0] > 2:
        for i in range(6):
            cos_sim = cosim_matrix[1][i] #(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix))[0][i] 
            if cos_sim > 1: cos_sim = 1 # math precision creating problems!
            angle_in_radians = math.acos(cos_sim)
            #print('Cosine sim: %.3f and angle between documents 2 and %d: ' 
                  #% (cos_sim, i+1), end=' ')
            #int ('%.3f degrees, %.3f radians' 
                   #% (math.degrees(angle_in_radians), angle_in_radians))
    
    
    return cosim_matrix

def movie_to_ID(movies):
    ''' converts movies mapping from "id to title" to "title to id" '''
    
    movie_to_ID_matrix = {}
    
    for id_movie, movie in movies.items():
        movie_to_ID_matrix[movie] = id_movie
    
        
    
    return movie_to_ID_matrix

def prefs_to_2D_list(prefs):
    '''
    Convert prefs dictionary into 2D list used as input for the MF class
    
    Parameters: 
        prefs: user-item matrix as a dicitonary (dictionary)
        
    Returns: 
        ui_matrix: (list) contains user-item matrix as a 2D list
        
    '''
    ui_matrix = []
    
    user_keys_list = list(prefs.keys())
    num_users = len(user_keys_list)
    #print (len(user_keys_list), user_keys_list[:10]) # debug
    
    itemPrefs = transformPrefs(prefs) # traspose the prefs u-i matrix
    item_keys_list = list(itemPrefs.keys())
    num_items = len(item_keys_list)
    #print (len(item_keys_list), item_keys_list[:10]) # debug
    
    sorted_list = True # <== set manually to test how this affects results
    
    if sorted_list == True:
        user_keys_list.sort()
        item_keys_list.sort()
        print ('\nsorted_list =', sorted_list)
        
    # initialize a 2D matrix as a list of zeroes with 
    #     num users (height) and num items (width)
    
    for i in range(num_users):
        row = []
        for j in range(num_items):
            row.append(0.0)
        ui_matrix.append(row)
          
    # populate 2D list from prefs
    # Load data into a nested list

    for user in prefs:
        for item in prefs[user]:
            user_idx = user_keys_list.index(user)
            movieid_idx = item_keys_list.index(item) 
            
            try: 
                # make it a nested list
                ui_matrix[user_idx][movieid_idx] = prefs [user][item] 
            except Exception as ex:
                print (ex)
                print (user_idx, movieid_idx)   
                
    # return 2D user-item matrix
    return ui_matrix

def user_preference(ratings, movie_to_ID,  features):
    '''
        Calculates the prefernce matrix 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- features: an np.array whose height is based on number of items
                     and width equals the number of unique features (e.g., genre)
        -- movie_title_to_id: dictionary that maps movie title to movieid
        
        Returns:
        -- feature_pref: feature matrix
        --avg_pref: average values of the feature matrix
        --avg_avg_pref: average value of the preference matrix
        
    '''
    
    feature_pref = []
    avg_pref = []
    total_sum = 0
    avg_avg_pref = []
    for movie, id_movie in movie_to_ID.items():
        movie_feature = []
        for fts in features[int(id_movie)-1]:
                if fts == 0 or movie not in ratings:
                    movie_feature.append(0)
                else:
                    movie_feature.append(ratings[movie])
       
        feature_pref.append(movie_feature)
    
    for j in range(len(feature_pref[0])):
        sum_col = 0
        count = 0
        for i in range(len(feature_pref)):
            sum_col += feature_pref[i][j]
            if feature_pref[i][j] != 0:
                count = count +1
            
        avg_pref.append(sum_col)
        total_sum += sum_col
        if count != 0:
            avg_avg_pref.append(sum_col/count)
        else:
            avg_avg_pref.append(0)
    
    for i in range(len(avg_pref)):
        avg_pref[i] = avg_pref[i] / total_sum
    
        
    return feature_pref, avg_pref, avg_avg_pref

def get_FE_recommendations(prefs, features, movie_title_to_id, user):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- features: an np.array whose height is based on number of items
                     and width equals the number of unique features (e.g., genre)
        -- movie_title_to_id: dictionary that maps movie title to movieid
        -- user: string containing name of user requesting recommendation        
        
        Returns:
        -- rankings: A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    
    ratings = prefs[user]
    feature_pref, avg_pref, avg_avg_pref = user_preference(ratings, movie_title_to_id, features)
    scores={}
    #print(avg_pref)
 
    
    for item, id_item in movie_title_to_id.items():
        if item in ratings: continue
        non_zero = 0
        normalized_weight = []
        prediction = 0
        id_movie = int(id_item) -1
        for i in  range(len(features[0])):
            if features[id_movie][i]!= 0:
                non_zero += avg_pref[i]
        
        if non_zero == 0: continue
    
        for i in  range(len(features[0])):
            if features[id_movie][i]!= 0:
                normalized_weight.append((avg_pref[i]/non_zero)*avg_avg_pref[i])
        
        prediction = sum(normalized_weight)
        scores[item] = prediction
        
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores



        
def get_TFIDF_recommendations(prefs,cosim_matrix, user, threshold):
    '''
        Calculates recommendations for a given user TFIDF

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- cosim_matrix: list containing item_feature-item_feature cosine similarity matrix 
        -- user: string containing name of user requesting recommendation 
        - threshold: threshold value
        
        Returns:
        -- ranknigs: A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    
    userRatings=prefs[user]
    scores={}
    totalSim={}
    threshold = 0
    
    #loop over all items
  
    for (item, sim_matrix) in cosim_matrix.items():
        
    
        # Ignore if this user has already rated this item
        if item in userRatings: continue
    
            
        for movie, similarity in sim_matrix.items():
            # ignore scores of zero or lower
            if similarity <= threshold: continue
            if movie not in userRatings: continue
            # Weighted sum of rating times similarity
            rating = userRatings[movie]
            scores.setdefault(item,0)
            scores[item]+=similarity*rating
            # Sum of all the similarities
            totalSim.setdefault(item,0)
            totalSim[item]+=similarity
        
        
  
    # Divide each total score by total weighting to get an average

    rankings=[(score/totalSim[item],item) for item,score in scores.items()]    
  
    # Return the rankings from highest to lowest
    rankings.sort()
    rankings.reverse()
    
    #print(rankings)
    return rankings



def get_hybrid_recommendations(prefs,cosim_matrix, item_sim_matrix, user, Weight_parameter):
    '''
        Calculates hybrid recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- cosim_matrix: list containing item_feature-item_feature cosine similarity matrix 
        -- item_sim_matrix: item-based similarity matrix
        -- user: string containing name of user requesting recommendation 
        -- item_sim_matrix: item similarity matrix
        --weight_parameter: weight parameter of CF
        
        Returns:
        -- ranknigs: A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    
    userRatings=prefs[user]
    scores={}
    totalSim={}
    threshold = 0
    
    #loop over all items
  
    for (item, sim_matrix) in cosim_matrix.items():
        
    
        # Ignore if this user has already rated this item
        if item in userRatings: continue
    
        for (movie, similarity)  in sim_matrix.items():
            
            if movie not in userRatings: continue
               
            # Weighted sum of rating times similarity
            
            if similarity <= 0:
                if item in item_sim_matrix[movie]:
                    if item_sim_matrix[movie][item] > threshold:
                        #print("matrix")
                        #print(item_sim_matrix[movie])
                        #print("item")
                        #print(item_sim_matrix[movie][item])
                        #print(item_sim_matrix[item])
                        #print(item)
                        #print(movie)
                        #print(item_sim_matrix[item][movie])
                        rating = userRatings[movie]
                        scores.setdefault(item,0)
                        scores[item]+= item_sim_matrix[movie][item] *rating * Weight_parameter
                        # Sum of all the similarities
                        totalSim.setdefault(item,0) 
                        totalSim[item]+= item_sim_matrix[movie][item]
                        #print(totalSim)
            else: 
                rating = userRatings[movie]
                scores.setdefault(item,0)
                scores[item]+= similarity*rating
                # Sum of all the similarities
                totalSim.setdefault(item,0)
                totalSim[item]+=similarity
                #print(totalSim)
                
              
            
        
        
  
    # Divide each total score by total weighting to get an average


    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]    
  
    # Return the rankings from highest to lowest
    rankings.sort( )
    rankings.reverse( )
    
    #print(rankings)
    return rankings

def get_hybrid_recommendations_item(prefs,cosim_matrix, item_sim_matrix, user, Weight_parameter, item):
    '''
        Calculates recommendations for a given user for a given movie

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- cosim_matrix: list containing item_feature-item_feature cosine similarity matrix 
        -- item_sim_matrix: item-based similarity matrix

        -- user: string containing name of user requesting recommendation 
        - top_N: number of top matches to return
        
        Returns:
        -- ranknigs: A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    
    userRatings=prefs[user]
    scores={}
    totalSim={}
    sim_matrix = cosim_matrix[item]
    threshold = 0
    
    
    
        
    for movie, similarity in sim_matrix.items():
        if movie not in userRatings: continue
        if similarity <= threshold: 
            if item in item_sim_matrix[movie]:
                if item_sim_matrix[movie][item] > threshold:
                    rating = userRatings[movie]
                    scores.setdefault(item, 0)
                    scores[item] += item_sim_matrix[movie][item] * rating * Weight_parameter
                    totalSim.setdefault(item, 0)
                    totalSim[item] += item_sim_matrix[movie][item]
        else:
            rating = userRatings[movie]
            scores.setdefault(item,0)
            scores[item]+=similarity*rating
                # Sum of all the similarities
            totalSim.setdefault(item,0)
            totalSim[item]+=similarity
      
        
      
    
        
        
  
    # Divide each total score by total weighting to get an average

    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]    
  
    # Return the rankings from highest to lowest
    rankings.sort( )
    rankings.reverse( )
    
    #print(rankings)
    return rankings  

def get_TFIDF_recommendations_item(prefs,cosim_matrix, user, movie):
    '''
        Calculates recommendations for a given user for a given movie in LOOCVSIM

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- cosim_matrix: list containing item_feature-item_feature cosine similarity matrix 
        -- user: string containing name of user requesting recommendation 
        - movie: movie removed for LOOCVSIM
        
        Returns:
        -- ranknigs: A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    
    userRatings=prefs[user]
    scores={}
    totalSim={}
    sim_matrix = cosim_matrix[movie]
    threshold = 0
    
  
        
    
    
            
    for movies, similarity in sim_matrix.items():
            # ignore scores of zero or lower
        if similarity <= threshold: continue
        if movies not in userRatings: continue
        # Weighted sum of rating times similarity
        rating = userRatings[movies]
        scores.setdefault(movie,0)
        scores[movie]+=similarity*rating
            # Sum of all the similarities
        totalSim.setdefault(movie,0)
        totalSim[movie]+=similarity
        
        
  
    # Divide each total score by total weighting to get an average

    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]    
  
    # Return the rankings from highest to lowest
    rankings.sort( )
    rankings.reverse( )
    
    #print(rankings)
    return rankings  


def top_N(rankings, top_N):
    """
    top_N: produce top N recommendations for users
         
    Parameters:
        rankings: the list of rankings
        top_N: how many recommendations the users want
         
        Returns:
         error_total: MSE, or MAE, or RMSE totals for this set of conditions
         error_list: list of actual-predicted differences
    """
        
    top_N_matches = []
    count = 0 
    while (len(top_N_matches) != top_N and count < len(rankings)):
        top_N_matches.append(rankings[count])
        count = count + 1
    return top_N_matches



def loo_cv_sim_TFIDF(prefs, sim_matrix):
        """
        Leave-One_Out Evaluation: evaluates recommender system ACCURACY for TFIDF
         
         Parameters:
         prefs: critics, etc.
         sim_matrix: pre-computed similarity matrix
         
        Returns:
         error_total: MSE, or MAE, or RMSE totals for this set of conditions
         error_list: list of actual-predicted differences
        """
        
        start_time = time.time()
        prefs_copy = cp.deepcopy(prefs)
        error_list = []
        error_total = 0
        count = 0
        error_mse = 0
        error_rmse=0
        error_list_rmse = []
        error_mae = 0
        error_list_mae = []
        c=0
        
        #print(sim_matrix)
 
    
        for person, item in prefs.items():
            c = c+1
            for movie in item:
                delete = prefs_copy[person].pop(movie)
                rec =  get_TFIDF_recommendations_item(prefs_copy, sim_matrix, person, movie)
                #rint(person)
                #print()
              
    
             
                   
                prefs_copy[person][movie] = delete
                in_value = False
                    
               
               
                for ratings in rec:
                    in_value = False
                    if movie == ratings[1]:
                            in_value = True
                            prediction = ratings[0]
                            real_val = prefs[person][movie]
                            err = pow((prediction - real_val),2)
                            error_list.append (err)
                            error_total += err
                            count += 1
                            error_mse += err
                            error_mae += abs(prediction - real_val)
                            error_rmse += err
                            error_list_rmse.append(err)
                            error_list_mae.append(error_mae)
            if((c+1) % 10 == 0):
                
                print("Number of users processed: ", (c+1) )
                if count == 0:
                    time_per_user=(time.time() - start_time)/(c+1)
                    print("===> {} secs for {} users, {} time per user: ".format(round(time.time() - start_time,2), c+1, round(time_per_user,3)))
                    print("MSE:", "%.10f" %(error_mse),  ", MAE:", "%.10f" % (error_mae),  ", RMSE:", "%.10f" % (sqrt(error_rmse)))
                else:
                    print("===> {} secs for {} users, {} time per user: ".format(round(time.time() - start_time,2), c+1, round((time.time() - start_time)/count),3))
                    print("MSE:", "%.10f" %(error_mse/count),  ", MAE:", "%.10f" % (error_mae/count),  ", RMSE:", "%.10f" % (sqrt(error_rmse/count)))
                 
                                                            
                            
        if count == 0:
            print("MSE:", "%.10f" %(error_mse),  ", MAE:", "%.10f" % (error_mae),  ", RMSE:", "%.10f" % (sqrt(error_rmse)), ", Coverage:", "%.10f" % (len(error_list)))
            return error_mse, error_mae, error_rmse, len(error_list)
        else:
            print("MSE:", "%.10f" %(error_mse/count),  ", MAE:", "%.10f" % (error_mae/count),  ", RMSE:", "%.10f" % (sqrt(error_rmse/count)), ", Coverage:", "%.10f" % (len(error_list)))
            return error_mse/count, error_mae/count, error_rmse/count, len(error_list)#Main


def loo_cv_sim_hybrid(prefs, sim_matrix, item_matrix, Weight_parameter):
        """
        Leave-One_Out Evaluation: evaluates recommender system ACCURACY
         
         Parameters:
         prefs: critics, etc.
         sim: distance, pearson, etc.
         algo: user-based recommender, item-based recommender, etc.
         sim_matrix: pre-computed similarity matrix
         
        Returns:
         error_total: MSE, or MAE, or RMSE totals for this set of conditions
         error_list: list of actual-predicted differences
        """
        
        start_time = time.time()
        prefs_copy = cp.deepcopy(prefs)
        error_list = []
        error_total = 0
        count = 0
        error_mse = 0
        error_rmse=0
        error_list_rmse = []
        error_mae = 0
        error_list_mae = []
        c=0
        
        #print(sim_matrix)
        
    
            
    
        for person, item in prefs.items():
                c = c+1
                for movie in item:
                    delete = prefs_copy[person].pop(movie)
                    rec =  get_hybrid_recommendations_item(prefs_copy, sim_matrix, item_matrix, person, Weight_parameter, movie)
                    #rint(person)
                    #print()
                  
        
                 
                       
                    prefs_copy[person][movie] = delete
                    in_value = False
                        
                   
                   
                    for ratings in rec:
                        in_value = False
                        if movie == ratings[1]:
                                in_value = True
                                prediction = ratings[0]
                                real_val = prefs[person][movie]
                                err = pow((prediction - real_val),2)
                                error_list.append (err)
                                error_total += err
                                count += 1
                                error_mse += err
                                error_mae += abs(prediction - real_val)
                                error_rmse += err
                                error_list_rmse.append(err)
                                error_list_mae.append(error_mae)
                if((c+1) % 10 == 0):
                    
                    print("Number of users processed: ", (c+1) )
                    if count == 0:
                        time_per_user=(time.time() - start_time)/(c+1)
                        print("===> {} secs for {} users, {} time per user: ".format(round(time.time() - start_time,2), c+1, round(time_per_user,3)))
                        print("MSE:", "%.10f" %(error_mse),  ", MAE:", "%.10f" % (error_mae),  ", RMSE:", "%.10f" % (sqrt(error_rmse)))
                    else:
                        print("===> {} secs for {} users, {} time per user: ".format(round(time.time() - start_time,2), c+1, round((time.time() - start_time)/count),3))
                        print("MSE:", "%.10f" %(error_mse/count),  ", MAE:", "%.10f" % (error_mae/count),  ", RMSE:", "%.10f" % (sqrt(error_rmse/count)))
                     
    
    
        pickle.dump(error_list, open( "data_2/sq_error_hybrid__0.p", "wb" ))                                                    
                                
        if count == 0:
            print("MSE:", "%.10f" %(error_mse),  ", MAE:", "%.10f" % (error_mae),  ", RMSE:", "%.10f" % (sqrt(error_rmse)), ", Coverage:", "%.10f" % (len(error_list)))
            return error_mse, error_mae, error_rmse, len(error_list)
        else:
            print("MSE:", "%.10f" %(error_mse/count),  ", MAE:", "%.10f" % (error_mae/count),  ", RMSE:", "%.10f" % (sqrt(error_rmse/count)), ", Coverage:", "%.10f" % (len(error_list)))
            return error_mse/count, error_mae/count, error_rmse/count, len(error_list)#Main

def loo_cv_sim_FE(prefs, features, movie_title_to_id):
        """
        Leave-One_Out Evaluation: evaluates recommender system ACCURACY
         
         Parameters:
         prefs: critics, etc.
         sim: distance, pearson, etc.
         algo: user-based recommender, item-based recommender, etc.
         sim_matrix: pre-computed similarity matrix
         
        Returns:
             error_total: MSE, or MAE, or RMSE totals for this set of conditions
         error_list: list of actual-predicted differences
        """
        
        start_time = time.time()
        prefs_copy = cp.deepcopy(prefs)
        error_list = []
        error_total = 0
        count = 0
        error_mse = 0
        error_rmse=0
        error_list_rmse = []
        error_mae = 0
        error_list_mae = []
        c=0
        
        #print(sim_matrix)
        
 
    
        for person, item in prefs.items():
            c = c+1
            for movie in item:
                delete = prefs_copy[person].pop(movie)
                rec =  get_FE_recommendations(prefs_copy, features, movie_title_to_id, person)
            
    
             
                   
                prefs_copy[person][movie] = delete
                in_value = False
                    
               
               
                for ratings in rec:
                    in_value = False
                    if movie == ratings[0]:
                            in_value = True
                            prediction = ratings[1]
                            real_val = prefs[person][movie]
                            err = pow((prediction - real_val),2)
                            error_list.append (err)
                            error_total += err
                            count += 1
                            error_mse += err
                            error_mae += abs(prediction - real_val)
                            error_rmse += err
                            error_list_rmse.append(err)
                            error_list_mae.append(error_mae)
            if((c+1) % 10 == 0):
                
                print("Number of users processed: ", (c+1) )
                if count == 0:
                    time_per_user=(time.time() - start_time)/(c+1)
                    print("===> {} secs for {} users, {} time per user: ".format(round(time.time() - start_time,2), c+1, round(time_per_user,3)))
                    print("MSE:", "%.10f" %(error_mse),  ", MAE:", "%.10f" % (error_mae),  ", RMSE:", "%.10f" % (sqrt(error_rmse)))
                else:
                    print("===> {} secs for {} users, {} time per user: ".format(round(time.time() - start_time,2), c+1, round((time.time() - start_time)/count),3))
                    print("MSE:", "%.10f" %(error_mse/count),  ", MAE:", "%.10f" % (error_mae/count),  ", RMSE:", "%.10f" % (sqrt(error_rmse/count)))
                 
        pickle.dump(error_list, open( "data_2/sq_error_FE.p", "wb" ))                                                   
                            
        if count == 0:
            print("MSE:", "%.10f" %(error_mse),  ", MAE:", "%.10f" % (error_mae),  ", RMSE:", "%.10f" % (sqrt(error_rmse)), ", Coverage:", "%.10f" % (len(error_list)))
            return error_mse, error_mae, error_rmse, len(error_list)
        else:
            print("MSE:", "%.10f" %(error_mse/count),  ", MAE:", "%.10f" % (error_mae/count),  ", RMSE:", "%.10f" % (sqrt(error_rmse/count)), ", Coverage:", "%.10f" % (len(error_list)))
            return error_mse/count, error_mae/count, error_rmse/count, len(error_list)#Main

    
def popular_items(prefs, filename):
    ''' Computes/prints popular items analytics    
        -- popular items: most rated (sorted by # ratings)
        -- popular items: highest rated (sorted by avg rating)
        -- popular items: highest rated items that have at least a 
                          "threshold" number of ratings
        
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- filename: string containing name of file being analyzed
        
        Returns:
        -- None

    '''
    #a dictionary with key = movie name, values = list [a,b], where a = total votes
    #b = average rating
    movie_ratings = {}
    avg_ratings = {}
    most_rated = {"Title": [], "#Ratings": [], "Avg Rating": []}
    highest_rated = {"Title": [], "Avg Rating": [], "#Ratings": []}
    overall_rated = {"Title": [], "Avg Rating": [], "#Ratings": []}
    
    for v in prefs.values() :
        for k in v.keys():
            #initialize movie ratings
            movie_ratings[k] = 0
            avg_ratings[k] = 0
    for v in prefs.values():
        for k,v in v.items():
            #count the number of ratings of each movies
            movie_ratings[k] += 1
            avg_ratings[k] = ((avg_ratings[k]* ( movie_ratings[k]-1)) + v ) / (movie_ratings[k])
            
   
    
    movie_ratings_sorted = sorted(movie_ratings.items(), key=lambda x: x[1], reverse = True) 
    avg_ratings_sorted = sorted(avg_ratings.items(), key=lambda x: x[1], reverse = True)    
   
    #most votes
    for i in movie_ratings_sorted:
        most_rated["Title"].append(i[0])
        most_rated["#Ratings"].append(i[1])
       
        for m in avg_ratings_sorted:
            if m[0] == i[0]:
              most_rated["Avg Rating"].append(m[1])
              
    #highest_rated
    for i in avg_ratings_sorted:
        highest_rated["Title"].append(i[0])
        highest_rated["Avg Rating"].append(i[1])
        for m in movie_ratings_sorted:
            if m[0] == i[0]:
              highest_rated["#Ratings"].append(m[1])
              
    #overall_rated
    for i in avg_ratings_sorted:            
            for m in movie_ratings_sorted:
                if m[0] == i[0]:
                    if m[1] >= 20:
                        overall_rated["Title"].append(i[0])
                        overall_rated["#Ratings"].append(m[1])
                        overall_rated["Avg Rating"].append(i[1])
       
        
              
    df = pd.DataFrame(most_rated)
    
    print("Popular Items -- Most Rated")
    print(df)
    
    df = pd.DataFrame(highest_rated)
    
    
    print("Popular Items -- Highest Rated")
    print(df)
    
    df = pd.DataFrame(overall_rated)
    
    print("Overall best rated items (number of ratings >= 20)")
    print(df)
    

def data_stats(prefs, filename):
    ''' Computes/prints descriptive analytics:
        -- Total number of users, items, ratings
        -- Overall average rating, standard dev (all users, all items)
        -- Average item rating, standard dev (all users)
        -- Average user rating, standard dev (all items)
        -- Matrix ratings sparsity
        -- Ratings distribution histogram (all users, all items)
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- filename: string containing name of file being analyzed
        
        Returns:
        -- None
    '''
    users = 0
    items = 0
    ratings = 0
    sum_rating = 0
    lst = []
    movie_set = set()
    movie_dict = {}
    #Calculating the number of users/items/ratings
    for person in prefs:
        users += 1
        movies = prefs[person]
        for movie in movies:
            movie_dict[movie] = 0
            movie_set.add(movie)
            ratings += 1
            sum_rating += prefs[person][movie]
            lst.append(prefs[person][movie])
    #Printing the number of users/items/ratings
    print("Number of users: "+ str(users))
    print("Number of items: "+ str(len(movie_set)))
    print("Number of ratings: "+ str(ratings))
    
    #Calculating the overall average ratings and standard deviation
    avg_rating = sum_rating/ratings
    sum_diff = 0
    for person in prefs:
        movies = prefs[person]
        for movie in movies:
            movie_dict[movie] = movie_dict[movie] + 1
            rate = prefs[person][movie]
            diff = (rate - avg_rating)**2
            sum_diff += diff
    std_dev = (sum_diff/ratings) ** (1/2)
    avg_rating = "%.2f" % avg_rating
    std_dev = "%.2f" % std_dev
    
    
    #Printing the overall average ratings and standarad deviation
    print("Overall average rating: " +str(avg_rating)+ " out of 5, and std dev of " +str(std_dev))
    
    #Average item ratings and standarad deviation
    movie_lst = []
    for person in prefs:
        movies = prefs[person]
        for movie in movies:
            if movie not in movie_lst:
                movie_lst.append(movie)
    
    #Calculating the average item rating per user and storing into a list
    itm_lst = []
    for movie in movie_lst:
        temp_sum = 0
        temp_count = 0
        for person in prefs:
            temp_lst = prefs[person].keys()
            if movie in temp_lst:
                temp_sum += prefs[person][movie]
                temp_count += 1
        temp_avg = temp_sum/temp_count
        itm_lst.append(temp_avg)
    
    
    #Calculating the average item ratings and standarad deviation
    itm_avg_sum = 0
    count = 0
    for avg in itm_lst:
        itm_avg_sum += avg
        count = count + 1
    itm_rating = itm_avg_sum/count
    sum_diff = 0
    std_dev = 0
    for avg in itm_lst:
        diff = (avg - itm_rating)**2
        sum_diff += diff
    std_dev = (sum_diff/count) ** (1/2)
    
    #Printing the average item ratings and standarad deviation
    itm_rating = "%.2f" % itm_rating
    std_dev = "%.2f" % std_dev
    print("Average item rating: " +str(itm_rating)+ " out of 5, and std dev of " +str(std_dev))
    
    #Calculating the average user ratings and standarad deviation
    usr_lst = []
    for person in prefs:
        movies = prefs[person]
        temp_sum = 0
        temp_count = 0
        for movie in movies:
            temp_sum += prefs[person][movie]
            temp_count += 1
        temp_avg = temp_sum/temp_count
        usr_lst.append(temp_avg)
    
    usr_avg_sum = 0
    for avg in usr_lst:
        usr_avg_sum += avg
    usr_rating = usr_avg_sum/users
    sum_diff = 0
    std_dev = 0
    for avg in usr_lst:
        diff = (avg - usr_rating)**2
        sum_diff += diff
    std_dev = (sum_diff/users) ** (1/2)
    #Printing the average user ratings and standarad deviation
    usr_rating = "%.2f" % usr_rating
    std_dev = "%.2f" % std_dev
    print("Average user rating: " +str(usr_rating)+ " out of 5, and std dev of " +str(std_dev))
    
    #Calculating the max sparsity
    max_rating_sparsity = 1 - (ratings/(users*len(movie_set)))
    max_rating_sparsity *= 100
    #Printing the max sparsity
    max_rating_sparsity = "%.2f" % max_rating_sparsity
    print("User-item Matrix Sparsity: "+ max_rating_sparsity+"%")
    
    #Calculating average numer of ratings 
    avg_rating_user = ratings / users
    
    
    #standard deviation
    std_avg_user = 0
    total = 0
    ratings_per_users = []
    min_ratings = 0
    max_ratings = 0
    median_ratings = 0
    
    for movies in prefs.values():
         num_ratings = len(movies)
         ratings_per_users.append(num_ratings)
         total += pow(num_ratings - avg_rating_user,2)
    
    std_avg_user = sqrt(total / users)
    
    
    print("Average number of ratings per users: %f, and std dev of %f  " %(avg_rating_user, std_avg_user))
    
    ratings_per_users.sort()
    
    
    min_ratings = ratings_per_users[0]
    max_ratings = ratings_per_users[users-1]
    
    if (users % 2 == 1):
         median_ratings = ratings_per_users[math.floor(users / 2)]
    else:
         #round down or up?
         median_ratings = (ratings_per_users[int(users/2)-1] + ratings_per_users[int(users/2)]) / 2
    print("Min, Max, Median number of ratings per users: %d, %d, %d" %(min_ratings, max_ratings, median_ratings))

    
    #calculate average number of ratings per item
    
    total_ratings_movie = 0
    total_std = 0
    total_lst = []
    for score in movie_dict.values():
        total_ratings_movie = total_ratings_movie + score
        total_lst.append(score)
        
    total_lst.sort()
    min_movie_rating = total_lst[0]
    max_movie_rating = total_lst[len(movie_dict)-1]
    median_movie_rating = statistics.median(total_lst)
        
    avg_rating_movie = total_ratings_movie / len(movie_dict)
    
    for score in movie_dict.values():
        total_std += pow((score - avg_rating_movie),2)
    
    std_avg_movie = sqrt(total_std / len(movie_dict))
    
    print("Average number of ratings per items: %f, and std dev of %f  " %(avg_rating_movie, std_avg_movie))
    print("Min, Max, Median number of ratings per item: %d, %d, %d" %(min_movie_rating, max_movie_rating, median_movie_rating))
    print()
  


    
    plt.hist(lst, bins= [1,2, 3,4,5])
    plt.title("histogram") 
    plt.show()

    
    
# Returns a distance-based similarity score for person1 and person2
def sim_distance(prefs,person1,person2, sim_weight = 1):
    '''
        Calculate Euclidean distance similarity 
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        
        Returns:
        -- Euclidean distance similarity for RS, as a float
        
    '''
    
    # Get the list of shared_items
    si={}
    for item in prefs[person1]: 
        if item in prefs[person2]: 
            si[item]=1
    
    # if they have no ratings in common, return 0
    if len(si)==0: 
        return 0
    
    # Add up the squares of all the differences
    ## Note: Calculate similarity between any two users across all items they
    ## have rated in common; i.e., includes the sum of the squares of all the
    ## differences
    
    sum_of_squares = 0
    
    for item in si:
        rate_1 = prefs[person1][item]
        rate_2 = prefs[person2][item]
        diff = (rate_1 - rate_2)**2
        sum_of_squares += diff
#     print(1/(1+sqrt(sum_of_squares)))
    similarity = 1/(1+sqrt(sum_of_squares))
    # Returns Euclidean distance similarity for RS
    if sim_weight > 1 and len(si) < sim_weight:
        # If we are apply weight then multiply n/sim_weight
#         print("weighting")
#         print(similarity*(float(len(si))/sim_weight))
        return (similarity*(float(len(si))/sim_weight))
    else:
        return similarity

# Returns the Pearson correlation coefficient for p1 and p2
def sim_pearson(prefs,p1,p2, sim_weight):
    '''
        Calculate Pearson Correlation similarity 
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        
        Returns:
        -- Pearson Correlation similarity as a float
        
    '''
    
    ## place your code here!
    ##
    ## REQUIREMENT! For this function, calculate the pearson correlation
    ## "longhand", i.e, calc both numerator and denominator as indicated in the
    ## formula. You can use sqrt (from math module), and average from numpy.
    ## Look at the sim_distance() function for ideas.
    ##
    sum_p1 = 0
    count = 0
    sum_p2 = 0
    numer = 0
    denumer_x = 0
    denumer_y = 0
    
    variance = {}
    
    
    for items in prefs[p1]:
        if items in prefs[p2]: 
            variance[items] = [prefs[p1][items], prefs[p2][items]]  
    
    for items in variance:
        sum_p1 += variance[items][0]
        count += 1
        sum_p2 += variance[items][1]
    if (count == 0):
        avg_p1 = 0
        avg_p2 = 0
    else:
        avg_p1 = sum_p1 / count
        avg_p2 = sum_p2 / count
        
    
    for items in variance.values():
        numer += (items[0]-avg_p1)* (items[1]-avg_p2)
        denumer_x += pow((items[0]-avg_p1),2)
        denumer_y += pow((items[1]-avg_p2),2)
        
    deno = sqrt(denumer_x)* sqrt(denumer_y)
    if deno == 0:
        return 0
   ##print(numer)
    #print(denumer_x)
    #print(denumer_y)
    
    if sim_weight > 1 and count < sim_weight:
#         print(sim_weight)
        return (numer/deno)*(count/sim_weight)
    else:
        return (numer/deno)
    
#Helper function for Pearson correlation
def avg(prefs, person, person2):
    '''
        Calculate the average rating for person based on their shared 
        movie rating of person2
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user 1
        -- person2: string containing name of user 2
        
        Returns:
        -- Average of person's rating
        
    '''
    count = 0
    sum_rating = 0
    movies = prefs[person]
    movies2 = prefs[person2]
    for movie in movies:
        if movie in movies2:
            sum_rating += prefs[person][movie]
            count += 1
    if count == 0:
        return 0
    return sum_rating/count

# Transpose the pref matrix
def transformPrefs(prefs):
    pref = {}
    for person in prefs:
        movies = prefs[person]
        for movie in movies:
            rating = prefs[person][movie]
            pref.setdefault(movie,{}) # make it a nested dicitonary
            pref[movie][person]=float(rating)
    return pref

# Returns a list of similar matches for person in tuples
def topMatches(prefs,person,similarity, n, sim_weight):
    '''
        Returns the best matches for person from the prefs dictionary
 
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        -- n: number of matches to find/return (5 is default)
       
        Returns:
        -- A list of similar matches with 0 or more tuples,
           each tuple contains (similarity, item name).
           List is sorted, high to low, by similarity.
           An empty list is returned when no matches have been calc'd.
       
    '''    
    scores=[(similarity(prefs,person,other, sim_weight),other)
                    for other in prefs if other!=person]
    scores.sort()
    scores.reverse()
    return scores[0:n]

################################User-Based####################################
# Create the user-to-user similarity matrix
def calculateSimilarUsers(prefs,n,similarity, sim_weight):

    '''
        Creates a dictionary of users showing which other users they are most 
        similar to. 
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A dictionary with a similarity matrix
        
    '''     
    result={}
    # Invert the preference matrix to be item-centric
    
    c=0
    for user in prefs:
      # Status updates for larger datasets
#         c+=1
#         if c%100==0: 
#             print ("%d / %d") % (c,len(prefs))
            
        # Find the most similar items to this one
        if int(sim_weight) > 1:
            scores=topMatches(prefs,user,similarity = similarity,n=n, sim_weight = sim_weight)
        else:
            scores=topMatches(prefs,user,similarity = similarity,n=n, sim_weight = sim_weight)
        result[user]=scores
    return result

# Create the list of recommendation for person
def getRecommendationsSim(prefs,person,sim_matrix, similarity=sim_pearson, sim_weight = 1, threshold = 0):
    '''
       Similar to getRecommendations() but uses the user-user similarity matrix 
       created by calculateSimUsers().
    '''

    totals={}
    simSums={}
   

    for other in prefs:
      # don't compare me to myself
        sim=0
        if other==person: 
            continue
        # sim=similarity(UUmatrix,person,other, sim_weight = sim_weight)
        for (sims, user) in sim_matrix[person]:
            if user == other:
                sim = sims
                
        
#         print(sim)
        # ignore scores of zero or lower
        if sim <=threshold: continue
        for item in prefs[other]:
            
            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
  
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]
  
    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings

# Calc User-based CF recommendations for all users
def get_all_UU_recs(prefs, sim, num_users=10, top_N=5, sim_weight = 1 ):
    ''' 
    Print user-based CF recommendations for all users in dataset
    Parameters
    -- prefs: nested dictionary containing a U-I matrix
    -- sim: similarity function to use (default = sim_pearson)
    -- num_users: max number of users to print (default = 10)
    -- top_N: max number of recommendations to print per user (default = 5)
    Returns: None
    '''
    # print(sim)
    for person in prefs:
        print ('User-based CF recs for %s: ' % (person), 
               getRecommendationsSim(prefs, person, similarity=sim, sim_weight=sim_weight)) 

# Compute Leave_One_Out evaluation
def loo_cv(prefs, sim, sim_weight, threshold):
    """
    Leave_One_Out Evaluation: evaluates recommender system ACCURACY
     
     Parameters:
         prefs dataset: critics, ml-100K, etc.
         metric: MSE, MAE, RMSE, etc.
         sim: distance, pearson, etc.
         algo: user-based recommender, item-based recommender, etc.
     
    Returns:
         error_total: MSE, MAE, RMSE totals for this set of conditions
         error_list: list of actual-predicted differences
    
    
    Algo Pseudocode ..
    Create a temp copy of prefs
    
    For each user in temp copy of prefs:
      for each item in each user's profile:
          delete this item
          get recommendation (aka prediction) list
          restore this item
          if there is a recommendation for this item in the list returned
              calc error, save into error list
          otherwise, continue
      
    return mean error, error list
    """

    start_time = time.time()
    temp_copy = cp.deepcopy(prefs)
    error_mse = 0
    error_list = []
    error_rmse=0
    error_list_rmse = []
    error_mae = 0
    error_list_mae = []
    count = 0 
    
    for person in prefs:
        movies = prefs[person]
        for movie in movies:
            temp = movie
            orig = temp_copy[person].pop(movie)
            rec = getRecommendationsSim(temp_copy, person, similarity=sim, sim_weight=sim_weight, threshold = threshold)

            found = False
            predict = 0
            for element in rec:
                count += 1
                err = (element[0] - orig) ** 2
                error_mse += err
                error_mae = (abs(element[0] - orig))
                error_rmse += sqrt(err)
                    
                error_list.append(err)
                error_list_rmse.append(err)
                error_list_mae.append(error_mae)
            
                found = True
                predict = element[0]
                    
                if found==True and count%10==0:
                    print("Number of users processed: ", count )
                    #print("--- %f seconds --- for %d users " % (time.time() - start_time), count)
                    print("===> {} secs for {} users, {} time per user: ".format(time.time() - start_time, count, (time.time() - start_time)/count))
                    print("MSE:", "%.10f" %(error_mse/count),  ", MAE:", "%.10f" % (error_mae/count),  ", RMSE:", "%.10f" % (error_rmse/count), ", Coverage:", "%.10f" % (len(error_list))/count)
                temp_copy[person][movie]= orig

    pickle.dump(error_list, open( "data_2/sq_error_User_dist_25_0.p", "wb" ))
                
    if count != 0:
        
        return error_mse/count, error_mae/count, error_rmse/count, len(error_list)  
    else:
        pass

################################Item-Based####################################

# Create the item-item similarity matrix
def calculateSimilarItems(prefs,n,similarity, sim_weight):
 
    '''
        Creates a dictionary of items showing which other items they are most
        similar to.
 
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)
       
        Returns:
        -- A dictionary with a similarity matrix
       
    '''    
    result={}
    # Invert the preference matrix to be item-centric
    itemPrefs=transformPrefs(prefs)
    c=0
    for item in itemPrefs:
      # Status updates for larger datasets
        #c+=1
        #if c%100==0:
            #print ("%d / %d") % (c,len(itemPrefs))
           
        # Find the most similar items to this one
        scores=topMatches(itemPrefs,item,similarity,n=n, sim_weight = sim_weight)
        result[item]=scores
    return result

# Create the list of recommendation for person
def getRecommendedItems(prefs,user, itemMatch, sim_weight, threshold) :
    '''
        Calculates recommendations for a given user 
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''    
    userRatings=prefs[user]
    scores={}
    totalSim={}
    # Loop over items rated by this user
    for (item, rating) in userRatings.items( ):
  
      # Loop over items similar to this one
        for (similarity, item2) in itemMatch[item]:
    
            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
            # ignore scores of zero or lower
            if similarity<=threshold: continue            
            # Weighted sum of rating times similarity
            scores.setdefault(item2,0)
            scores[item2]+=similarity*rating
            # Sum of all the similarities
            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity
  
    # Divide each total score by total weighting to get an average

    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]    
  
    # Return the rankings from highest to lowest
    rankings.sort( )
    rankings.reverse( )
    return rankings

# Calc Item-based CF recommendations for all users
def get_all_II_recs(prefs, itemsim, sim_method, num_users=10, top_N=5):

# Compute Leave_One_Out evaluation
    ''' 
    Print item-based CF recommendations for all users in dataset
    Parameters
    -- prefs: U-I matrix (nested dictionary)
    -- itemsim: item-item similarity matrix (nested dictionary)
    -- sim_method: name of similarity method used to calc sim matrix (string)
    -- num_users: max number of users to print (integer, default = 10)
    -- top_N: max number of recommendations to print per user (integer, default = 5)
    Returns: None
    
    '''
    for person in prefs:
        print ('Item-based CF recs for %s, %s: ' % (person, sim_method), 
                getRecommendedItems(prefs, itemsim, person)) 
        
def loo_cv_sim(prefs, sim, sim_matrix, threshold, sim_weight, algo):
        """
        Leave-One_Out Evaluation: evaluates recommender system ACCURACY
         
         Parameters:
             prefs dataset: critics, etc.
         metric: MSE, or MAE, or RMSE
         sim: distance, pearson, etc.
         algo: user-based recommender, item-based recommender, etc.
             sim_matrix: pre-computed similarity matrix
         
        Returns:
             error_total: MSE, or MAE, or RMSE totals for this set of conditions
         error_list: list of actual-predicted differences
        """
        
        start_time = time.time()
        prefs_copy = cp.deepcopy(prefs)
        error_list = []
        error_total = 0
        count = 0
        error_mse = 0
        error_rmse=0
        error_list_rmse = []
        error_mae = 0
        error_list_mae = []
        c=0
        
    
 
    
        for person, item in prefs.items():
            c = c+1
            for movie in item:
                delete = prefs_copy[person].pop(movie)
                rec = algo(prefs_copy, person, sim_matrix, sim_weight= sim_weight, threshold = threshold)
    
             
                   
                prefs_copy[person][movie] = delete
                in_value = False
                    
               
               
                for item in rec:
                    in_value = False
                    for movie in prefs[person]:
                        if movie == item[1]:
                            in_value = True
                            prediction = item[0]
                            real_val = prefs[person][movie]
                            err = pow((prediction - real_val),2)
                            error_list.append (err)
                            error_total += err
                            count += 1
                            error_mse += err
                            error_mae += abs(prediction - real_val)
                            error_rmse += err
                            error_list_rmse.append(err)
                            error_list_mae.append(error_mae)
            if((c+1) % 10 == 0):
                
                print("Number of users processed: ", (c+1) )
                if count == 0:
                    time_per_user=(time.time() - start_time)/(c+1)
                    print("===> {} secs for {} users, {} time per user: ".format(round(time.time() - start_time,2), c+1, round(time_per_user,3)))
                    print("MSE:", "%.10f" %(error_mse),  ", MAE:", "%.10f" % (error_mae),  ", RMSE:", "%.10f" % (sqrt(error_rmse)))
                else:
                    print("===> {} secs for {} users, {} time per user: ".format(round(time.time() - start_time,2), c+1, round((time.time() - start_time)/count),3))
                    print("MSE:", "%.10f" %(error_mse/count),  ", MAE:", "%.10f" % (error_mae/count),  ", RMSE:", "%.10f" % (sqrt(error_rmse/count)))
                 
        pickle.dump(error_list, open( "data_2/sq_error_Item_dist_50_0.p", "wb" ))                                                   
                            
        if count == 0:
            print("MSE:", "%.10f" %(error_mse),  ", MAE:", "%.10f" % (error_mae),  ", RMSE:", "%.10f" % (sqrt(error_rmse)), ", Coverage:", "%.10f" % (len(error_list)))
            return error_mse, error_mae, error_rmse, len(error_list)
        else:
            print("MSE:", "%.10f" %(error_mse/count),  ", MAE:", "%.10f" % (error_mae/count),  ", RMSE:", "%.10f" % (sqrt(error_rmse/count)), ", Coverage:", "%.10f" % (len(error_list)))
            return error_mse/count, error_mae/count, error_rmse/count, len(error_list)#Main


#This is for t_test
def print_loocv_results(error_list):
    ''' Print LOOCV SIM results '''
           
    #print()
    #print(error_list)
    error = sum(tuple(error_list))/len(error_list)          
    print ('MSE =', error)
    
    return(error, error_list)
                                
        
#Main
def main():
    ''' User interface for Python console '''
    
    # Load critics dict from file
    path = os.getcwd() # this gets the current working directory
                       # you can customize path for your own computer here
    print('\npath: %s' % path) # debug
    done = False
    prefs = {}
    pref = {}
    item_distance_matrix = {}
    item_pearson_matrix = {}
    user_distance_matrix = {}
    user_pearson_matrix = {}
    movie_similarity_matrix = {}
    item_similarity_matrix = {}
    ratings = []
    sim_method = ""
    done = False
    Weight_parameter = 1

    while not done: 
        print()
        # Start a simple dialog
        file_io = input('R(ead) critics data from file?, \n'
                        'RML(ead ml-100k dataset, \n'
                        'S(tats) print?, \n'
                        'PD-R(ead) critics data from file for MF?, \n'
                        'PD-RML100(ead) ml100K data from file for MF?, \n'
                        'T(est/train datasets for MF)?, \n'
                        'MF-ALS(atrix factorization- Alternating Least Squares)?, \n'
                        'MF-SGD(atrix factorization- Stochastic Gradient Descent)?, \n'
                        'Sim(ilarity matrix) - Item Setup?, \n'
                        'Simu(ilarity Matrix) - User Setup?, \n'
                        'FE(ature Encoding) Setup?, \n'
                        'TFIDF(and cosine sim) Setup?, \n'
                        'H(ybrid Matrices) Setup, \n'
                        'LCVSIM(eave one out cross-validation)?, \n'
                        'RECS(ecommendations -- all algos)?, \n'
                        'Test(of Hypothesis)?, \n'
                        '==>> ')
        
        if file_io == 'R' or file_io == 'r':
            print()
            file_dir = 'data/'
            datafile = 'critics_ratings.data'
            itemfile = 'critics_movies.item'
            genrefile = 'critics_movies.genre' # movie genre file
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            movies, genres, features = from_file_to_2D(path, file_dir+genrefile, file_dir+itemfile)
            print('Number of users: %d\nList of users:' % len(prefs), 
                  list(prefs.keys()))
            
            fe_ran = False
            tfidf_ran = False
            user_pearson_ran = False
            user_distance_ran = False
            item_pearson_ran = False
            item_distance_ran = False
            hybrid_ran = False
            
        
        elif file_io == 'RML' or file_io == 'rml':
           print()
           file_dir = '/data/ml-100k/' # path from current directory
           datafile = 'u.data'  # ratings file
           itemfile = 'u.item'  # movie titles file  
           genrefile = 'u.genre' # movie genre file
           print ('Reading "%s" dictionary from file' % datafile)
           prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
           movies, genres, features = from_file_to_2D(path, file_dir+genrefile, file_dir+itemfile)
           print('Number of users: %d\nList of users [0:10]:' 
                 % len(prefs), list(prefs.keys())[0:10] )  
           
           fe_ran = False
           tfidf_ran = False
           user_pearson_ran = False
           user_distance_ran = False
           item_pearson_ran = False
           item_distance_ran = False
           hybrid_ran = False
         
          
           
        elif file_io == 'PD-R' or file_io == 'pd-r':
            
            # Load user-item matrix from file
            
            ## Read in data: critics
            file_dir = '/data/' # for critics
          
            datafile = 'critics_ratings.data'
            itemfile = 'critics_movies.item'
            genrefile = 'critics_movies.genre' # movie genre file
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            user_list =  list(prefs.keys())
            #print('\npath: %s\n' % path_name + data_folder) # debug: print path info
            names = ['user_id', 'item_id', 'rating', 'timestamp'] # column headings
            
            
            #Create pandas dataframe
            df = pd.read_csv(path + file_dir + 'critics_ratings_userIDs.data', sep='\t', names=names) # for critics
            ratings = file_info(df)
            
            # set test/train in case they were set by a previous file I/O command
            test_train_done = False
            movies, genres, features = from_file_to_2D(path, file_dir+genrefile, file_dir+itemfile)
            movie_ID = movie_to_ID(movies)
            movie_ID_array = []
            for element in movie_ID.keys():
                movie_ID_array.append(element)
            
            
            SGD_ran = False
            ALS_ran = False



            print()
            print('Test and Train arrays are empty!')
            print()
    
        elif file_io == 'PD-RML100' or file_io == 'pd-rml100':
            
            # Load user-item matrix from file
            ## Read in data: ml-100k
            file_dir = '/data/ml-100k/' # for ml-100k
            datafile = 'u.data'
            itemfile = 'u.item'
            genrefile = 'u.genre' # movie genre file
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            user_list =  list(prefs.keys())                   
            #print('\npath: %s\n' % path_name + data_folder) # debug: print path info
            names = ['user_id', 'item_id', 'rating', 'timestamp'] # column headings
    
            #Create pandas dataframe
            df = pd.read_csv(path + file_dir + 'u.data', sep='\t', names=names) # for ml-100k
            ratings = file_info(df)
            
            test_train_done = False
            print()
            print('Test and Train arrays are empty!')
            print()
            
            movies, genres, features = from_file_to_2D(path, file_dir+genrefile, file_dir+itemfile)
            movie_ID = movie_to_ID(movies)
            movie_ID_array = []
            for element in movie_ID.keys():
                movie_ID_array.append(element)
            
            SGD_ran = False
            ALS_ran = False
    
       
            
        elif file_io == 'T' or file_io == 't':
            if len(ratings) > 0:
                answer = input('Generate both test and train data? Y or y, N or n: ')
                if answer == 'N' or answer == 'n':
                    TRAIN_ONLY = True
                else:
                    TRAIN_ONLY = False
                
                #print('TRAIN_ONLY  in EVAL =', TRAIN_ONLY) ## debug
                train, test = train_test_split(ratings, TRAIN_ONLY) ## this should 
                ##     be only place where TRAIN_ONLY is needed!! 
                ##     Check for len(test)==0 elsewhere
                
                test_train_info(test, train) ## print test/train info
        
                ## How is MSE calculated for train?? self.ratings is the train
                ##    data when ExplicitMF is instantiated for both als and sgd.
                ##    So, MSE calc is between predictions based on train data 
                ##    against the actuals for train data
                ## How is MSE calculated for test?? It's comparing the predictions
                ##    based on train data against the actuals for test data
                
                test_train_done = True
                print()
                print('Test and Train arrays are ready!')
                print()
            else:
                print ('Empty U-I matrix, read in some data!')
                print()            
    
        elif file_io == 'MF-ALS' or file_io == 'mf-als':
            
            if len(ratings) > 0:
                if test_train_done:
                    
                    ## als processing
                    
                    print()
                    ## sample instantiations ..
                    if len(ratings) < 10: ## for critics
                        print('Sample for critics .. ')
                        iter_array = [1, 2, 5, 10, 20]
                        MF_ALS = ExplicitMF(train, learning='als', n_factors=2, user_fact_reg=1, item_fact_reg=1, max_iters=max(iter_array), verbose=True)
                        print('[2,1,20]')
                    
                    elif len(ratings) < 1000: ## for ml-100k
                        print('Sample for ml-100k .. ')
                        iter_array = [1, 2, 5 , 10, 20, 50] #, 100] #, 200]
                        MF_ALS = ExplicitMF(train, learning='als', n_factors=20, user_fact_reg=.01, item_fact_reg=.01, max_iters=max(iter_array), verbose=True) 
                        print('[20,0.01,50]')
                    
                  
                        
                    parms = input('Y or y to use these parameters or Enter to modify: ')# [2,0.01,10,False]
                    if parms == 'Y' or parms == 'y':
                        pass
                    else:
                        parms = eval(input('Enter new parameters as a list: [n_factors, reg, iters]: '))
                        
                        # instantiate with this set of parms
                        MF_ALS = ExplicitMF(train,learning='als', 
                                            n_factors=parms[0], 
                                            user_fact_reg=parms[1], 
                                            item_fact_reg=parms[1])
                       
                        # set up the iter_array for this run to pass on
                        orig_iter_array = [1, 2, 5, 10, 20, 50, 100, 200]
                        i_max = parms[2]
                        index = orig_iter_array.index(i_max)
                        iter_array = []
                        for i in range(0, index+1):
                            iter_array.append(orig_iter_array[i])
                            
                    # run the algo and plot results
                    MF_ALS.calculate_learning_curve(iter_array, test) 
                    plot_learning_curve(iter_array, MF_ALS )
                    predictions_ALS = MF_ALS.predict_all() 
                    ALS_ran = True
                    
                    

                    
                else:
                    print ('Empty test/train arrays, run the T command!')
                    print()                    
            else:
                print ('Empty U-I matrix, read in some data!')
                print()
                    
        elif file_io == 'MF-SGD' or file_io == 'mf-sgd':
            
            if len(ratings) > 0:
                
                if test_train_done:
                
                    ## sgd processing
                     
                    ## sample instantiations ..
                    if len(ratings) < 10: ## for critics
                        # Use these parameters for small matrices
                        print('Sample for critics .. ')
                        iter_array = [1, 2, 5, 10, 20]                     
                        MF_SGD = ExplicitMF(train, 
                                            n_factors=2, 
                                            learning='sgd', 
                                            sgd_alpha=0.075,
                                            sgd_beta=0.01, 
                                            max_iters=max(iter_array), 
                                            sgd_random=False)
                        print('[2, 0.075, 0.01, 20]')
                        print()
                        

                    elif len(ratings) < 1000:
                       # Use these parameters for ml-100k
                        print('Sample for ml-100k .. ')
                        iter_array = [1, 2, 5, 10, 20]                     
                        MF_SGD = ExplicitMF(train, 
                                            n_factors=2, 
                                            learning='sgd', 
                                            sgd_alpha=0.02,
                                            sgd_beta=0.2, 
                                            max_iters=max(iter_array), 
                                            sgd_random=False, verbose=True)
                         
                     
                    parms = input('Y or y to use these parameters or Enter to modify: ')# [2,0.01,10,False]
                    if parms == 'Y' or parms == 'y':
                        pass
                    else:
                        parms = eval(input('Enter new parameters as a list: [n_factors K, learning_rate alpha, reg beta, max_iters: ')) #', random]: '))
                        MF_SGD = ExplicitMF(train, n_factors=parms[0], 
                                            learning='sgd', 
                                            sgd_alpha=parms[1], 
                                            sgd_beta=parms[2], 
                                            max_iters=parms[3], 
                                            sgd_random=False, verbose=True)  

                        orig_iter_array = [1, 2, 5, 10, 20, 50, 100, 200]
                        i_max = parms[3]
                        index = orig_iter_array.index(i_max)
                        iter_array = []
                        for i in range(0, index+1):
                            iter_array.append(orig_iter_array[i])
                         
                    MF_SGD.calculate_learning_curve(iter_array, test) # start the training
                    predictions_SGD = MF_SGD.predict_all() 
                    
    

                    SGD_ran = True
                
                   
                    plot_learning_curve(iter_array, MF_SGD)    
                     
                else:
                    print ('Empty test/train arrays, run the T command!')
                    print()   
                
            ##
            ## Place for new Grid_search command ..
            ## for sgd: values of K, alpha learning rate, beta regularization, max iters
            ## for als: values of num_factors, user and item regularization, max iters

            else:
                print ('Empty U-I matrix, read in some data!')
                print()   
                
                
        #STATS command
        elif file_io == 'S' or file_io == 's':
            print()
            filename = 'critics_ratings.data'
            if len(prefs) > 0:
                data_stats(prefs, filename)
                popular_items(prefs, filename)
            else: # Make sure there is data  to process ..
                print ('Empty dictionary, R(ead) in some data!')
      

      
        
        
        elif file_io == 'LCVSIM' or file_io == 'lcvsim':
             ready = False # sub command in progress
             #threshold = float(input('threshold(enter a digit)?\n'))
             #print()
             if len(prefs) > 0:             
                print('LOO_CV_SIM Evaluation')
                algo = input('Enter U(ser), I(tem), FE, TFIDF, or (H)ybrid \n')
                
                #item-based
                if algo == 'I' or algo == 'i':
                    method = input('Enter Pearson or Distance: ')
                    threshold = float(input('threshold(enter a digit, 0<= digit <=1)?\n'))
                    sim_weight = int(input('similarity weight(enter a digit: 1, 25, or, 50)?\n'))
                    algo = getRecommendedItems
                    if method == "Pearson":
                        sim = sim_pearson
                        error_mse, error_rmse, error_mae, Coverage = loo_cv_sim(prefs, sim, item_pearson_matrix, threshold, sim_weight, algo)
                    else:
                        sim = sim_distance
                        error_mse, error_rmse, error_mae, Coverage = loo_cv_sim(prefs, sim, item_distance_matrix, threshold, sim_weight, algo)

                #user-based
                elif algo == 'U' or algo == 'u':
                    method = input('Enter Pearson or Distance: ')
                    threshold = float(input('threshold(enter a digit, 0 <= digit <=1)?\n'))
                    sim_weight = int(input('similarity weight(enter a digit: 1,25, or 50)?\n'))
                    algo = getRecommendationsSim
                    if method == "Pearson":
                        sim = sim_pearson
                        error_mse, error_rmse, error_mae, Coverage = loo_cv_sim(prefs, sim, user_pearson_matrix, threshold, sim_weight, algo)
                    else:
                        sim = sim_distance
                        error_mse, error_rmse, error_mae, Coverage = loo_cv_sim(prefs, sim, user_distance_matrix, threshold, sim_weight, algo)                
                    
                elif algo == 'TFIDF' or algo == 'tfidf':
                    loo_cv_sim_TFIDF(prefs, movie_similarity_matrix)
                    
                elif algo == 'FE' or algo == 'fe':
                    loo_cv_sim_FE(prefs, features, movie_ID)
                
                elif algo == 'H' or algo == 'h':
                  
                    
                    loo_cv_sim_hybrid(prefs, movie_similarity_matrix, item_similarity_matrix, Weight_parameter)
    

                else:
                    print('invalid input')
               
                
                
             else:
                 print ('Empty dictionary, run R(ead) OR Empty Sim Matrix, run Sim!')
        
        
        #Setting up User-based matrices 
        elif file_io == 'Simu' or file_io == 'simu':
            print()
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson? ')
                sim_weight = int(input('similarity weight(enter a digit: 1,25, or 50)?\n'))
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        user_distance_matrix = pickle.load(open( "save_itemsim_distance.p", "rb" ))
                        sim_method = 'sim_distance'
                        user_distance_ran = True
    
                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        user_pearson_matrix = pickle.load(open( "save_itemsim_pearson.p", "rb" ))  
                        sim_method = 'sim_pearson'
                        user_pearson_ran = True
                        
                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        user_distance_matrix = calculateSimilarUsers(prefs,100, similarity=sim_distance, sim_weight = sim_weight)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(user_distance_matrix, open( "save_itemsim_distance.p", "wb" ))
                        sim_method = 'sim_distance'
                        user_distance_ran = True

                        
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        user_pearson_matrix = calculateSimilarUsers(prefs,100, similarity=sim_pearson, sim_weight = sim_weight)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(user_pearson_matrix, open( "save_itemsim_distance.p", "wb" ))
                        sim_method = 'sim_pearson'
                        user_pearson_ran = True
                    
                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue
                    
                    ready = True # sub command completed successfully
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter Sim(ilarity matrix) again and choose a Write command')
                    print()
                
                

                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!') 

        #Setting up Item-based Matrices
        elif file_io == 'Sim' or file_io == 'sim':
            print()
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson? ')
                sim_weight = int(input('similarity weight(enter a digit: 1, 25, or 50)?\n'))
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        item_distance_matrix = pickle.load(open( "item_distance_critics.p", "rb" ))
                        sim_method = 'sim_distance'
                        item_distance_ran = True
    
                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        item_pearson_matrix = pickle.load(open( "item_pearson_critics.p", "rb" ))  
                        sim_method = 'sim_pearson'
                        item_pearson_ran = True

                        
                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        item_distance_matrix  = calculateSimilarItems(prefs,100, similarity=sim_distance, sim_weight = sim_weight)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(item_distance_matrix, open( "item_distance_critics.p", "wb" ))
                        sim_method = 'sim_distance'
                        item_distance_ran = True

                 
                        
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        item_pearson_matrix  = calculateSimilarItems(prefs, 100, similarity=sim_pearson, sim_weight = sim_weight)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(item_pearson_matrix, open( "item_pearson_critics.p", "wb" ))
                        sim_method = 'sim_pearson'
                        item_pearson_ran = True
                      
                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue
                    
                    ready = True # sub command completed successfully
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter Sim(ilarity matrix) again and choose a Write command')
                    print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!') 
        
        #Setting up FE matrix for recommendations
        elif file_io == 'FE' or file_io == 'fe' or file_io == 'Fe':
            print()
            #movie_title_to_id = movie_to_ID(movies)
            # determine the U-I matrix to use ..
            if len(prefs) > 0 and len(prefs) <= 10: # critics
                # convert prefs dictionary into 2D list
                R = to_array(prefs)
                
                '''
                # e.g., critics data (CES)
                R = np.array([
                [2.5, 3.5, 3.0, 3.5, 2.5, 3.0],
                [3.0, 3.5, 1.5, 5.0, 3.5, 3.0],
                [2.5, 3.0, 0.0, 3.5, 0.0, 4.0],
                [0.0, 3.5, 3.0, 4.0, 2.5, 4.5],
                [3.0, 4.0, 2.0, 3.0, 2.0, 3.0],
                [3.0, 4.0, 0.0, 5.0, 3.5, 3.0],
                [0.0, 4.5, 0.0, 4.0, 1.0, 0.0],
                ])            
                '''      
                print('critics')
                print(R)
                print()
                print('features')
                print(features)
                
                movie_ID = movie_to_ID(movies)

                
                fe_ran = True

            elif len(prefs) > 10:
                print('ml-100k')   
                R = to_array(prefs)
                movie_ID = movie_to_ID(movies)

                fe_ran = True
                
            else:
                print ('Empty dictionary, read in some data!')
                print()
                
            
                
        #Setting up TFIDF matrix for TFIDF
        elif file_io == 'TFIDF' or file_io == 'tfidf':
            print()
            # determine the U-I matrix to use ..
            if len(prefs) > 0 and len(prefs) <= 10:
                R = to_array(prefs)
                feature_str = to_string(features)                 
                feature_docs = to_docs(feature_str, genres)
                

                print('critics')
                print(R)
                print()
                print('features')
                print(features)
                print()
                print('feature docs')
                print(feature_docs) 
                cosim_matrix = cosine_sim(feature_docs)
                print()
                print('cosine sim matrix')
                
                
                tfidf_ran = True
                 
                '''

                print(cosim_mat                <class 'numpy.ndarray'> 
                
                [[1.         0.         0.35053494 0.         0.         0.61834884]
                [0.         1.         0.19989455 0.17522576 0.25156892 0.        ]
                [0.35053494 0.19989455 1.         0.         0.79459157 0.        ]
                [0.         0.17522576 0.         1.         0.         0.        ]
                [0.         0.25156892 0.79459157 0.         1.         0.        ]
                [0.61834884 0.         0.         0.         0.         1.        ]]
                '''
                
                '''
                #print and plot histogram of similarites
                upper_triangle = np.triu(cosim_matrix, 1)
                fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
                
                

                n, m = cosim_matrix.shape

                m = np.triu_indices(n=n, k= 1, m=m)
                
                
                # array([0.35, 0.42, 0.31, 0.25, 0.38, 0.41, 0.21, 0.36, 0.46, 0.31])

                #print(cosim_matrix[m].mean())
                new_cosim_matrix = [] #remove all 0
                for element in cosim_matrix[m]:
                    if element != 0:
                        new_cosim_matrix.append(element)
                # 0.346


                # plt.hist(upper_triangle, bins=[0, 0.2, 0.4, 0.6, 0.8, 1])
                
                # plt.show()
                # plt.hist(upper_triangle, bins=[0.1, 0.4, 0.6, 0.8, 1])
                # plt.show()

                

            # We can set the number of bins with the *bins* keyword argument.
                axs[0].hist(cosim_matrix[m], bins=[0, 0.2, 0.4, 0.6, 0.8, 1])
                axs[1].hist(cosim_matrix[m], bins=[0.1, 0.4, 0.6, 0.8, 1])
                plt.show()
                
 
                print("mean of arr : ", np.mean(new_cosim_matrix))
                print("std of arr : ", np.std(new_cosim_matrix))
                '''
                
                movie_similarity_matrix = {}
                
                for i in range(len(movies)):
                    for j  in range(len(movies)):
                         if i == j: continue
                         id_dict_1 = str(i+1)
                         id_dict_2 = str(j+1)
                         movie_similarity_matrix.setdefault(movies[id_dict_1],{}) # make it a nested dicitonary
                         movie_similarity_matrix[movies[id_dict_1]][movies[id_dict_2]]= cosim_matrix[i][j]
                  
            elif len(prefs) > 10:
                print('ml-100k')   
                
                
                R = to_array(prefs)
                feature_str = to_string(features)                 
                feature_docs = to_docs(feature_str, genres)
                
                print(R[:3][:5])
                print()
                print('features')
                print(features[0:5])
                print()
                print('feature docs')
                print(feature_docs[0:5]) 
                cosim_matrix = cosine_sim(feature_docs)
                print()
                print('cosine sim matrix')
                print (type(cosim_matrix), len(cosim_matrix))
                print()
                
                tfidf_ran = True
                
                 
                '''
                <class 'numpy.ndarray'> 1682
                
                [[1.         0.         0.         ... 0.         0.34941857 0.        ]
                 [0.         1.         0.53676706 ... 0.         0.         0.        ]
                 [0.         0.53676706 1.         ... 0.         0.         0.        ]
                 [0.18860189 0.38145435 0.         ... 0.24094937 0.5397592  0.45125862]
                 [0.         0.30700538 0.57195272 ... 0.19392295 0.         0.36318585]
                 [0.         0.         0.         ... 0.53394963 0.         1.        ]]
                '''
                
                #print and plot histogram of similarites)
                upper_triangle = np.triu(cosim_matrix, 1)
                fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

                # plt.hist(upper_triangle, bins=[0, 0.2, 0.4, 0.6, 0.8, 1])
                
                # plt.show()
                # plt.hist(upper_triangle, bins=[0.1, 0.4, 0.6, 0.8, 1])
                # plt.show()

            # We can set the number of bins with the *bins* keyword argument.
                n, m = cosim_matrix.shape

                m = np.triu_indices(n=n, k= 1, m=m)
                
                
                # array([0.35, 0.42, 0.31, 0.25, 0.38, 0.41, 0.21, 0.36, 0.46, 0.31])

                new_cosim_matrix = [] #remove all 0
                for element in cosim_matrix[m]:
                    if element != 0:
                        new_cosim_matrix.append(element)
                # 0.346
                
                
                # plt.hist(upper_triangle, bins=[0, 0.2, 0.4, 0.6, 0.8, 1])
                
                # plt.show()
                # plt.hist(upper_triangle, bins=[0.1, 0.4, 0.6, 0.8, 1])
                # plt.show()

                

            # We can set the number of bins with the *bins* keyword argument.
                axs[0].hist(cosim_matrix[m], bins=[0, 0.2, 0.4, 0.6, 0.8, 1])
                axs[1].hist(cosim_matrix[m], bins=[0., 0.4, 0.6, 0.8, 1])
                plt.show()
                
 
                print("mean of arr : ", np.mean(new_cosim_matrix))
                print("std of arr : ", np.std(new_cosim_matrix))
                
                
                
                movie_similarity_matrix = {}
                
                for i in range(len(movies)):
                    for j  in range(len(movies)):
                         if i == j: continue
                         id_dict_1 = str(i+1)
                         id_dict_2 = str(j+1)
                         movie_similarity_matrix.setdefault(movies[id_dict_1],{}) # make it a nested dicitonary
                         movie_similarity_matrix[movies[id_dict_1]][movies[id_dict_2]]= cosim_matrix[i][j]
                            
            else:
                print ('Empty dictionary, read in some data!')
                print()
                
        #Setting up hybrid matrix for hybrid
        elif file_io == 'H' or file_io == 'h':
            print()
            if len(prefs) > 0 and len(prefs) <= 10:
                method = input('Enter Pearson or Distance: ')
                file_io = input('Enter significance weighting CF (1,25, or 50): ')
                sim_weight = float(file_io)
                weight = input('Enter weight parameter for hybrid (0 <= sig <= 1): ')
              
                
                R = to_array(prefs)
                feature_str = to_string(features)                 
                feature_docs = to_docs(feature_str, genres)
                cosim_matrix = cosine_sim(feature_docs)
                item_similarity_matrix = {}
                
                
                hybrid_ran = True
                
                
                Weight_parameter = float(weight)
                if (method == 'Pearson'):
                    item_sim_matrix = calculateSimilarItems(prefs,100,sim_pearson, sim_weight)
                else:
                    item_sim_matrix = calculateSimilarItems(prefs,100,sim_distance, sim_weight)
                
                
                for i in range(len(movies)):
                    for j in range(len(movies)):
                        if i == j: continue
                        id_dict_1 = str(i+1)
                        id_dict_2 = str(j+1)
                        item_similarity_matrix.setdefault(movies[id_dict_1],{})
                        for films in item_sim_matrix[movies[id_dict_1]]:
                            if (films[1] == movies[id_dict_2]):
                                item_similarity_matrix[movies[id_dict_1]][movies[id_dict_2]]= films[0]
                
               
   
                movie_similarity_matrix = {}
                
                for i in range(len(movies)):
                    for j  in range(len(movies)):
                         if i == j: continue
                         id_dict_1 = str(i+1)
                         id_dict_2 = str(j+1)
                         movie_similarity_matrix.setdefault(movies[id_dict_1],{}) # make it a nested dicitonary
                         movie_similarity_matrix[movies[id_dict_1]][movies[id_dict_2]]= cosim_matrix[i][j]
                         
                  
            elif len(prefs) > 10:
                print('ml-100k')   
                
                method = input('Enter Pearson or Distance: ')
                file_io = input('Enter signficance weighting for CF (1,25, or 50): ')
                sim_weight = float(file_io)
                weight = input('Enter weight parameter for hybrid (0 <= sig <= 1): ')
                R = to_array(prefs)
                feature_str = to_string(features)                 
                feature_docs = to_docs(feature_str, genres)
                cosim_matrix = cosine_sim(feature_docs)
                item_similarity_matrix = {}
                
                
                hybrid_ran = True
                
                
                Weight_parameter = float(weight)
                if (method == 'Pearson'):
                    item_sim_matrix = calculateSimilarItems(prefs,100, sim_pearson, sim_weight)
                
                else:
                    item_sim_matrix = calculateSimilarItems(prefs,100, sim_distance, sim_weight)
                
                
                for i in range(len(movies)):
                    for j in range(len(movies)):
                        if i == j: continue
                        id_dict_1 = str(i+1)
                        id_dict_2 = str(j+1)
                        item_similarity_matrix.setdefault(movies[id_dict_1],{})
                        for films in item_sim_matrix[movies[id_dict_1]]:
                            if (films[1] == movies[id_dict_2]):
                                item_similarity_matrix[movies[id_dict_1]][movies[id_dict_2]]= films[0]
                
               
   
                movie_similarity_matrix = {}
                
                for i in range(len(movies)):
                    for j  in range(len(movies)):
                         if i == j: continue
                         id_dict_1 = str(i+1)
                         id_dict_2 = str(j+1)
                         movie_similarity_matrix.setdefault(movies[id_dict_1],{}) # make it a nested dicitonary
                         movie_similarity_matrix[movies[id_dict_1]][movies[id_dict_2]]= cosim_matrix[i][j]
                         
                            
            else:
                print ('Empty dictionary, read in some data!')
                print()
                
        #RECS for critics
        elif file_io == 'RECS' or file_io == 'recs':
            print()
            
            if len(prefs) > 0 and len(prefs) <= 10: # critics
                print('critics') 
                algo = input('Enter Algorithm ( (U)U-CF, (I)I-CF, (S)GD, (A)LS, FE, TFIDF, H(ybrid) ): ')
                if algo == 'FE' or algo == 'fe':
                    if fe_ran:
                        userID = input('Enter username (for critics) or return to quit: ')
                        num_rec = int(input('Enter the number of recommendations (has to be an interger > 0): '))
                        if userID !='':
                            # Go run the FE algo
                            rankings = get_FE_recommendations(prefs, features, movie_ID, userID)
                            if len(rankings) == 0 or num_rec > len(rankings):
                                print(rankings)
                            else:
                                print('Recommendations:')
                                print()
                                print(rankings[0])
                                print()
                                print("Top Movies:")
                                print()
                                print(top_N(rankings, num_rec))
                    else:
                        print('Run the FE command first to set up FE data')
                
                elif algo == 'TFIDF' or algo == 'tfidf':
                    if tfidf_ran:        
                        userID = input('Enter username (for critics) or return to quit: ')
                        TFIDF_threshold = float(input('Enter the value of threshold (0 <= threshold <= 1): '))
                        num_rec = int(input('Enter the number of recommendations (has to be an interger > 0): '))
                        
                        if userID !='':
                           rankings = get_TFIDF_recommendations(prefs,movie_similarity_matrix, userID, TFIDF_threshold)
                           if len(rankings) == 0 or num_rec > len(rankings):
                               print(rankings)
                           else:
                               print('Recommendations:')
                               print()
                               print(rankings[0])
                               print()
                               print("Top Movies:")
                               print()
                               print(top_N(rankings, num_rec))
                    else:
                        print('Run the TFIDF command first to set up TFIDF data')                    
               
            
                elif algo == 'U' or algo == 'u':
                 
                    if user_distance_ran:
                        userID = input('Enter username (for critics) or return to quit: ')
                        threshold = int(input('threshold(enter a digit, 0 <= digit <= 1)?\n'))
                        num_rec = int(input('Enter the number of recommendations (has to be an interger > 0): '))
                        if userID !='':
                            rankings = getRecommendationsSim(prefs, userID, user_distance_matrix, sim_distance, sim_weight, threshold = 0)
                            if len(rankings) == 0 or num_rec > len(rankings):
                               print(rankings)
                            else:
                               print('Recommendations:')
                               print()
                               print(rankings[0])
                               print()
                               print("Top Movies:")
                               print()
                               print(top_N(rankings, num_rec))
                            
                    elif user_pearson_ran:
                        userID = input('Enter username (for critics) or return to quit: ')
                        threshold = int(input('threshold(enter a digit, 0<= digit <=1)?\n'))
                        num_rec = int(input('Enter the number of recommendations (has to be an interger > 0): '))
                        if userID !='':
                            rankings = getRecommendationsSim(prefs, userID, user_pearson_matrix, sim_pearson, sim_weight, threshold)
                            if len(rankings) == 0 or num_rec > len(rankings):
                               print(rankings)
                            else:
                               print('Recommendations:')
                               print()
                               print(rankings[0])
                               print()
                               print("Top Movies:")
                               print()
                               print(top_N(rankings, num_rec))
                    else:
                        print('Run the Simu command first to set up Simu data') 
                
                
                elif algo == 'I' or algo == 'i':
                     
                    if item_distance_ran:
                        userID = input('Enter username (for critics) or return to quit: ')
                        threshold = int(input('threshold(enter a digit, 0 <= digit <= 1)?\n'))
                        num_rec = int(input('Enter the number of recommendations (has to be an interger > 0: '))
                        if userID !='':
                            rankings = getRecommendedItems(prefs,userID, item_distance_matrix, sim_weight, threshold)
                            if len(rankings == 0) or num_rec > len(rankings):
                               print(rankings)
                            else:
                               print('Recommendations:')
                               print()
                               print(rankings[0])
                               print()
                               print("Top Movies:")
                               print()
                               print(top_N(rankings, num_rec))
                             
                    elif item_pearson_ran:
                        userID = input('Enter username (for critics) or return to quit: ')
                        threshold = int(input('threshold(enter a digit, 0 <= digit <= 1)?\n'))
                        num_rec = int(input('Enter the number of recommendations (has to be an interger > 0): '))
                        if userID !='':
                            rankings = getRecommendedItems(prefs,userID, item_pearson_matrix, sim_weight, threshold)
                            if len(rankings) == 0 or num_rec > len(rankings):
                               print(rankings)
                            else:
                               print('Recommendations:')
                               print()
                               print(rankings[0])
                               print()
                               print("Top Movies:")
                               print()
                               print(top_N(rankings, num_rec))
                             
                    else:
                        print('Run the Sim command first to set up Sim data')
                        
                elif algo == 'h' or algo == 'H':
                    
                   if hybrid_ran:
                        user = input('Enter username (for critics) or return to quit: ')
                       
                        num_rec = int(input('Enter the number of recommendations (has to be an interger > 0): '))
                        
            
                        rankings = get_hybrid_recommendations(prefs,movie_similarity_matrix,item_similarity_matrix, user, Weight_parameter)
                        if len(rankings) == 0 or num_rec > len(rankings):
                               print(rankings)
                        else:
                               print('Recommendations:')
                               print()
                               print(rankings[0])
                               print()
                               print("Top Movies:")
                               print()
                               print(top_N(rankings, num_rec))
                 
                   else:
                        print('Run the Hybrid command first to set up Hybrid data')

                elif algo == 'S' or algo == 's':
                    
                   if SGD_ran:
                        user = input('Enter username (for critics) or return to quit: ')
                        num_rec = int(input('Enter the number of recommendations (has to be an interger > 0): '))
                        rankings = []
                        for i in range(len(ratings)):
                            for j in range(len(ratings[0])):
                                if user_list[i] == user:
                                    if ratings[i][j] == 0:
                                        rankings.append((predictions_SGD[i][j], movie_ID_array[j]))
                        rankings.sort()
                        rankings.reverse()
                        if len(rankings) == 0 or num_rec > len(rankings):
                               print(rankings)
                        else:
                               print('Recommendations:')
                               print()
                               print(rankings[0])
                               print()
                               print("Top Movies:")
                               print()
                               print(top_N(rankings, num_rec))
                 
                   else:
                        print('Run the MF-SGD command first to set up MF-SGD data')

                elif algo == 'A' or algo == 'a':
                    
                   if ALS_ran:
                        user = input('Enter username (for critics) or return to quit: ')
                        num_rec = int(input('Enter the number of recommendations (has to be an interger > 0): '))
                        rankings = []
                        for i in range(len(ratings)):
                            for j in range(len(ratings[0])):
                                if user_list[i] == user:
                                    if ratings[i][j] == 0:
                                        rankings.append((predictions_ALS[i][j], movie_ID_array[j]))
                        rankings.sort()
                        rankings.reverse()
                        if len(rankings) == 0 or num_rec > len(rankings):
                               print(rankings)
                        else:
                               print('Recommendations:')
                               print()
                               print(rankings[0])
                               print()
                               print("Top Movies:")
                               print()
                               print(top_N(rankings, num_rec))
                 
                   else:
                        print('Run the MF-ALS command first to set up MF-ALS data')
        
                
                
                else:
                    print('Algorithm %s is invalid, try again!' % algo)
                
                
                
            #RECS for RML 100K      
            elif len(prefs) > 10:
                print('ml-100k') 
                algo = input('Enter Algorithm ( (U)U-CF, (I)I-CF, (S)GD, (A)LS, FE, TFIDF, H(ybrid) ): ')
                if algo == 'FE' or algo == 'fe':
                    if fe_ran:
                        userID = input('Enter userid (for ml-100k) or return to quit: ')
                        num_rec = int(input('Enter the number of recommendations (has to be an interger > 0): '))
                        if userID !='':
                            rankings = get_FE_recommendations(prefs, features, movie_ID, userID)
                            if len(rankings) == 0 or num_rec > len(rankings):
                               print(rankings)
                            else:
                               print('Recommendations:')
                               print()
                               print(rankings[0])
                               print()
                               print("Top Movies:")
                               print()
                               print(top_N(rankings, num_rec))
                            
                    else:
                        print('Run the FE command first to set up FE data')
                        
                elif algo == 'TFIDF' or algo == 'tfidf':  
                    if tfidf_ran: 
                        userID = input('Enter userid (for ml-100k) or return to quit: ')
                        TFIDF_threshold = float(input('Enter the value of threshold (0 <= threshold <= 1): '))
                        num_rec = int(input('Enter the number of recommendations (has to be an interger > 0): '))
                        if userID !='':
                            rankings = get_TFIDF_recommendations(prefs,movie_similarity_matrix, userID, TFIDF_threshold)
                            if len(rankings) == 0 or num_rec > len(rankings):
                               print(rankings)
                            else:
                               print('Recommendations:')
                               print()
                               print(rankings[0])
                               print()
                               print("Top Movies:")
                               print()
                               print(top_N(rankings, num_rec))
                    else:
                        print('Run the TFIDF command first to set up TFIDF data')  
                    
                
        
                elif algo == 'U' or algo == 'u':
                 
                    if user_distance_ran:
                        userID = input('Enter username (for ML-100k) or return to quit: ')
                        threshold = int(input('threshold(enter a digit, 0 <= digit <= 1)?\n'))

                        if userID !='':
                            rankings = getRecommendationsSim(prefs, userID, user_distance_matrix, sim_distance, sim_weight, threshold)
                            num_rec = int(input('Enter the number of recommendations (has to be an interger > 0): '))
                            if len(rankings) == 0 or num_rec > len(rankings):
                               print(rankings)
                            else:
                               print('Recommendations:')
                               print()
                               print(rankings[0])
                               print()
                               print("Top Movies:")
                               print()
                               print(top_N(rankings, num_rec))
                            
                    elif user_pearson_ran:
                        userID = input('Enter username (for ML-100k) or return to quit: ')
                        if userID !='':
                            rankings = getRecommendationsSim(prefs, userID, user_pearson_matrix, sim_pearson, sim_weight, threshold)
                            num_rec = int(input('Enter the number of recommendations (has to be an interger > 0): '))
                            if len(rankings) == 0 or num_rec > len(rankings):
                               print(rankings)
                            else:
                               print('Recommendations:')
                               print()
                               print(rankings[0])
                               print()
                               print("Top Movies:")
                               print()
                               print(top_N(rankings, num_rec))
                    else:
                        print('Run the Simu command first to set up Simu data') 
                
                
                elif algo == 'I' or  algo == 'i':
                     
                    if item_distance_ran:
                        userID = input('Enter username (for ML-100k) or return to quit: ')
                        threshold = int(input('threshold(enter a digit, 0 <= digit <= 1)?\n'))
                        num_rec = int(input('Enter the number of recommendations (has to be an interger > 0): '))


                        if userID !='':
                            rankings = getRecommendedItems(prefs,userID, item_distance_matrix, sim_weight, threshold)
                            if len(rankings) == 0 or num_rec > len(rankings):
                               print(rankings)
                            else:
                               print('Recommendations:')
                               print()
                               print(rankings[0])
                               print()
                               print("Top Movies:")
                               print()
                               print(top_N(rankings, num_rec))
                             
                    elif item_pearson_ran:
                        userID = input('Enter username (for ML-100k) or return to quit: ')
                        num_rec = int(input('Enter the number of recommendations (has to be an interger > 0): '))
                        threshold = int(input('threshold(enter a digit, 0 <= digit <= 1)?\n'))


                        if userID !='':
                            rankings = getRecommendedItems(prefs,userID, item_pearson_matrix, sim_weight, threshold) 
                            if len(rankings) == 0 or num_rec > len(rankings):
                               print(rankings)
                            else:
                               print('Recommendations:')
                               print()
                               print(rankings[0])
                               print()
                               print("Top Movies:")
                               print()
                               print(top_N(rankings, num_rec))
                             
                    else:
                        print('Run the Sim command first to set up Sim data') 

                elif algo == 'h' or algo == 'H':
                   
                    if hybrid_ran:
                    
                        user = input('Enter username (for ML-100k) or return to quit: ') 
                        num_rec = int(input('Enter the number of recommendations (has to be an interger > 0): '))

                        

                            
                            
                        
                        rankings = get_hybrid_recommendations(prefs,movie_similarity_matrix,item_similarity_matrix, user, Weight_parameter)
                        if len(rankings) == 0 or num_rec > len(rankings):
                               print(rankings)
                        else:
                               print('Recommendations:')
                               print()
                               print(rankings[0])
                               print()
                               print("Top Movies:")
                               print()
                               print(top_N(rankings, num_rec))
                    
                    else:
                         print('Run the hybrid command first to set up hybrid data')    
                
                

                elif algo == 'S' or algo == 's':
                    
                   if SGD_ran:
                        user = input('Enter username (for ML-100k) or return to quit: ')
                        num_rec = int(input('Enter the number of recommendations (has to be an interger > 0): '))
                        rankings = []
                        for i in range(len(ratings)):
                            for j in range(len(ratings[0])):
                                if user_list[i] == user:
                                    if ratings[i][j] == 0:
                                        if predictions_SGD[i][j] > 5:
                                            rankings.append((5, movie_ID_array[j]))
                                        elif predictions_SGD[i][j] < 0: 
                                            rankings.append((0, movie_ID_array[j]))
                                        else:
                                            rankings.append((predictions_SGD[i][j], movie_ID_array[j]))                        
                        rankings.sort()
                        rankings.reverse()
                        
                        if len(rankings) == 0 or num_rec > len(rankings):
                               print(rankings)
                        else:
                               print('Recommendations:')
                               print()
                               print(rankings[0])
                               print()
                               print("Top Movies:")
                               print()
                               print(top_N(rankings, num_rec))
                 
                   else:
                        print('Run the MF-SGD command first to set up MF-SGD data')
    
                elif algo == 'A' or algo == 'a':
                    
                   if ALS_ran:
                        user = input('Enter username (for ML-100k) or return to quit: ')
                        num_rec = int(input('Enter the number of recommendations (has to be an interger > 0): '))

                        rankings = []
                        for i in range(len(ratings)):
                            for j in range(len(ratings[0])):
                                if user_list[i] == user:
                                    if ratings[i][j] == 0:
                                        if predictions_ALS[i][j] > 5:
                                            rankings.append((5, movie_ID_array[j]))
                                        elif predictions_ALS[i][j] < 0: 
                                            rankings.append((0, movie_ID_array[j]))
                                        else:
                                            rankings.append((predictions_ALS[i][j], movie_ID_array[j]))
                                            
                        rankings.sort()
                        rankings.reverse()
                        if len(rankings) == 0 or num_rec > len(rankings):
                               print(rankings)
                        else:
                               print('Recommendations:')
                               print()
                               print(rankings[0])
                               print()
                               print("Top Movies:")
                               print()
                               print(top_N(rankings, num_rec))
                 
                   else:
                        print('Run the MF-ALS command first to set up MF-ALS data')
        
            

                else:
                    print('Algorithm %s is invalid, try again!' % algo)
                

            else:
                print ('Empty dictionary, read in some data!')
                print()
                
        elif file_io == 'Test' or file_io == 'test':

            print()
            # Load LOOCV results from file, print
            print('Results for Item_sim_distance_50_0:')
            sq_diffs_info = pickle.load(open( "data_2/sq_error_Item_dist_50_0.p", "rb" ))
            Item_errors_u_lcv_MSE, Item_errors_u_lcv = print_loocv_results(sq_diffs_info)   
            print()
            # Load LOOCV results from file, print
            print('Results for Hybrid:')
            sq_diffs_info = pickle.load(open( "data_2/sq_error_hybrid__0.p", "rb" ))
            hybrid_lcv_MSE, hybrid_lcv = print_loocv_results(sq_diffs_info) 
            
            print()
            print ('t-test for Item-sim distance vs Hybrid',len(Item_errors_u_lcv), len(hybrid_lcv))
            print ('Null Hypothesis is that the means (MSE values for Item-sim distance vs Hybrid) are equal')
            
            ## Calc with the scipy function
            t_u_lcv, p_u_lcv = stats.ttest_ind(Item_errors_u_lcv,hybrid_lcv)
            print("t = " + str(t_u_lcv))
            print("p = %.20f" %(p_u_lcv))
            print()
            print('==>> We reject  the null hypothesis that the means are equal because p<0.05') # The two-tailed p-value    
            print('==>> The means are not equal')
            
            input('\nContinue? ')

            print()
            # Load LOOCV SIM results from file, print
            print('Results for SGD~2~0.02~0.02:')
            sq_diffs_info = pickle.load(open( "data_2/sq_error_SGD_2_1.p", "rb" ))
            distance_errors_i_lcvsim_MSE, SGD_errors = print_loocv_results(sq_diffs_info)   
            print()
            # Load LOOCV SIM results from file, print
            print('Results for FE:')
            sq_diffs_info = pickle.load(open( "data_2/sq_error_FE.p", "rb" ))
            pearson_errors_i_lcvsim_MSE, FE_errors= print_loocv_results(sq_diffs_info) 
            
            print()
            print ('t-test for SGD~2~0.02~0.02 vs FE', len(SGD_errors), len(FE_errors))
            print ('Null Hypothesis is that the means (MSE values for SGD~2~0.02~0.02 vs FE) are equal')
            
            ## Calc with the scipy function
            t_i_lcvsim, p_i_lcvsim = stats.ttest_ind(SGD_errors, FE_errors)
            print("t = " + str(t_i_lcvsim))
            print("p = " + str(p_i_lcvsim))
            print('==>> We reject  the null hypothesis that the means are equal because p<0.05') # The two-tailed p-value    
            print('==>> The means are not be equal')
            
            input('\nContinue? ')
            
            print()
            print ('Cross t-tests')
            
            print()
            print ('t-test for hybrid-distance vs SGD',len(hybrid_lcv), len(SGD_errors))
            print ('Null Hypothesis is that the means (MSE values for hybrid -pearson and SGD ) are equal')
            
            ## Calc with the scipy function
            t_u_lcv_i_lcvsim_distance, p_u_lcv_i_lcvsim_distance = stats.ttest_ind(hybrid_lcv, SGD_errors)


            
            print()
            print('MSE hybrid-distance, SGD:',hybrid_lcv_MSE , distance_errors_i_lcvsim_MSE)
            print("t = " + str(t_u_lcv_i_lcvsim_distance))
            print("p = " + str(p_u_lcv_i_lcvsim_distance))
            print('==>> We reject  the null hypothesis that the means are equal because p<0.05') # The two-tailed p-value    
            print('==>> The means are not be equal')
            

        
            
                   
        else:
            done = True

    
    print('\nGoodbye!')
        
if __name__ == '__main__':
    main()