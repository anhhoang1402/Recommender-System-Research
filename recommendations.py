'''
CSC381: Building a simple Recommender System
The final code package is a collaborative programming effort between the
CSC381 student(s) named below, the class instructor (Carlos Seminario), and
source code from Programming Collective Intelligence, Segaran 2007.
CSC381 Programmer/Researcher: <<Anh Hoang, Mengfan Wang, Mike Remezo, Malavika, Kalani>>
'''

import os
import time
import matplotlib
from matplotlib import pyplot as plt 
import numpy as np 
import math
from math import sqrt 
import copy as cp
import pickle ## add this to the list of import statements

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
    users = 0
    movies_lst = set()
    items = 0
    ratings = 0
    sum_rating = 0
    lst = []
    
    #Calculating the number of users/items/ratings
    for person in prefs:
       
        users += 1


        movies = prefs[person]
        if(len(movies) > items):
            items = len(movies)
        for movie in movies:
            movies_lst.add(movie)
            ratings += 1
            sum_rating += prefs[person][movie]
            lst.append(prefs[person][movie])
    #Printing the number of users/items/ratings
    items = len(movies_lst)
    print("Number of users: "+ str(users))
    print("Number of items: "+ str(items))
    print("Number of ratings: "+ str(ratings))
    
    #Calculating the overall average ratings and standarad deviation
    avg_rating = sum_rating/ratings
    sum_diff = 0
    for person in prefs:
        movies = prefs[person]
        for movie in movies:
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
    for avg in itm_lst:
        itm_avg_sum += avg
    itm_rating = itm_avg_sum/items
    sum_diff = 0
    std_dev = 0
    for avg in itm_lst:
        diff = (avg - itm_rating)**2
        sum_diff += diff
    std_dev = (sum_diff/items) ** (1/2)
    
    #Printing the average item ratings and standarad deviation
    itm_rating = "%.2f" % itm_rating
    std_dev = "%.2f" % std_dev
    print("Average item rating: " +str(itm_rating)+ " out of 5, and std dev of " +str(std_dev))
    
    #Calculating the average user ratings and standarad deviation
    usr_lst = []

    dicti_sum ={}
    dicti_count = {}
    dicti_avg = {}
    for person in prefs:
        movies = prefs[person]
        for movie in movies:
            if movie not in dicti_sum.keys():
                dicti_sum[movie] = prefs[person][movie]
            else:
                dicti_sum[movie] += prefs[person][movie]
            if movie not in dicti_count.keys():
                dicti_count[movie] = 1
            else:
                dicti_count[movie] += 1
            dicti_avg[movie] = dicti_sum[movie]/dicti_count[movie]
                
    #Calculating the highest ranked movies
    sorted_avg = sorted(dicti_avg.items(), key = lambda kv: kv[1])
    dicti_avg = dict(sorted_avg)
    avg = dicti_avg.keys()
    avg_item = []
    for i, e in reversed(list(enumerate(avg))):
        avg_item.append(e)
    
    #Calculating the most ranked movies
    sorted_count = sorted(dicti_count.items(), key = lambda kv: kv[1])
    dicti_count = dict(sorted_count)
    most = dicti_count.keys()
    most_item = []
    for i, e in reversed(list(enumerate(most))):
        most_item.append(e)
    
    overall_best = []
    for movie in avg_item:
        if dicti_count[movie] >= 20:
            overall_best.append(movie)
            
    print("Popular items -- most rated:")
    table_data = [
        ['Title', '#Ratings', 'Avg Rating'],
        [most_item[0], dicti_count[most_item[0]], "%.2f" % (dicti_avg[most_item[0]])], 
        [most_item[1], dicti_count[most_item[1]], "%.2f" % (dicti_avg[most_item[1]])],
        [most_item[2], dicti_count[most_item[2]], "%.2f" % (dicti_avg[most_item[2]])],
        [most_item[3], dicti_count[most_item[3]], "%.2f" % (dicti_avg[most_item[3]])],
        [most_item[4], dicti_count[most_item[4]], "%.2f" % (dicti_avg[most_item[4]])]
    ]
    for row in table_data:
        print("{: >20} {: >20} {: >20}".format(*row))
    
    print("\n")        
    print("Popular items -- highest rated:")
    table_data1 = [
        ['Title', 'Avg Rating', '#Ratings'],
        [avg_item[0], "%.2f" % (dicti_avg[avg_item[0]]), dicti_count[avg_item[0]]], 
        [avg_item[1], "%.2f" % (dicti_avg[avg_item[1]]), dicti_count[avg_item[1]]],
        [avg_item[2], "%.2f" % (dicti_avg[avg_item[2]]), dicti_count[avg_item[2]]],
        [avg_item[3], "%.2f" % (dicti_avg[avg_item[3]]), dicti_count[avg_item[3]]],
        [avg_item[4], "%.2f" % (dicti_avg[avg_item[4]]), dicti_count[avg_item[4]]]
    ]
    for row in table_data1:
        print("{: >20} {: >20} {: >20}".format(*row))
    
    print("\n")  
    print("Overall best rated items (number of ratings >= 20):")
    table_data2 = [
        ['Title', 'Avg Rating', '#Ratings'],
        [overall_best[0], "%.2f" % (dicti_avg[overall_best[0]]), dicti_count[overall_best[0]]], 
        [overall_best[1], "%.2f" % (dicti_avg[overall_best[1]]), dicti_count[overall_best[1]]],
        [overall_best[2], "%.2f" % (dicti_avg[overall_best[2]]), dicti_count[overall_best[2]]],
        [overall_best[3], "%.2f" % (dicti_avg[overall_best[3]]), dicti_count[overall_best[3]]],
        [overall_best[4], "%.2f" % (dicti_avg[overall_best[4]]), dicti_count[overall_best[4]]],
        [overall_best[5], "%.2f" % (dicti_avg[overall_best[5]]), dicti_count[overall_best[5]]], 
        [overall_best[6], "%.2f" % (dicti_avg[overall_best[6]]), dicti_count[overall_best[6]]],
        [overall_best[7], "%.2f" % (dicti_avg[overall_best[7]]), dicti_count[overall_best[7]]],
        [overall_best[8], "%.2f" % (dicti_avg[overall_best[8]]), dicti_count[overall_best[8]]],
        [overall_best[9], "%.2f" % (dicti_avg[overall_best[9]]), dicti_count[overall_best[9]]],
        [overall_best[10], "%.2f" % (dicti_avg[overall_best[10]]), dicti_count[overall_best[10]]], 
        [overall_best[11], "%.2f" % (dicti_avg[overall_best[11]]), dicti_count[overall_best[11]]],
        [overall_best[12], "%.2f" % (dicti_avg[overall_best[12]]), dicti_count[overall_best[12]]],
        [overall_best[13], "%.2f" % (dicti_avg[overall_best[13]]), dicti_count[overall_best[13]]],
        [overall_best[14], "%.2f" % (dicti_avg[overall_best[14]]), dicti_count[overall_best[14]]],
        [overall_best[15], "%.2f" % (dicti_avg[overall_best[15]]), dicti_count[overall_best[15]]], 
        [overall_best[16], "%.2f" % (dicti_avg[overall_best[16]]), dicti_count[overall_best[16]]],
        [overall_best[17], "%.2f" % (dicti_avg[overall_best[17]]), dicti_count[overall_best[17]]],
        [overall_best[18], "%.2f" % (dicti_avg[overall_best[18]]), dicti_count[overall_best[18]]],
        [overall_best[19], "%.2f" % (dicti_avg[overall_best[19]]), dicti_count[overall_best[19]]]
    ]
    for row in table_data2:
        print("{: >20} {: >20} {: >20}".format(*row))
    


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
    dicti_sum ={}
    dicti_count = {}
    dicti_avg = {}
    for person in prefs:
        movies = prefs[person]
        for movie in movies:
            if movie not in dicti_sum.keys():
                dicti_sum[movie] = prefs[person][movie]
            else:
                dicti_sum[movie] += prefs[person][movie]
            if movie not in dicti_count.keys():
                dicti_count[movie] = 1
            else:
                dicti_count[movie] += 1
            dicti_avg[movie] = dicti_sum[movie]/dicti_count[movie]
                
    #Calculating the highest ranked movies
    sorted_avg = sorted(dicti_avg.items(), key = lambda kv: kv[1])
    dicti_avg = dict(sorted_avg)
    avg = dicti_avg.keys()
    avg_item = []
    for i, e in reversed(list(enumerate(avg))):
        avg_item.append(e)
    
    #Calculating the most ranked movies
    sorted_count = sorted(dicti_count.items(), key = lambda kv: kv[1])
    dicti_count = dict(sorted_count)
    most = dicti_count.keys()
    most_item = []
    for i, e in reversed(list(enumerate(most))):
        most_item.append(e)
    
    overall_best = []
    for movie in avg_item:
        if dicti_count[movie] >= 5:
            overall_best.append(movie)
            
    print("Popular items -- most rated:")
    table_data = [
        ['Title', '#Ratings', 'Avg Rating'],
        [most_item[0], dicti_count[most_item[0]], "%.2f" % (dicti_avg[most_item[0]])], 
        [most_item[1], dicti_count[most_item[1]], "%.2f" % (dicti_avg[most_item[1]])],
        [most_item[2], dicti_count[most_item[2]], "%.2f" % (dicti_avg[most_item[2]])],
        [most_item[3], dicti_count[most_item[3]], "%.2f" % (dicti_avg[most_item[3]])],
        [most_item[4], dicti_count[most_item[4]], "%.2f" % (dicti_avg[most_item[4]])]
    ]
    for row in table_data:
        print("{: >20} {: >20} {: >20}".format(*row))
    
    print("\n")        
    print("Popular items -- highest rated:")
    table_data1 = [
        ['Title', 'Avg Rating', '#Ratings'],
        [avg_item[0], "%.2f" % (dicti_avg[avg_item[0]]), dicti_count[avg_item[0]]], 
        [avg_item[1], "%.2f" % (dicti_avg[avg_item[1]]), dicti_count[avg_item[1]]],
        [avg_item[2], "%.2f" % (dicti_avg[avg_item[2]]), dicti_count[avg_item[2]]],
        [avg_item[3], "%.2f" % (dicti_avg[avg_item[3]]), dicti_count[avg_item[3]]],
        [avg_item[4], "%.2f" % (dicti_avg[avg_item[4]]), dicti_count[avg_item[4]]]
    ]
    for row in table_data1:
        print("{: >20} {: >20} {: >20}".format(*row))
    
    print("\n")  
    print("Popular items -- highest rated:")
    table_data2 = [
        ['Title', 'Avg Rating', '#Ratings'],
        [overall_best[0], "%.2f" % (dicti_avg[overall_best[0]]), dicti_count[overall_best[0]]], 
        [overall_best[1], "%.2f" % (dicti_avg[overall_best[1]]), dicti_count[overall_best[1]]],
        [overall_best[2], "%.2f" % (dicti_avg[overall_best[2]]), dicti_count[overall_best[2]]],
        [overall_best[3], "%.2f" % (dicti_avg[overall_best[3]]), dicti_count[overall_best[3]]],
        [overall_best[4], "%.2f" % (dicti_avg[overall_best[4]]), dicti_count[overall_best[4]]]
    ]
    for row in table_data2:
        print("{: >20} {: >20} {: >20}".format(*row))
    
    
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
def topMatches(prefs,person,similarity=sim_pearson, n=100, sim_weight = 1):
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
def calculateSimilarUsers(prefs,n=100,similarity=sim_pearson, sim_weight=1):

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
    #user-user sim matrix
#     UUmatrix = calculateSimilarUsers(prefs,n=100,similarity=similarity, sim_weight = sim_weight) 
    # # print(UUmatrix)
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
                
    if count != 0:
        
        return error_mse/count, error_mae/count, error_rmse/count, len(error_list)  
    else:
        pass

################################Item-Based####################################

# Create the item-item similarity matrix
def calculateSimilarItems(prefs,n=100,similarity=sim_pearson, sim_weight=1):
 
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
def getRecommendedItems(prefs,user, itemMatch, sim_weight = 1, threshold = 0) :
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
        for (item2, similarity) in itemMatch[item]:
    
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
                 
                                                            
                            
        if count == 0:
            print("MSE:", "%.10f" %(error_mse),  ", MAE:", "%.10f" % (error_mae),  ", RMSE:", "%.10f" % (sqrt(error_rmse)), ", Coverage:", "%.10f" % (len(error_list)))
            return error_mse, error_mae, error_rmse, len(error_list)
        else:
            print("MSE:", "%.10f" %(error_mse/count),  ", MAE:", "%.10f" % (error_mae/count),  ", RMSE:", "%.10f" % (sqrt(error_rmse/count)), ", Coverage:", "%.10f" % (len(error_list)))
            return error_mse/count, error_mae/count, error_rmse/count, len(error_list)#Main
        
                     
                                
        
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
    itemsim = {}
    sim_method = ""
    
    while not done: 
        print()
        # Start a simple dialog
        file_io = input('R(ead) critics data from file?, \n'
                        'RML(ead ml-100k dataset'
                        'P(rint) the U-I matrix?, \n'
                        'V(alidate) the dictionary?, \n'
                        'S(tats) print? \n'
                        'D(istance) critics data? \n'
                        'PC(earson Correlation) critics data? \n'
                        'U(ser-based CF Recommendations)? \n'
                        'LCV(eave one out cross-validation)? \n'
                        'Sim(ilarity matrix) calc?, \n'
                        'Simu?, \n'
                        'LCVSIM(eave one out cross-validation)?, \n'
                        'I(tem-based CF Recommendations)?,\n'
                        '==> ')
        
        if file_io == 'R' or file_io == 'r':
            print()
            file_dir = 'data/'
            datafile = 'critics_ratings.data'
            itemfile = 'critics_movies.item'
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            print('Number of users: %d\nList of users:' % len(prefs), 
                  list(prefs.keys()))
        
        elif file_io == 'RML' or file_io == 'rml':
           print()
           file_dir = 'data/' # path from current directory
           datafile = 'u.data'  # ratings file
           itemfile = 'u.item'  # movie titles file            
           print ('Reading "%s" dictionary from file' % datafile)
           prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
           print('Number of users: %d\nList of users [0:10]:' 
                 % len(prefs), list(prefs.keys())[0:10] )  
        
        elif file_io == 'P' or file_io == 'p':
            # print the u-i matrix
            print()
            if len(prefs) > 0:
                print ('Printing "%s" dictionary from file' % datafile)
                print ('User-item matrix contents: user, item, rating')
                for user in prefs:
                    for item in prefs[user]:
                        print(user, item, prefs[user][item])
            else:
                print ('Empty dictionary, R(ead) in some data!')
                
        elif file_io == 'V' or file_io == 'v':      
            print()
            if len(prefs) > 0:
                # Validate the dictionary contents ..
                print ('Validating "%s" dictionary from file' % datafile)
                print ("critics['Lisa']['Lady in the Water'] =", 
                       prefs['Lisa']['Lady in the Water']) # ==> 2.5
                print ("critics['Toby']:", prefs['Toby']) 
                # ==> {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 
                #      'Superman Returns': 4.0}
            else:
                print ('Empty dictionary, R(ead) in some data!')
                
        elif file_io == 'S' or file_io == 's':
            print()
            filename = 'critics_ratings.data'
            if len(prefs) > 0:
                data_stats(prefs, filename)
                popular_items(prefs, filename)
            else: # Make sure there is data  to process ..
                print ('Empty dictionary, R(ead) in some data!')
        elif file_io == 'D' or file_io == 'd':
            print()
            if len(prefs) > 0:            
                print('Examples:')
                print ('Distance sim Lisa & Gene:', sim_distance(prefs, 'Lisa', 'Gene')) # 0.29429805508554946
                num=1
                den=(1+ sqrt( (2.5-3.0)**2 + (3.5-3.5)**2 + (3.0-1.5)**2 + (3.5-5.0)**2 + (3.0-3.0)**2 + (2.5-3.5)**2))
                print('Distance sim Lisa & Gene (check):', num/den)    
                print ('Distance sim Lisa & Michael:', sim_distance(prefs, 'Lisa', 'Michael')) # 0.4721359549995794
                print()
                
                print('User-User distance similarities:')
                
                for person1 in prefs:
                    for person2 in prefs:
                        if person1 != person2:
                            dist = sim_distance(prefs, person1, person2)
                            print("Distance between "+person1+" and "+ person2 +": "+ "%.2f" %(dist))
                
                print()
        elif file_io == 'U' or file_io == 'u':
            print()
            if len(prefs) > 0:             
                print ('Example:')
                user_name = 'Toby'
                print("What is your similarity weighting")
                file_io = input('1 (None)?, \n'
                                '25 (n/25)?, \n'
                                '50 (n/50)?,, \n'
                                '==> ')
                sim_weight = 1
                sim_weight = file_io
                sim_weight = int(sim_weight)
                print("What similarity do you want to use")
                file_io = input('D(istance)?, \n'
                                'P(earson)?, \n'
                                '==> ')
                sim = file_io
                sim_algo = sim_pearson
                if sim == 'D' or file_io == 'd':
                    sim_algo = sim_distance
                else:
                    sim_algo = sim_pearson
                print(sim_algo)
                get_all_UU_recs(prefs, sim_algo, 10,5, sim_weight = sim_weight)
                print('\n')
                print ('User-based CF recs for %s, sim_pearson: ' % (user_name), 
                       getRecommendationsSim(prefs, user_name, similarity = sim_algo, sim_weight = sim_weight)) 
                        # [(3.3477895267131017, 'The Night Listener'), 
                        #  (2.8325499182641614, 'Lady in the Water'), 
                        #  (2.530980703765565, 'Just My Luck')]
                # print ('User-based CF recs for %s, sim_distance: ' % (user_name),
                #        getRecommendationsSim(prefs, user_name, similarity=sim_distance, sim_weight = sim_weight)) 
                #         # [(3.457128694491423, 'The Night Listener'), 
                #         #  (2.778584003814924, 'Lady in the Water'), 
                #         #  (2.422482042361917, 'Just My Luck')]
                # print()
                
                # print('User-based CF recommendations for all users:')
                # print('\n')
                # print('Using sim_pearson: ')
                # get_all_UU_recs(prefs, 'pearson', 10,5, sim_weight = sim_weight)
                # print('\n')
                # print('Using sim_distance:')
                # get_all_UU_recs(prefs, 'distance', 10,5, sim_weight = sim_weight)
                # print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!')   
        
        elif file_io == 'PC' or file_io == 'pc':
            print()
            if len(prefs) > 0:             
                print ('Example:')
                print ('Pearson sim Lisa & Gene:', sim_pearson(prefs, 'Lisa', 'Gene')) # 0.39605901719066977
                print()
                
                print('Pearson for all users:')
                # Calc Pearson for all users
                for person1 in prefs:
                    for person2 in prefs:
                        if person1 != person2:
                            pear = sim_pearson(prefs, person1, person2)
                            print("Pearson sim for "+person1+" and "+ person2 +": "+ "%.10f" %(pear))
                
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!')    
        
        elif file_io == 'LCV' or file_io == 'lcv':
            ready = False # sub command in progress
            threshold = float(input('threshold(enter a digit)?\n'))
            print()
            if len(prefs) > 0:             
                    
                
                # print("What is your similarity weight")
                sim_weight = int(input('similarity weight(enter a digit)?\n'))

                sim = file_io
                sim_algo = sim_pearson
                if sim == 'D' or file_io == 'd':
                    sim_algo = sim_distance
                else:
                    sim_algo = sim_pearson
                
                error, error_list, error_rmse, error_list_rmse, error_mae, error_list_mae = loo_cv(prefs, sim_algo, sim_weight, threshold=threshold)
                print(error, error_rmse, error_mae)
            else:
                print ('Empty dictionary, R(ead) in some data!')    
    
        elif file_io == 'Sim' or file_io == 'sim' or file_io== 'Simu' or file_io=='simu':
            print()
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson? ')
                sim_weight = int(input('similarity weight(enter a digit)?\n'))
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_distance.p", "rb" ))
                        sim_method = 'sim_distance'
    
                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_pearson.p", "rb" ))  
                        sim_method = 'sim_pearson'
                        
                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        if file_io=='sim' or file_io== 'Sim':
                            itemsim = calculateSimilarItems(prefs,similarity=sim_distance, sim_weight = sim_weight)                     
                            # Dump/save dictionary to a pickle file
                            pickle.dump(itemsim, open( "save_itemsim_distance.p", "wb" ))
                            sim_method = 'sim_distance'
                        elif file_io=='simu' or file_io== 'Simu':
                            itemsim = calculateSimilarUsers(prefs,similarity=sim_distance, sim_weight = sim_weight)                     
                            # Dump/save dictionary to a pickle file
                            pickle.dump(itemsim, open( "save_itemsim_distance.p", "wb" ))
                            sim_method = 'sim_distance'
                        
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        if file_io=='sim' or file_io== 'Sim':
                            itemsim = calculateSimilarItems(prefs,similarity=sim_pearson, sim_weight = sim_weight)                     
                            # Dump/save dictionary to a pickle file
                            pickle.dump(itemsim, open( "save_itemsim_distance.p", "wb" ))
                            sim_method = 'sim_pearson'
                        elif file_io=='simu' or file_io== 'Simu':
                            itemsim = calculateSimilarUsers(prefs,similarity=sim_pearson, sim_weight = sim_weight)                     
                            # Dump/save dictionary to a pickle file
                            pickle.dump(itemsim, open( "save_itemsim_distance.p", "wb" ))
                            sim_method = 'sim_pearson'
                    
                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue
                    
                    ready = True # sub command completed successfully
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter Sim(ilarity matrix) again and choose a Write command')
                    print()
                
                

                if len(itemsim) > 0 and ready == True: 
                    # Only want to print if sub command completed successfully
                    print ('Similarity matrix based on %s, len = %d' 
                           % (sim_method, len(itemsim)))
                    print()
                    ##
                    ## enter new code here, or call a new function, 
                    ##    to print the sim matrix
                    ##
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!') 
        
        elif file_io == 'LCVSIM' or file_io == 'lcvsim':
             ready = False # sub command in progress
             threshold = float(input('threshold(enter a digit)?\n'))
             print()
             if len(prefs) > 0 and itemsim !={}:             
                print('LOO_CV_SIM Evaluation')
                sim_weight = int(input('similarity weight(enter a digit)?\n'))
                algo = input('Enter U(ser) or I(tem) algo \n')
                if algo == 'I' or algo == 'i':
                    algo = getRecommendedItems
                elif algo == 'U' or algo == 'u':
                    algo = getRecommendationsSim
                else:
                    print('invalid input')
                if len(prefs) == 7:
                    prefs_name = 'critics'
                    

                    sim = file_io
                    sim_algo = sim_pearson
                    if sim == 'D' or file_io == 'd':
                        sim_algo = sim_distance
                    else:
                        sim_algo = sim_pearson
#                 metric = input ('Enter error metric: MSE, MAE, RMSE: ')
#                 if metric == 'MSE' or metric == 'MAE' or metric == 'RMSE' or \
# 		        metric == 'mse' or metric == 'mae' or metric == 'rmse':
#                     metric = metric.upper()
#                 else:
#                     metric = 'MSE'
#                 print(sim_weight)
                if sim_method == 'sim_pearson': 
                    sim = sim_pearson
                    error_mse, error_rmse, error_mae, Coverage = loo_cv_sim(prefs, sim,itemsim, threshold, sim_weight, algo)
                    #print('Stats for %s: %.5f, len(SE list): %d, using %s' % (prefs_name, error_total, len(error_list), sim) )
                    print()
                elif sim_method == 'sim_distance':
                    sim = sim_distance
                    error_mse, error_rmse, error_mae, Coverage = loo_cv_sim(prefs, sim,itemsim, threshold, sim_weight, algo)
                   
                    #print('Stats for %s: %.5f, len(SE list): %d, using %s' % (prefs_name, error_total, len(error_list), sim) )
                    print()
                else:
                    print('Run Sim(ilarity matrix) command to create/load Sim matrix!')
#                 if prefs_name == 'critics':
#                     print(error_list)
             else:
                 print ('Empty dictionary, run R(ead) OR Empty Sim Matrix, run Sim!')
        
        elif file_io == 'I' or file_io == 'i':
            print()

            if len(prefs) > 0 and len(itemsim) > 0:                
#                 print ('Example:')
#                 user_name = 'Toby'
                print("What is your similarity threshold")
                file_io = input('1 (None)?, \n'
                                '25 (n/25)?, \n'
                                '50 (n/50)?,, \n'
                                '==> ')
                sim_weight = 1
                sim_weight = file_io
                sim_weight = int(sim_weight)
#                 print(sim_method)
#                 print ('Item-based CF recs for %s, %s: ' % (user_name, sim_method), 
#                         getRecommendedItems(prefs, itemsim, user_name, sim_weight = 1)) 
                
                ##
                ## Example:
                ## Item-based CF recs for Toby, sim_distance:  
                ##     [(3.1667425234070894, 'The Night Listener'), 
                ##      (2.9366294028444346, 'Just My Luck'), 
                ##      (2.868767392626467, 'Lady in the Water')]
                ##
                ## Example:
                ## Item-based CF recs for Toby, sim_pearson:  
                ##     [(3.610031066802183, 'Lady in the Water')]
                ##
    
                print()
                
                print('Item-based CF recommendations for all users:')
                # Calc Item-based CF recommendations for all users
        
                ## add some code above main() to calc Item-based CF recommendations 
                ## ==> write a new function to do this, as follows
               
                get_all_II_recs(prefs, itemsim, sim_method)# num_users=10, and top_N=5 by default  '''
                # Note that the item_sim dictionry and the sim_method string are
                #   setup in the main() Sim command
                
                ## Expected Results ..
                
                ## Item-based CF recs for all users, sim_distance:  
                ## Item-based CF recommendations for all users:
                ## Item-based CF recs for Lisa, sim_distance:  []
                ## Item-based CF recs for Gene, sim_distance:  []
                ## Item-based CF recs for Michael, sim_distance:  [(3.2059731906295044, 'Just My Luck'), (3.1471787551061103, 'You, Me and Dupree')]
                ## Item-based CF recs for Claudia, sim_distance:  [(3.43454674373048, 'Lady in the Water')]
                ## Item-based CF recs for Mick, sim_distance:  []
                ## Item-based CF recs for Jack, sim_distance:  [(3.5810970647618663, 'Just My Luck')]
                ## Item-based CF recs for Toby, sim_distance:  [(3.1667425234070894, 'The Night Listener'), (2.9366294028444346, 'Just My Luck'), (2.868767392626467, 'Lady in the Water')]
                ##
                ## Item-based CF recommendations for all users:
                ## Item-based CF recs for Lisa, sim_pearson:  []
                ## Item-based CF recs for Gene, sim_pearson:  []
                ## Item-based CF recs for Michael, sim_pearson:  [(4.0, 'Just My Luck'), (3.1637361366111816, 'You, Me and Dupree')]
                ## Item-based CF recs for Claudia, sim_pearson:  [(3.4436241497684494, 'Lady in the Water')]
                ## Item-based CF recs for Mick, sim_pearson:  []
                ## Item-based CF recs for Jack, sim_pearson:  [(3.0, 'Just My Luck')]
                ## Item-based CF recs for Toby, sim_pearson:  [(3.610031066802183, 'Lady in the Water')]
                    
                print()
                
            else:
                if len(prefs) == 0:
                    print ('Empty dictionary, R(ead) in some data!')
                else:
                    print ('Empty similarity matrix, use Sim(ilarity) to create a sim matrix!')    
                    
                    
        else:
            done = True
    
    print('\nGoodbye!')
        
if __name__ == '__main__':
    main()