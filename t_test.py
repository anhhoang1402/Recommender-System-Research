'''
LOOCV results for critics data using 
Similarity methods: sim_distance(), sim_pearson()
Recommenders: User-based CF, Item-based CF

Examples of Hypothesis Testing
==> Only for use in the CSC 381 Recommender Systems course.

Author: Carlos Seminario

'''

import os
import pickle
import numpy as np
# from matplotlib import pyplot as plt
from scipy import stats # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html#scipy.stats.ttest_ind
'''
scipy.stats.ttest_ind(a, b, axis=0, equal_var=True, nan_policy='propagate')
 Calculate the T-test for the means of two independent samples of scores.
 This is a two-sided test for the null hypothesis that 2 independent samples 
 have identical average (expected) values. This test assumes that the 
 populations have identical variances by default.
'''

def print_loocv_results(error_list):
    ''' Print LOOCV SIM results '''



                
    #print()
    error = sum(tuple(error_list))/len(error_list)          
    print ('MSE =', error)
    
    return(error, error_list)
                
                
def main():
    ''' User interface for Python console '''
    
    path = os.getcwd() # this gets the current working directory
                       # you can customize path for your own computer here
    print('\npath: %s' % path) # debug    
    
    print()
    # Load LOOCV results from file, print
    print('Results for User_sim_distance_25_0:')
    sq_diffs_info = pickle.load(open( "data_2/sq_error_USer_dist_25_0.p", "rb" ))
    distance_errors_u_lcv_MSE, distance_errors_u_lcv = print_loocv_results(sq_diffs_info)   
    print()
    # Load LOOCV results from file, print
    print('Results for User_sim_pearson_25_0:')
    sq_diffs_info = pickle.load(open( "data_2/sq_error_USer_Pearson_25_0.p", "rb" ))
    pearson_errors_u_lcv_MSE, pearson_errors_u_lcv = print_loocv_results(sq_diffs_info) 
    
    print()
    print ('t-test for User-sim distance vs sim pearson',len(distance_errors_u_lcv), len(pearson_errors_u_lcv))
    print ('Null Hypothesis is that the means (MSE values for User-sim distance vs sim pearson) are equal')
    
    ## Calc with the scipy function
    t_u_lcv, p_u_lcv = stats.ttest_ind(distance_errors_u_lcv,pearson_errors_u_lcv)
    print("t = " + str(t_u_lcv))
    print("p = " + str(p_u_lcv))
    print()
    print('==>> We reject  the null hypothesis that the means are equal because p<0.05') # The two-tailed p-value    
    print('==>> The means are not be equal')
    
    input('\nContinue? ')

    print()
    # Load LOOCV SIM results from file, print
    print('Results for Item_sim_distance_25_0:')
    sq_diffs_info = pickle.load(open( "data_2/sq_error_item-distance_25_0.p", "rb" ))
    distance_errors_i_lcvsim_MSE, distance_errors_i_lcvsim = print_loocv_results(sq_diffs_info)   
    print()
    # Load LOOCV SIM results from file, print
    print('Results for Item_sim_pearson_25_0:')
    sq_diffs_info = pickle.load(open( "data_2/sq_error_Item_Pearson_25_0.p", "rb" ))
    pearson_errors_i_lcvsim_MSE, pearson_errors_i_lcvsim = print_loocv_results(sq_diffs_info) 
    
    print()
    print ('t-test for Item_sim_distance_25_0 vs sim_pearson_25_0', len(pearson_errors_i_lcvsim), len(distance_errors_i_lcvsim))
    print ('Null Hypothesis is that the means (MSE values for Item_sim_distance_25_0 vs sim_pearson_25_0) are equal')
    
    ## Calc with the scipy function
    t_i_lcvsim, p_i_lcvsim = stats.ttest_ind(pearson_errors_i_lcvsim, distance_errors_i_lcvsim)
    print("t = " + str(t_i_lcvsim))
    print("p = " + str(p_i_lcvsim))
    print('==>> We reject  the null hypothesis that the means are equal because p<0.05') # The two-tailed p-value    
    print('==>> The means are not be equal')
    
    input('\nContinue? ')
    
    print()
    print ('Cross t-tests')
    
    print()
    print ('t-test for User_sim_distance_25_0 vs Item_sim_distance_25_0',len(distance_errors_i_lcvsim), len(distance_errors_u_lcv))
    print ('Null Hypothesis is that the means (MSE values for Item-LCVSIM distance and User-LCV distance) are equal')
    
    ## Calc with the scipy function
    t_u_lcv_i_lcvsim_distance, p_u_lcv_i_lcvsim_distance = stats.ttest_ind(distance_errors_i_lcvsim, distance_errors_u_lcv)


    
    print()
    print('User_sim_distance_25_0, Item_sim_distance_25_0:', distance_errors_i_lcvsim_MSE, distance_errors_u_lcv_MSE)
    print("t = " + str(t_u_lcv_i_lcvsim_distance))
    print("p = " + str(p_u_lcv_i_lcvsim_distance))
    print('==>> We reject  the null hypothesis that the means are equal because p<0.05') # The two-tailed p-value    
    print('==>> The means are not be equal')
    

    print()
    print ('t-test for User_sim_Pearson_25_0 Vs Item_sim_Pearson_25_0',len(pearson_errors_i_lcvsim), len(pearson_errors_u_lcv))
    print ('Null Hypothesis is that the means (MSE values for for User_sim_Pearson_25_0 Vs Item_sim_Pearson_25_0) are equal')
    
    ## Cross Checking with the scipy function
    t_u_lcv_i_lcvsim_pearson, p_u_lcv_i_lcvsim_pearson = stats.ttest_ind(pearson_errors_i_lcvsim, pearson_errors_u_lcv)
    print()
    print('for User_sim_Pearson_25_0 Vs Item_sim_Pearson_25_0:', pearson_errors_i_lcvsim_MSE, pearson_errors_u_lcv_MSE)   
    print("t = " + str(t_u_lcv_i_lcvsim_pearson))
    print("p = " + str(p_u_lcv_i_lcvsim_pearson))
    print('==>> We reject  the null hypothesis that the means are equal because p<0.05') # The two-tailed p-value    
    print('==>> The means are not be equal')
    


if __name__ == '__main__':
    main()