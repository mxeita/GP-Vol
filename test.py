#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 11:59:48 2023

@author: mohamedeita

"""
import prob
import numpy as np
import matplotlib.pyplot as plt
import unittest

theta_1=prob.Theta(0.01622808,0.00031241,1.68705742,0.15598422,0.05555129,)

theta_2=prob.Theta([0.18755063,  0.0065807 ], -0.12707845,1.36697442,1.68705742,0.01270785)

theta_3=prob.Theta([0.30255202,  0.21196404,  0.29871223],  [-0.12707845, -0.05733982] ,0.85508543, 1.95229489, 0.06340297)


class TestTheta(unittest.TestCase):

    def test_len_1(self):
        ans = len(theta_1)
        self.assertEqual(ans, 5, 'The length of theta_1 is wrong')
    
    def test_len_2(self):
         ans = len(theta_2)
         self.assertEqual(ans, 6, 'The length of theta_2 is wrong')
    
    def test_len_3(self):
        ans = len(theta_3)
        self.assertEqual(ans, 8, 'The length of theta_3 is wrong')
    
    def test_to_list(self):
        ans = theta_1.to_list()
        self.assertEqual(ans, [0.01622808, 0.00031241, 1.68705742, 0.15598422, 0.05555129], 
                         'The to._list() of theta_1 is wrong')
    
    def test_to_list_2(self):
         ans = theta_2.to_list()
         self.assertEqual(ans, [0.18755063, 0.0065807, -0.12707845, 1.36697442, 1.68705742, 0.01270785], 
                          'The to._list() of theta_2 is wrong')
    
    def test_to_list_3(self):
        ans = theta_3.to_list()
        self.assertEqual(ans, [0.30255202, 0.21196404, 0.29871223, -0.12707845, -0.05733982, 0.85508543, 1.95229489, 0.06340297], 
                         'The to._list() of theta_3 is wrong')
    

if __name__ == '__main__':
    unittest.main()

# sigma_e=1

# T=200

# v_rand=np.random.normal(0,sigma_e,size=T)
# x_rand=np.random.normal(0,scale=np.sqrt(np.exp(v_rand)),size=T)

# plt.plot(v_rand, color='red')
# plt.plot(x_rand)

# gp=prob.GaussianProcess(prob.Theta(0,0,1,1,1))
# gp.set_x_t_and_v_t(x_rand, v_rand)
# pm=gp.get_params(Lamda=True)

# print(pm)


particles=np.zeros((10000,4))

for i in range(10000):
    particles[i]=prob.mvn.rvs(mean=[1,2,3,4], cov=[[4,0.1,0.1,0.1],
                                                   [0.1,3,0.1,0.1],
                                                   [0.1,0.1,2,0.1],
                                                   [0.1,0.1,0.1,1]])
print(np.mean(particles,axis=0))
part_centred=particles-np.mean(particles,axis=0)
print(np.mean(part_centred,axis=0))
print(prob.get_empirical_var_2(particles))


