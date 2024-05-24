from __future__ import annotations
import numpy as np
from softpy.evolutionary.singlestate import MetaHeuristicsAlgorithm
import time
from joblib import Parallel, delayed
import gc
import os
import pickle


def calculate_generation(iterations, sel):
        t1 = time.time()

        with open('pop.pkl', 'rb') as f:
            pop = pickle.load(f)
        with open('fit.pkl', 'rb') as f:
            fit = pickle.load(f)
        
        list = []
        for j in range (int(iterations)):  
            px1, current_step = sel(fit)
            p1 = pop[px1]
            px2, current_step = sel(fit)
            p2 = pop[px2]
            list.append(p1.recombine(p2).mutate())
            list.append(p2.recombine(p1).mutate())    
        t2 = time.time()
        print(str(os.getpid()) + " " + str(iterations))
        return list

class GeneticAlgorithm(MetaHeuristicsAlgorithm):
    '''
    A generic implementation of an evolutionary algorithm. It supports different individual candidates' representation formats and can thus be used to
    implement the traditional genetic algorithm as well as variants such as genetic programming. It requires specification of the selection and fitness
    function, of the population size as well as whether the algorithm should apply elitism (and with how many elite individuals) or not.
    The **kwargs argument is used to provide additional arguments for individual candidates' initialization.
    Note that the algorithm is designed to solve a maximization problem: if a minimization problem is to be solved instead, this must be taken care of in the
    fitness function.
    '''
    def __init__(self, pop_size: int, candidate_type, selection_func, fitness_func, elitism=False, n_elite=1, **kwargs):
        self.pop_size = pop_size
        self.selection_func = selection_func
        self.candidate_type = candidate_type
        self.fitness_func = fitness_func
        self.kwargs = kwargs
        self.elitism = elitism
        self.n_elite = n_elite

    def fit(self, n_iters=10, keep_history=False, show_iters=False, n_cores=1): 
        start_time = time.time()
        if self.elitism and (self.pop_size + self.n_elite)%2 != 0:
            self.pop_size += 1

        self.population = np.array([self.candidate_type.generate(**self.kwargs) for i in range(self.pop_size)])
        self.fitness = np.zeros(self.population.shape)
        self.best = None
        self.fitness_best = np.NINF

        if keep_history:
            self.best_h = np.empty(n_iters+1, dtype=self.candidate_type)
            self.fitness_h = np.zeros(n_iters+1)
            self.fitness_h[:] = np.NINF
        profile_arr = []

        init_time = time.time()
        elab_tot = int(self.pop_size-self.n_elite)/2
        elab_thread = self.divide_number(elab_tot, n_cores)
                
        with Parallel(n_jobs = n_cores) as parallel:
            for it in range(n_iters+1):
                iter_start = time.time()
                if show_iters:
                    print(it)
                self.fitness = np.vectorize(self.fitness_func)(self.population) #crea un vettore a cui applica per ogni elemento di population la funzione fitness
                v = np.max(self.fitness)
                self.best = self.population[np.argmax(self.fitness)] if v > self.fitness_best else self.best
                self.fitness_best = v if v > self.fitness_best else self.fitness_best

                if keep_history:
                    self.best_h[it] = self.best
                    self.fitness_h[it] = self.fitness_best

                q = np.empty(self.pop_size, dtype=self.candidate_type)
                if self.elitism:
                    q[:self.n_elite] = self.population[np.argsort(self.fitness)[::-1][:self.n_elite]]
                
                i = self.n_elite if self.elitism else 0
                sub = self.n_elite if self.elitism else 0
                current_step = None

                idx = np.random.permutation(range(len(self.fitness)))
                self.population = self.population[idx]
                self.fitness = self.fitness[idx]

                
                with open('pop.pkl', 'wb') as f:
                    pickle.dump(self.population, f)
                with open('fit.pkl', 'wb') as f:
                    pickle.dump(self.fitness, f)


                s1_time = time.time()
                                
                results = parallel(delayed(calculate_generation)(elab_thread[k], self.selection_func) for k in range (int(n_cores)))

                s2_time = time.time()
                print("----- Iterazione "+str(it)+" -----")
                
                q[sub:] = np.concatenate(results)
                
                self.population = q

                gc.collect()
                
                profile_arr.append([s1_time - iter_start, s2_time-s1_time, 0])

        return self.best, profile_arr, init_time-start_time
    

    def divide_number(self, num, parts):
        part_size = num // parts
    
        remainder = num % parts
        
        parts_list = [part_size] * parts
        
        for i in range(int(remainder)):
            parts_list[i] += 1
        
        return parts_list



    

class SteadyStateGeneticAlgorithm(MetaHeuristicsAlgorithm):
    '''
    A generic implementation of a steady-state evolutionary algorithm. It supports different individual candidates' representation formats and can thus be used to
    implement the traditional genetic algorithm as well as variants such as genetic programming. It requires specification of the selection and fitness
    function, as well as of the population size.
    The **kwargs argument is used to provide additional arguments for individual candidates' initialization.
    Note that the algorithm is designed to solve a maximization problem: if a minimization problem is to be solved instead, this must be taken care of in the
    fitness function.
    '''
    def __init__(self, pop_size: int, candidate_type, selection_func, fitness_func, **kwargs):
        self.pop_size = pop_size
        self.selection_func = selection_func
        self.candidate_type = candidate_type
        self.fitness_func = fitness_func
        self.kwargs = kwargs

    def fit(self, n_iters=10, keep_history=False):
        '''
        Applies the genetic algorithm for a given number of iterations. Notice that the implemented recombination is non-standard as it is called two
        times rather than only once. The algorithm allows for global state tracking in the selection function (as in stochastic universal selection) by
        using an explictly defined state tracking variable (current_step). At each iteration, two individual candidate at random are selected for being replaced by
        the generated children individual candidates.
        '''
        self.population = np.array([self.candidate_type.generate(**self.kwargs) for i in range(self.pop_size)])
        self.fitness = np.zeros(self.population.shape)
        self.best = None
        self.fitness_best = np.NINF

        if keep_history:
            self.best_h = np.empty(n_iters+1, dtype=self.candidate_type)
            self.fitness_h = np.zeros(n_iters+1)
            self.fitness_h[:] = np.NINF

        self.fitness = np.vectorize(self.fitness_func)(self.population)
        v = np.max(self.fitness)
        self.best = self.population[np.argmax(self.fitness)] if v > self.fitness_best else self.best
        self.fitness_best = v if v > self.fitness_best else self.fitness_best

        if keep_history:
            self.best_h[0] = self.best
            self.fitness_h[0] = self.fitness_best

        for it in range(1, n_iters+1):
            p1, current_step = self.population[self.selection_func(self.fitness, current_step=current_step)]
            p2, current_step = self.population[self.selection_func(self.fitness, current_step=current_step)]
            c1 = p1.recombine(p2).mutate()
            c2 = p2.recombine(p1).mutate()

            f1 = self.fitness_func(c1)
            f2 = self.fitness_func(c2)
            if f1 > self.fitness_best:
                self.best = c1
                self.fitness_best = f1
            if f2 > self.fitness_best:
                self.best = c2
                self.fitness_best = f2

            if keep_history:
                self.best_h[it] = self.best
                self.fitness_h[it] = self.fitness_best 

            die = np.random.choice(range(self.fitness.shape[0]), 2, replace=False)
            self.population[die[0]] = c1
            self.population[die[1]] = c2

        return self.best
