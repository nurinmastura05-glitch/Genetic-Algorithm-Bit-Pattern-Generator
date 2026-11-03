import streamlit as st
import random
import numpy as np

# =====================================================================
# PARAMETERS (As specified in problem statement)
# =====================================================================
POPULATION_SIZE = 300    # Condition 1: Population setup is 300
CHROMOSOME_LENGTH = 80   # Condition 3: Length of individuals is 80
TARGET_ONES = 50         # Condition 2: Maximum value when count of 1s = 50
MAX_FITNESS = 80         # Condition 4: Fitness reaches 80 when count of 1s = 50
NUM_GENERATIONS = 50     # Condition 5: Number of generations is 50

MUTATION_RATE = 0.01     # Probability of bit flip
CROSSOVER_RATE = 0.8     # Probability of single-point crossover


def fitness_function(individual):
    """
    Calculates fitness based on conditions 2 & 4:
    - Maximum fitness (80) occurs when count of 1s is 50.
    - Fitness decreases linearly as count deviates from 50.
    """
    num_ones = sum(individual)
    fitness = MAX_FITNESS - abs(num_ones - TARGET_ONES)
    return fitness


def create_individual(length):
    """Generates a random bit string individual of specified length."""
    return [random.randint(0, 1) for _ in range(length)]


def create_population(pop_size, length):
    """Generates an initial population of individuals."""
    return [create_individual(length) for _ in range(pop_size)]


def selection(population, fitnesses):
    """Tournament selection mechanism (size = 3)."""
    tournament_size = 3
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]


def crossover(parent1, parent2):
    """Single-point crossover between two parents."""
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, CHROMOSOME_LENGTH - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1[:], parent2[:]


def mutate(individual):
    """Bit-flip mutation based on mutation rate."""
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]
    return individual


def run_genetic_algorithm():
    """Main Genetic Algorithm execution loop."""
    # 1. Initialize Population
    population = create_population(POPULATION_SIZE, CHROMOSOME_LENGTH)
    
    best_overall_individual = None
    best_overall_fitness = -1
    
    print(f"{'Gen':<6} | {'Best Fitness':<14} | {'Ones Count':<12} | {'Sample Segment (First 20 Bits)'}")
    print("-" * 70)
    
    for gen in range(1, NUM_GENERATIONS + 1):
        # Evaluate Fitness
        fitnesses = [fitness_function(ind) for ind in population]
        
        # Track best individual of current generation
        max_fit = max(fitnesses)
        best_ind = population[fitnesses.index(max_fit)]
        ones_cnt = sum(best_ind)
        
        if max_fit > best_overall_fitness:
            best_overall_fitness = max_fit
            best_overall_individual = best_ind
            
        sample_bits = "".join(map(str, best_ind[:20])) + "..."
        print(f"{gen:<6} | {max_fit:<14} | {ones_cnt:<12} | {sample_bits}")
        
        # Stop early if maximum possible fitness (80) is reached
        if max_fit == MAX_FITNESS:
            print(f"\n[!] Target criteria (Fitness={MAX_FITNESS}) achieved at Generation {gen}!")
            break
            
        # 2. Reproduction / Next Generation Generation
        new_population = []
        
        # Elitism: Retain top individual directly
        new_population.append(best_ind[:])
        
        while len(new_population) < POPULATION_SIZE:
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            
            child1, child2 = crossover(parent1, parent2)
            
            new_population.append(mutate(child1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(child2))
                
        population = new_population

    # Final Summary Output
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Best Fitness Achieved : {best_overall_fitness} / {MAX_FITNESS}")
    print(f"Number of 1s          : {sum(best_overall_individual)} / {CHROMOSOME_LENGTH}")
    print(f"Bit Pattern String    : {''.join(map(str, best_overall_individual))}")

if __name__ == "__main__":
    run_genetic_algorithm()