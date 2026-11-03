import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Genetic Algorithm Simulation",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Genetic Algorithm Simulation")
st.write("Optimization using Genetic Algorithm")

# =========================================================
# PARAMETERS
# =========================================================
POPULATION_SIZE = 300
CHROMOSOME_LENGTH = 80
TARGET_ONES = 50
MAX_FITNESS = 80
NUM_GENERATIONS = 50

MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.8


# =========================================================
# FITNESS FUNCTION
# =========================================================
def fitness_function(individual):
    num_ones = sum(individual)
    return MAX_FITNESS - abs(num_ones - TARGET_ONES)


# =========================================================
# CREATE INDIVIDUAL AND POPULATION
# =========================================================
def create_individual(length):
    return [random.randint(0, 1) for _ in range(length)]


def create_population(pop_size, length):
    return [
        create_individual(length)
        for _ in range(pop_size)
    ]


# =========================================================
# SELECTION
# =========================================================
def selection(population, fitnesses):
    tournament_size = 3

    selected = random.sample(
        list(zip(population, fitnesses)),
        tournament_size
    )

    selected.sort(key=lambda x: x[1], reverse=True)

    return selected[0][0]


# =========================================================
# CROSSOVER
# =========================================================
def crossover(parent1, parent2):

    if random.random() < CROSSOVER_RATE:

        point = random.randint(1, CHROMOSOME_LENGTH - 1)

        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]

        return child1, child2

    return parent1[:], parent2[:]


# =========================================================
# MUTATION
# =========================================================
def mutate(individual):

    for i in range(len(individual)):

        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]

    return individual


# =========================================================
# GENETIC ALGORITHM
# =========================================================
def run_genetic_algorithm():

    population = create_population(
        POPULATION_SIZE,
        CHROMOSOME_LENGTH
    )

    history = []

    best_overall_individual = None
    best_overall_fitness = -1

    for gen in range(1, NUM_GENERATIONS + 1):

        # Evaluate fitness
        fitnesses = [
            fitness_function(ind)
            for ind in population
        ]

        # Find best individual
        max_fit = max(fitnesses)

        best_index = fitnesses.index(max_fit)

        best_ind = population[best_index][:]

        ones_count = sum(best_ind)

        # Track best overall individual
        if max_fit > best_overall_fitness:

            best_overall_fitness = max_fit

            best_overall_individual = best_ind[:]

        # Store generation information
        history.append({
            "Generation": gen,
            "Best Fitness": max_fit,
            "Number of Ones": ones_count,
            "Average Fitness": sum(fitnesses) / len(fitnesses)
        })

        # Stop if maximum fitness is achieved
        if max_fit == MAX_FITNESS:
            break

        # Create next generation
        new_population = []

        # Elitism
        new_population.append(best_ind[:])

        while len(new_population) < POPULATION_SIZE:

            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)

            child1, child2 = crossover(parent1, parent2)

            new_population.append(mutate(child1))

            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(child2))

        population = new_population

    return (
        best_overall_individual,
        best_overall_fitness,
        history
    )


# =========================================================
# DISPLAY PARAMETERS
# =========================================================
st.subheader("Algorithm Parameters")

col1, col2, col3 = st.columns(3)

col1.metric("Population Size", POPULATION_SIZE)
col2.metric("Chromosome Length", CHROMOSOME_LENGTH)
col3.metric("Target Number of Ones", TARGET_ONES)

col4, col5, col6 = st.columns(3)

col4.metric("Maximum Fitness", MAX_FITNESS)
col5.metric("Maximum Generations", NUM_GENERATIONS)
col6.metric("Mutation Rate", MUTATION_RATE)


# =========================================================
# RUN SIMULATION
# =========================================================
st.divider()

st.subheader("Run Genetic Algorithm")

if st.button("▶ Run Simulation", type="primary"):

    with st.spinner("Running Genetic Algorithm..."):

        best_individual, best_fitness, history = (
            run_genetic_algorithm()
        )

    st.success("Genetic Algorithm completed successfully!")

    # Results
    st.subheader("Final Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Best Fitness",
        f"{best_fitness} / {MAX_FITNESS}"
    )

    col2.metric(
        "Number of Ones",
        f"{sum(best_individual)} / {CHROMOSOME_LENGTH}"
    )

    col3.metric(
        "Generations Used",
        len(history)
    )

    # Chromosome
    st.subheader("Best Chromosome")

    chromosome = "".join(map(str, best_individual))

    st.code(chromosome)

    # Generation table
    st.subheader("Generation Statistics")

    df = pd.DataFrame(history)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # Fitness graph
    st.subheader("Fitness Progression")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        df["Generation"],
        df["Best Fitness"],
        marker="o",
        label="Best Fitness"
    )

    ax.plot(
        df["Generation"],
        df["Average Fitness"],
        linestyle="--",
        label="Average Fitness"
    )

    ax.axhline(
        y=MAX_FITNESS,
        linestyle=":",
        label="Maximum Fitness"
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Genetic Algorithm Fitness Progress")

    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    if best_fitness == MAX_FITNESS:
        st.success("Target fitness of 80 achieved!")
    else:
        st.warning(
            "Maximum fitness was not achieved within 50 generations."
        )

else:

    st.info(
        "Click the Run Simulation button to start the Genetic Algorithm."
    )