from matplotlib import pyplot as plt

from clourd_resource import CloudResourceAllocation
from generators import generate_cost_matrix, generate_processing_times


def compare_different_number_of_resources(num_tasks, max_resources, trials=5):
    resources_costs_initial = []
    resources_costs_final = []
    for num_resources in range(1, max_resources + 1):
        cost_initial, cost_final = [], []
        for _ in range(trials):
            cost_matrix = generate_cost_matrix(num_tasks, num_resources)
            processing_times = generate_processing_times(num_tasks)
            system = CloudResourceAllocation(num_tasks, num_resources, cost_matrix, processing_times)
            system.initial_optimization()
            cost_initial.append(system.calculate_total_cost())
            system.minimize_splr()
            system.minimize_gelr()
            system.evolutionary_optimization()
            cost_final.append(system.calculate_total_cost())
        resources_costs_initial.append(sum(cost_initial) / trials)
        resources_costs_final.append(sum(cost_final) / trials)
    return resources_costs_initial, resources_costs_final


def compare_initialization_methods(num_tasks, num_resources, trials=10):
    random_costs = []
    initial_costs = []
    for _ in range(trials):
        cost_matrix = generate_cost_matrix(num_tasks, num_resources)
        processing_times = generate_processing_times(num_tasks)

        # Losowa inicjalizacja
        system_random = CloudResourceAllocation(num_tasks, num_resources, cost_matrix, processing_times)
        system_random.random_initialization()
        random_costs.append(system_random.calculate_total_cost())

        # Inicjalizacja początkowa
        system_initial = CloudResourceAllocation(num_tasks, num_resources, cost_matrix, processing_times)
        system_initial.initial_optimization()
        initial_costs.append(system_initial.calculate_total_cost())

    # Generowanie wykresu
    plt.figure()
    plt.boxplot([random_costs, initial_costs], labels=['Losowa', 'Początkowa'])
    plt.ylabel('Koszt')
    plt.title('Porównanie Metod Inicjalizacji')
    plt.savefig('PorównanieMetodInicjalizacji.jpg')


def compare_tasks_resources(max_tasks, max_resources, trials=5):
    tasks_costs = []
    for num_tasks in range(1, max_tasks + 1):
        cost = []
        for _ in range(trials):
            cost_matrix = generate_cost_matrix(num_tasks, max_resources)
            processing_times = generate_processing_times(num_tasks)
            system = CloudResourceAllocation(num_tasks, max_resources, cost_matrix, processing_times)
            system.initial_optimization()
            system.minimize_splr()
            system.minimize_gelr()
            system.evolutionary_optimization()
            cost.append(system.calculate_total_cost())
        tasks_costs.append(sum(cost) / trials)

    resources_costs = []
    for num_resources in range(1, max_resources + 1):
        cost = []
        for _ in range(trials):
            cost_matrix = generate_cost_matrix(max_tasks, num_resources)
            processing_times = generate_processing_times(max_tasks)
            system = CloudResourceAllocation(max_tasks, num_resources, cost_matrix, processing_times)
            system.initial_optimization()
            system.minimize_splr()
            system.minimize_gelr()
            system.evolutionary_optimization()
            cost.append(system.calculate_total_cost())
        resources_costs.append(sum(cost) / trials)

    return tasks_costs, resources_costs


def compare_brute_force_performance(num_tasks, num_resources, trials=10):
    optimized_costs = []
    brute_force_costs = []

    for _ in range(trials):
        cost_matrix = generate_cost_matrix(num_tasks, num_resources)
        processing_times = generate_processing_times(num_tasks)

        # Optymalizacja za pomocą istniejących metod
        system = CloudResourceAllocation(num_tasks, num_resources, cost_matrix, processing_times)
        system.initial_optimization()
        optimized_costs.append(system.calculate_total_cost())

        # Podejście Brute Force
        system.brute_force_optimization()
        brute_force_costs.append(system.calculate_total_cost())

    # Generowanie wykresu
    plt.figure()
    plt.boxplot([optimized_costs, brute_force_costs], labels=['Optymalizowane', 'Brute Force'])
    plt.ylabel('Koszt')
    plt.title('Porównanie Wydajności Brute Force')
    plt.savefig('PorównanieWydajnościBruteForce.jpg')

    # Zwracanie średnich kosztów
    return sum(optimized_costs) / len(optimized_costs), sum(brute_force_costs) / len(brute_force_costs)

