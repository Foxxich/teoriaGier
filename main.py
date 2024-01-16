from clourd_resource import CloudResourceAllocation
from compares import compare_initialization_methods, compare_tasks_resources, compare_different_number_of_resources, \
    compare_brute_force_performance
from files import save_results_to_csv
from generators import generate_cost_matrix, generate_processing_times
import time

if __name__ == '__main__':
    num_tasks = 5
    num_resources = 5

    # Zapis wyników do pliku CSV
    headers = ["Approach", "Execution Time", "Initial Cost", "Final Cost"]
    data = []
    cost_matrix = generate_cost_matrix(num_tasks, num_resources)
    processing_times = generate_processing_times(num_tasks)

    print("Processing Times:")
    print(processing_times)

    # Utworzenie instancji klasy i uruchomienie procesu optymalizacji
    start_time = time.time()
    allocation_system = CloudResourceAllocation(num_tasks, num_resources, cost_matrix, processing_times)
    allocation_system.random_initialization()
    allocation_system.initial_optimization()
    allocation_system.minimize_splr()
    allocation_system.minimize_gelr()
    allocation_system.evolutionary_optimization()
    execution_time = time.time() - start_time
    initial_cost = allocation_system.calculate_total_cost()

    print("\nComparison of Initialization Methods:")
    compare_initialization_methods(num_tasks, num_resources)

    final_cost = allocation_system.calculate_total_cost()
    data.append(["Comparison of Initialization Methods", execution_time, initial_cost, final_cost])

    # Porównanie wyników dla różnej liczby zadań
    start_time = time.time()
    tasks_costs, resources_costs = compare_tasks_resources(10, 10)
    execution_time = time.time() - start_time

    data.append(
        ["Comparison of Results for Different Number of Tasks", execution_time, tasks_costs[-1], resources_costs[-1]])

    # Porównanie wyników dla różnej liczby zasobów
    start_time = time.time()
    resources_costs_initial, resources_costs_final = compare_different_number_of_resources(num_tasks, 10)
    execution_time = time.time() - start_time

    data.append(["Comparison of Results for Different Number of Resources", execution_time, resources_costs_initial[-1],
                 resources_costs_final[-1]])

    # Porównanie oceny wydajności podejścia brute force
    start_time = time.time()
    optimized_cost, brute_force_cost = compare_brute_force_performance(num_tasks, num_resources)
    execution_time = time.time() - start_time

    data.append(["Brute Force Approach", execution_time, brute_force_cost, optimized_cost])

    # Zapisz dane do pliku CSV
    save_results_to_csv("wyniki_testow.csv", headers, data)

    print("Results of experiments have been saved to 'wyniki_testow.csv'")
