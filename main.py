from clourd_resource import CloudResourceAllocation
from compares import compare_initialization_methods, compare_tasks_resources, compare_different_number_of_resources, \
    compare_brute_force_performance
from files import save_results_to_csv
from generators import generate_cost_matrix, generate_processing_times

if __name__ == '__main__':
    num_tasks = 5
    num_resources = 5
    # Zapis wyników do pliku CSV
    headers = ["Test", "Parametr", "Koszt Początkowy", "Koszt Końcowy"]
    data = []
    cost_matrix = generate_cost_matrix(num_tasks, num_resources)
    processing_times = generate_processing_times(num_tasks)

    print("Czas:")
    print(processing_times)

    # Utworzenie instancji klasy i uruchomienie procesu optymalizacji
    allocation_system = CloudResourceAllocation(num_tasks, num_resources, cost_matrix, processing_times)
    allocation_system.random_initialization()
    allocation_system.print_allocation_matrix("Macierz Alokacji po Losowej Inicjalizacji:")
    allocation_system.initial_optimization()
    allocation_system.print_allocation_matrix("Macierz Alokacji po Optymalizacji Początkowej:")
    allocation_system.minimize_splr()
    allocation_system.minimize_gelr()
    allocation_system.evolutionary_optimization()
    allocation_system.print_allocation_matrix("Macierz Alokacji po Optymalizacjach:")
    if allocation_system.find_nash_equilibrium():
        print("Znaleziono równowagę Nasha.")
    else:
        print("Nie znaleziono równowagi Nasha.")

    print("\nPorównanie metod inicjalizacji:")
    compare_initialization_methods(num_tasks, num_resources)

    # Porównanie wyników dla różnej liczby zadań
    print("\nPorównanie wyników dla różnej liczby zadań:")
    tasks_costs, resources_costs = compare_tasks_resources(10, 10)
    for num_tasks, cost in zip(range(1, 11), tasks_costs):
        print(f"Liczba zadań: {num_tasks}, Średni koszt: {cost}")
    for num_resources, cost in zip(range(1, 11), resources_costs):
        print(f"Liczba zasobów: {num_resources}, Średni koszt: {cost}")

    # Porównanie wyników dla różnej liczby zasobów
    print("\nPorównanie wyników dla różnej liczby zasobów:")
    resources_costs_initial, resources_costs_final = compare_different_number_of_resources(num_tasks, 10)
    for num_resources, cost_initial, cost_final in zip(range(1, 2), resources_costs_initial, resources_costs_final):
        print(f"Liczba zasobów: {num_resources}, Koszt początkowy: {cost_initial}, Koszt końcowy: {cost_final}")
        # Porównanie oceny wydajności podejścia brute force
        print("\nPorównanie oceny wydajności podejścia brute force:")
        optimized_cost, brute_force_cost = compare_brute_force_performance(num_tasks, num_resources)
        print(f"Średni koszt optymalizacji: {optimized_cost}")
        print(f"Średni koszt Brute Force: {brute_force_cost}")

        data.append(["Brute Force dla małej instancji", "Optimized", brute_force_cost, ""])
        data.append(["Brute Force dla małej instancji", "Brute Force", "", brute_force_cost])

        # Zapisz dane do pliku CSV
        save_results_to_csv("wyniki_testow.csv", headers, data)

        print("Wyniki eksperymentów zostały zapisane do pliku 'wyniki_testow.csv'")
