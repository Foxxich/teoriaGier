import csv

import pulp
import random

from matplotlib import pyplot as plt


class CloudResourceAllocation:
    def __init__(self, num_tasks, num_resources, cost_matrix, processing_times):
        self.num_tasks = num_tasks
        self.num_resources = num_resources
        self.cost_matrix = cost_matrix
        self.processing_times = processing_times  # Czas przetwarzania dla każdego zadania
        self.allocation_matrix = [[0 for _ in range(num_resources)] for _ in range(num_tasks)]

    def initial_optimization(self):
        model = pulp.LpProblem("Initial_Resource_Allocation", pulp.LpMinimize)
        allocation_vars = pulp.LpVariable.dicts("Allocation",
                                                ((i, j) for i in range(self.num_tasks)
                                                 for j in range(self.num_resources)),
                                                cat='Binary')

        # Funkcja celu: minimalizacja sumy czasów przetwarzania na wszystkich zasobach
        model += pulp.lpSum([self.processing_times[i] * allocation_vars[i, j]
                             for i in range(self.num_tasks)
                             for j in range(self.num_resources)])

        # Ograniczenia jak poprzednio
        for i in range(self.num_tasks):
            model += pulp.lpSum([allocation_vars[i, j] for j in range(self.num_resources)]) == 1
        for j in range(self.num_resources):
            model += pulp.lpSum([allocation_vars[i, j] for i in range(self.num_tasks)]) <= 1

        model.solve()

        # Aktualizacja macierzy alokacji
        self.allocation_matrix = [[allocation_vars[i, j].varValue for j in range(self.num_resources)]
                                  for i in range(self.num_tasks)]

    def evolutionary_optimization(self):
        """
        Algorytm 3: Optymalizacja Ewolucyjna
        Założenie: Prosty mechanizm ewolucyjny, który losowo realokuje zadania
        """
        for _ in range(100):
            task = random.randint(0, self.num_tasks - 1)
            new_resource = random.randint(0, self.num_resources - 1)
            current_resource = self.get_current_resource(task)

            if current_resource != new_resource and self.can_improve_solution(task, new_resource):
                self.perform_reallocation(task, current_resource, new_resource)

    def can_improve_solution(self, task_index, resource_index):
        """
        Prosta heurystyka sprawdzająca możliwość poprawy
        """
        return self.calculate_utility(task_index, resource_index) < self.calculate_utility_after_reallocation(
            task_index, resource_index)

    def perform_reallocation(self, task_index, old_resource_index, new_resource_index):
        """
        Realokuje zadanie (task_index) z obecnego zasobu (old_resource_index)
        do nowego zasobu (new_resource_index).
        """
        # Usunięcie zadania z obecnego zasobu
        self.allocation_matrix[task_index][old_resource_index] = 0
        # Przypisanie zadania do nowego zasobu
        self.allocation_matrix[task_index][new_resource_index] = 1

    def minimize_splr(self):
        """
        Algorytm 1: Minimizacja SPELR
        """
        for task in range(self.num_tasks):
            min_splr = float('inf')
            best_resource = None
            current_resource = self.get_current_resource(task)  # Pobranie obecnego zasobu dla zadania

            for resource in range(self.num_resources):
                if resource != current_resource:  # Sprawdzanie, czy zasób jest różny od obecnego
                    # Obliczanie SPELR dla każdej pary zadanie-zasób
                    splr = self.compute_splr(task, resource)
                    # Znajdowanie najlepszego zasobu dla realokacji
                    if splr < min_splr:
                        min_splr = splr
                        best_resource = resource

            # Jeśli znaleziono lepszą realokację, wykonaj ją
            if best_resource is not None and min_splr < 0:
                self.perform_reallocation(task, current_resource, best_resource)

    def compute_splr(self, task_index, resource_index):
        """
        Oblicza SPELR dla danego zadania i zasobu.
        """
        current_utility = self.calculate_utility(task_index, self.get_current_resource(task_index))
        new_utility = self.calculate_utility(task_index, resource_index)
        return current_utility - new_utility

    def minimize_gelr(self):
        """
        Algorytm 2: Minimizacja GELR zgodnie z opisem w artykule
        """
        for resource in range(self.num_resources):
            mts = self.get_multiplexing_tasks(resource)
            nsts = []
            for task in mts:
                q = self.min_single(task, resource)
                if q != -1:
                    self.perform_reallocation(task, resource, q)
                    if self.calculate_utility(task, resource) - self.calculate_utility_after_reallocation(task,
                                                                                                          resource) < 0:
                        nsts.append(task)

            if nsts:
                min_gelr_task = min(nsts, key=lambda k: self.compute_gelr(k, resource))
                self.perform_reallocation(min_gelr_task, resource, self.min_single(min_gelr_task, resource))

    def compute_gelr(self, task_index, resource_index):
        """
        Prosta heurystyka obliczająca GELR
        """
        return abs(self.calculate_total_utility() - self.calculate_total_utility_after_reallocation(task_index,
                                                                                                    resource_index))

    def get_multiplexing_tasks(self, resource):
        """
        Zwraca zestaw zadań multiplexujących określony zasób.
        W przykładowej implementacji, zadanie jest uważane za multiplexujące,
        jeśli obecnie korzysta z danego zasobu.
        """
        multiplexing_tasks = []
        for task in range(self.num_tasks):
            if self.allocation_matrix[task][resource] == 1:
                multiplexing_tasks.append(task)
        return multiplexing_tasks

    def min_single(self, task, resource):
        """
        Znajduje optymalną realokację dla pojedynczego zadania.
        W tej implementacji, sprawdzamy każdy zasób inny niż obecnie przypisany
        i wybieramy ten, który minimalizuje GELR.
        """
        min_gelr = float('inf')
        best_resource = None
        for r in range(self.num_resources):
            if r != resource:
                current_gelr = self.compute_gelr(task, r)
                if current_gelr < min_gelr:
                    min_gelr = current_gelr
                    best_resource = r
        return best_resource

    def find_nash_equilibrium(self):
        changes_made = True
        while changes_made:
            changes_made = False
            for task in range(self.num_tasks):
                for resource in range(self.num_resources):
                    if self.can_improve_nash(task, resource):
                        self.perform_reallocation(task, self.get_current_resource(task), resource)
                        changes_made = True
            if not changes_made:
                return True
        return False

    def can_improve_nash(self, task_index, resource_index):
        """
        Sprawdza, czy istnieje możliwość poprawy użyteczności zgodnie z równowagą Nasha.
        """
        current_resource = self.get_current_resource(task_index)
        if current_resource == resource_index:
            return False

        # Obliczanie użyteczności przed i po potencjalnej realokacji
        current_utility = self.calculate_utility(task_index, current_resource)
        new_utility = self.calculate_utility(task_index, resource_index)

        # Sprawdzanie, czy nowa użyteczność jest większa niż obecna
        return new_utility > current_utility

    def calculate_utility(self, task_index, resource_index):
        if resource_index is not None and task_index < len(self.processing_times) and resource_index < len(
                self.cost_matrix[task_index]):
            return 1 / (self.processing_times[task_index] * self.cost_matrix[task_index][resource_index])
        else:
            return 0  # lub inna obsługa błędu

    def get_current_resource(self, task_index):
        for j in range(self.num_resources):
            if self.allocation_matrix[task_index][j] == 1:
                return j
        return -1  # Zwróć -1, jeśli nie znaleziono zasobu

    def calculate_utility_after_reallocation(self, task_index, resource_index):
        """
        Przykładowa implementacja obliczania użyteczności po realokacji.
        Zwraca przykładową wartość - należy dostosować do specyfiki problemu.
        """
        # Przykład: użyteczność po realokacji - załóżmy, że jest to zawsze lepsza
        return self.cost_matrix[task_index][resource_index] - 1

    def calculate_total_utility(self):
        """
         Oblicza całkowitą użyteczność aktualnego rozwiązania.
         Przykładowa implementacja - powinna być dostosowana do specyfiki problemu.
         """
        total_utility = 0
        for i in range(self.num_tasks):
            for j in range(self.num_resources):
                if self.allocation_matrix[i][j] == 1:
                    total_utility += self.calculate_utility(i, j)
        return total_utility

    def calculate_total_utility_after_reallocation(self, task_index, resource_index):
        """
        Oblicza całkowitą użyteczność po realokacji zadania.
        Przykładowa implementacja - powinna być dostosowana do specyfiki problemu.
        """
        # Symulacja realokacji zadania
        original_resource = self.get_current_resource(task_index)
        self.allocation_matrix[task_index][original_resource] = 0
        self.allocation_matrix[task_index][resource_index] = 1

        # Obliczenie całkowitej użyteczności po realokacji
        total_utility = self.calculate_total_utility()

        # Przywrócenie oryginalnej alokacji
        self.allocation_matrix[task_index][resource_index] = 0
        self.allocation_matrix[task_index][original_resource] = 1
        return total_utility

    def print_allocation_matrix(self, title="Macierz Alokacji"):
        """
        Wyświetla aktualny stan macierzy alokacji.
        """
        print(title)
        for row in self.allocation_matrix:
            print(' '.join(map(str, row)))
        print()

    def update_resource_parameters(self, num_tasks, num_resources):
        self.num_tasks = num_tasks
        self.num_resources = num_resources
        self.processing_times = [random.randint(1, 10) for _ in range(num_tasks)]
        self.cost_matrix = [[random.randint(1, 10) for _ in range(num_resources)] for _ in range(num_tasks)]
        self.allocation_matrix = [[0 for _ in range(num_resources)] for _ in range(num_tasks)]

    def run(self):
        # Uruchomienie początkowej optymalizacji
        self.initial_optimization()
        # Wyświetla macierz alokacji przed optymalizacją
        self.print_allocation_matrix("Macierz Alokacji Początkowej:")
        self.minimize_splr()
        self.minimize_gelr()
        self.evolutionary_optimization()
        # Wyświetla macierz alokacji po optymalizacji
        self.print_allocation_matrix("Macierz Alokacji Końcowej:")
        # Próba znalezienia równowagi Nasha
        if self.find_nash_equilibrium():
            print("Znaleziono równowagę Nasha.")
        else:
            print("Nie znaleziono równowagi Nasha.")

    # Metoda do Punktu 1: Porównanie Metod Inicjalizacji
    def compare_initialization_methods(self, methods):
        results = {}
        for method in methods:
            self.initialize_with_method(method)
            results[method] = self.collect_performance_data()
        return results

    def initialize_with_method(self, method):
        if method == "random":
            print("Inicjalizacja losowa")
            # Inicjalizacja losowa: Przypisanie zadań do zasobów w sposób losowy
            for i in range(self.num_tasks):
                random_resource = random.randint(0, self.num_resources - 1)
                self.allocation_matrix[i][random_resource] = 1

        elif method == "greedy":
            print("Inicjalizacja zachłanna")
            # Inicjalizacja zachłanna: Wybieranie zadań i zasobów w sposób zachłanny
            tasks_remaining = list(range(self.num_tasks))
            resources_remaining = list(range(self.num_resources))

            while tasks_remaining:
                task = random.choice(tasks_remaining)
                resource = random.choice(resources_remaining)

                self.allocation_matrix[task][resource] = 1

                # Usunięcie wybranego zadania i zasobu z dostępnych
                tasks_remaining.remove(task)
                resources_remaining.remove(resource)

    # Metoda do Punktu 2: Porównanie wyników dla różnej liczby zadań i zasobów
    def compare_different_configurations(self):
        results = {}
        for tasks in range(5, 11):
            for resources in range(5, 11):
                self.update_resource_parameters(tasks, resources)
                self.run()
                results[(tasks, resources)] = self.collect_performance_data()
        return results

    # Metoda do Punktu 3: Sposób na zbieranie danych
    def collect_performance_data(self):
        total_time = sum(self.processing_times)
        average_time = total_time / self.num_tasks
        user_satisfaction = self.calculate_user_satisfaction()
        return {"total_time": total_time, "average_time": average_time, "user_satisfaction": user_satisfaction}

    def calculate_user_satisfaction(self):
        return random.uniform(0, 1)  # Zwraca losową wartość zadowolenia użytkownika

    # Metoda do Punktu 4: Weryfikacja Wyników
    def verify_results(self, theoretical_results):
        actual_results = self.collect_performance_data()
        return actual_results == theoretical_results

    # Metoda do Punktu 5: Ocena wydajności w warunkach dynamicznych
    def dynamic_performance_evaluation(self, changes):
        for _ in range(changes):
            self.num_tasks = random.randint(5, 10)
            self.num_resources = random.randint(5, 10)
            self.run()
            print(self.collect_performance_data())

    def save_results_to_csv(self, results, filename='results.csv'):
        with open(filename, 'w', newline='') as csv_file:
            fieldnames = ["Tasks", "Resources", "Total Time", "Average Time", "User Satisfaction"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            # Write the header row
            writer.writeheader()

            # Write the data rows
            for (tasks, resources), data in results.items():
                writer.writerow({
                    "Tasks": tasks,
                    "Resources": resources,
                    "Total Time": data["total_time"],
                    "Average Time": data["average_time"],
                    "User Satisfaction": data["user_satisfaction"]
                })

    def plot_performance_data(self, data, filename):
        total_times = [data[key]["total_time"] for key in data]
        average_times = [data[key]["average_time"] for key in data]
        user_satisfactions = [data[key]["user_satisfaction"] for key in data]

        plt.figure(figsize=(12, 4))

        plt.subplot(131)
        plt.plot(total_times)
        plt.title("Total Times")

        plt.subplot(132)
        plt.plot(average_times)
        plt.title("Average Times")

        plt.subplot(133)
        plt.plot(user_satisfactions)
        plt.title("User Satisfaction")

        plt.tight_layout()
        plt.savefig('results.jpg')

    def run_and_save_results(self):
        results = {}
        for tasks in range(5, 11):
            for resources in range(5, 11):
                self.update_resource_parameters(tasks, resources)
                self.run()
                results[(tasks, resources)] = self.collect_performance_data()

        self.save_results_to_csv(results)
        self.plot_performance_data(results, "performance_plots.png")

if __name__ == '__main__':
    num_tasks = 5
    num_resources = 5
    cost_matrix = [[random.randint(1, 10) for _ in range(num_resources)] for _ in range(num_tasks)]
    processing_times = [random.randint(1, 10) for _ in range(num_tasks)]

    allocation_system = CloudResourceAllocation(num_tasks, num_resources, cost_matrix, processing_times)
    allocation_system.run_and_save_results()

    # Wywołanie metod dla poszczególnych punktów
    print("Porównanie metod inicjalizacji:")
    print(allocation_system.compare_initialization_methods(["random", "greedy"]))

    print("Porównanie różnych konfiguracji:")
    print(allocation_system.compare_different_configurations())

    print("Zbieranie danych o wydajności:")
    print(allocation_system.collect_performance_data())

    print("Weryfikacja wyników:")
    theoretical_results = {"total_time": 50, "average_time": 10, "user_satisfaction": 0.5}
    print(allocation_system.verify_results(theoretical_results))

    print("Ocena wydajności w warunkach dynamicznych:")
    allocation_system.dynamic_performance_evaluation(5)
