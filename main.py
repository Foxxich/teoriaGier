import pulp
import random


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

            if current_resource is None:
                continue  # Pomiń, jeśli zadanie nie ma przypisanego zasobu

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
                if resource != current_resource: # Sprawdzanie, czy zasób jest różny od obecnego
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
        current_resource = self.get_current_resource(task_index)
        if current_resource is None:
            # Obsługa sytuacji, gdy żaden zasób nie jest przypisany
            return float('inf')

        current_utility = self.calculate_utility(task_index, current_resource)
        new_utility = self.calculate_utility(task_index, resource_index)
        return current_utility - new_utility

    def get_current_resource(self, task_index):
        """
        Zwraca indeks obecnego zasobu przypisanego do zadania.
        """
        for j in range(self.num_resources):
            if self.allocation_matrix[task_index][j] == 1:
                return j
        return None

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
                if self.min_single(min_gelr_task, resource) != -1:
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
        best_resource = -1  # Use -1 to indicate no better resource found
        for r in range(self.num_resources):
            if r != resource:
                current_gelr = self.compute_gelr(task, r)
                if current_gelr < min_gelr:
                    min_gelr = current_gelr
                    best_resource = r
        return best_resource

    def find_nash_equilibrium(self):
        """
        Szukanie równowagi Nasha.
        """
        for task in range(self.num_tasks):
            for resource in range(self.num_resources):
                if self.can_improve_nash(task, resource):
                    return False
        return True

    def can_improve_nash(self, task_index, resource_index):
        """
        Sprawdza, czy istnieje możliwość poprawy użyteczności zgodnie z równowagą Nasha.
        """
        current_resource = self.get_current_resource(task_index)
        if current_resource is None or current_resource == resource_index:
            return False

        # Obliczanie użyteczności przed i po potencjalnej realokacji
        current_utility = self.calculate_utility(task_index, current_resource)
        new_utility = self.calculate_utility(task_index, resource_index)

        # Sprawdzanie, czy nowa użyteczność jest większa niż obecna
        return new_utility > current_utility

    def calculate_utility(self, task_index, resource_index):
        """
        Oblicza użyteczność zadania w kontekście alokacji do danego zasobu.
        """
        # Załóżmy, że użyteczność jest odwrotnie proporcjonalna do czasu przetwarzania
        # i jest związana z kosztem alokacji
        if self.allocation_matrix[task_index][resource_index] == 1:
            return 1 / (self.processing_times[task_index] * self.cost_matrix[task_index][resource_index])
        else:
            return 0

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

    def brute_force_optimization(self):
        best_cost = float('inf')
        best_allocation = None

        for allocation in itertools.product(range(self.num_resources), repeat=self.num_tasks):
            total_cost = 0
            for i, resource in enumerate(allocation):
                total_cost += self.processing_times[i] * self.cost_matrix[i][resource]
            if total_cost < best_cost:
                best_cost = total_cost
                best_allocation = allocation

        # Aktualizacja macierzy alokacji
        self.allocation_matrix = [[0 for _ in range(self.num_resources)] for _ in range(self.num_tasks)]
        for task, resource in enumerate(best_allocation):
            self.allocation_matrix[task][resource] = 1

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
        # Sprawdzenie, czy osiągnięto równowagę Nasha
        if self.find_nash_equilibrium():
            print("Znaleziono równowagę Nasha.")
        else:
            print("Nie znaleziono równowagi Nasha.")

    def random_initialization(self):
        for i in range(self.num_tasks):
            # Losowe przypisanie zadania do zasobu
            resource = random.randint(0, self.num_resources - 1)
            self.allocation_matrix[i][resource] = 1

def compare_different_number_of_tasks(max_tasks, num_resources):
    for num_tasks in range(1, max_tasks + 1):
        allocation_system = CloudResourceAllocation(
            num_tasks, num_resources, generate_cost_matrix(num_tasks, num_resources),
            generate_processing_times(num_tasks))
        allocation_system.run()
        # Tu dodaj kod do analizy i zapisu wyników

def compare_different_number_of_resources(num_tasks, max_resources):
    for num_resources in range(1, max_resources + 1):
        allocation_system = CloudResourceAllocation(
            num_tasks, num_resources, generate_cost_matrix(num_tasks, num_resources),
            generate_processing_times(num_tasks))
        allocation_system.run()
        # Tu dodaj kod do analizy i zapisu wyników

import itertools

def generate_cost_matrix(num_tasks, num_resources):
    return [[random.randint(1, 10) for _ in range(num_resources)] for _ in range(num_tasks)]

def generate_processing_times(num_tasks):
    return [random.randint(1, 10) for _ in range(num_tasks)]


if __name__ == '__main__':
    if __name__ == '__main__':
        # Przykładowe dane wejściowe
        num_tasks = 5
        num_resources = 5
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

        # Przeprowadzenie eksperymentów z różną liczbą zadań i zasobów
        print("Eksperymenty z różną liczbą zadań:")
        compare_different_number_of_tasks(10, num_resources)

        print("Eksperymenty z różną liczbą zasobów:")
        compare_different_number_of_resources(num_tasks, 10)

        # Podejście Brute Force dla małej instancji
        print("Podejście Brute Force dla małej instancji:")
        small_system = CloudResourceAllocation(3, 3, generate_cost_matrix(3, 3), generate_processing_times(3))
        small_system.brute_force_optimization()
        small_system.print_allocation_matrix("Macierz Alokacji po Brute Force:")
