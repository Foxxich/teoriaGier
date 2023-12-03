import pulp
import random


class CloudResourceAllocation:
    def __init__(self, num_tasks, num_resources, cost_matrix):
        """
        :param num_tasks: określa rozmiar problemu
        :param num_resources:  określa rozmiar problemu
        :param cost_matrix: macierz kosztów alokacji zadań do zasobów
        """
        self.num_tasks = num_tasks
        self.num_resources = num_resources
        self.cost_matrix = cost_matrix
        self.allocation_matrix = None

    def initial_optimization(self):
        """
        Używa programowania liniowego całkowitoliczbowego (Binary Integer Programming) do znajdowania początkowego
        rozwiązania
        """
        model = pulp.LpProblem("Initial_Resource_Allocation", pulp.LpMinimize)

        # Zmienne decyzyjne dla każdej pary zadanie-zasób
        allocation_vars = pulp.LpVariable.dicts("Allocation",
                                                ((i, j) for i in range(self.num_tasks)
                                                 for j in range(self.num_resources)),
                                                cat='Binary')

        # Funkcja celu: minimalizacja całkowitego kosztu
        model += pulp.lpSum([self.cost_matrix[i][j] * allocation_vars[i, j]
                             for i in range(self.num_tasks)
                             for j in range(self.num_resources)])

        # Ograniczenia: każde zadanie musi być przypisane do dokładnie jednego zasobu
        for i in range(self.num_tasks):
            model += pulp.lpSum([allocation_vars[i, j] for j in range(self.num_resources)]) == 1

        # Ograniczenia: każdy zasób może być przypisany do co najwyżej jednego zadania
        for j in range(self.num_resources):
            model += pulp.lpSum([allocation_vars[i, j] for i in range(self.num_tasks)]) <= 1

        # Rozwiązanie problemu
        model.solve()

        # Aktualizacja macierzy alokacji
        self.allocation_matrix = [[allocation_vars[i, j].varValue for j in range(self.num_resources)]
                                  for i in range(self.num_tasks)]

        # Wypisanie wyniku (opcjonalne)
        for i in range(self.num_tasks):
            for j in range(self.num_resources):
                if allocation_vars[i, j].varValue == 1:
                    print(f"Zadanie {i} jest przypisane do zasobu {j}")

    def evolutionary_optimization(self):
        """
        Algorytm 3: Optymalizacja Ewolucyjna
        Założenie: Prosty mechanizm ewolucyjny, który losowo realokuje zadania
        """
        for _ in range(100):  # Liczba iteracji
            task = random.randint(0, self.num_tasks - 1)
            resource = random.randint(0, self.num_resources - 1)

            if self.can_improve_solution(task, resource):
                self.perform_reallocation(task, resource)

    def can_improve_solution(self, task_index, resource_index):
        """
        Prosta heurystyka sprawdzająca możliwość poprawy
        """
        return self.calculate_utility(task_index, resource_index) < self.calculate_utility_after_reallocation(
            task_index, resource_index)

    def perform_reallocation(self, task_index, resource_index):
        """
        Realokuje zadanie do nowego zasobu
        """
        for j in range(self.num_resources):
            self.allocation_matrix[task_index][j] = 0
        self.allocation_matrix[task_index][resource_index] = 1

    def compute_splr(self, task_index, resource_index):
        """
        Prosta heurystyka obliczająca SPELR
        """
        return abs(
            self.calculate_utility(task_index, resource_index) -
            self.calculate_utility_after_reallocation(task_index, resource_index))

    def compute_gelr(self, task_index, resource_index):
        """
        Prosta heurystyka obliczająca GELR
        """
        return abs(self.calculate_total_utility() - self.calculate_total_utility_after_reallocation(task_index,
                                                                                                    resource_index))

    def minimize_splr(self):
        """
        Algorytm 1: Minimizacja SPELR
        """
        for task in range(self.num_tasks):
            for resource in range(self.num_resources):
                # Obliczanie SPELR dla każdej pary zadanie-zasób
                splr = self.compute_splr(task, resource)
                # Jeśli znajdziesz możliwość poprawy, wykonaj realokację
                if splr < 0:  # Przykładowe kryterium
                    self.perform_reallocation(task, resource)

    def minimize_gelr(self):
        """
        Algorytm 2: Minimizacja GELR
        """
        for resource in range(self.num_resources):
            # Znajdź zadanie, które minimalizuje GELR po realokacji tego zasobu
            min_gelr = float('inf')
            task_to_reallocate = None
            for task in range(self.num_tasks):
                gelr = self.compute_gelr(task, resource)
                if gelr < min_gelr:
                    min_gelr = gelr
                    task_to_reallocate = task
            # Wykonaj realokację dla znalezionego zadania
            if task_to_reallocate is not None:
                self.perform_reallocation(task_to_reallocate, resource)

    def find_nash_equilibrium(self):
        for task in range(self.num_tasks):
            for resource in range(self.num_resources):
                if self.can_improve_nash(task, resource):
                    return False
        return True

    def can_improve_nash(self, task_index, resource_index):
        """
        Prosta heurystyka sprawdzająca możliwość poprawy w kontekście Nasha
        """
        current_utility = self.calculate_utility(task_index, resource_index)
        new_utility = self.calculate_utility_after_reallocation(task_index, resource_index)
        return new_utility > current_utility

    def calculate_utility(self, task_index, resource_index):
        """
        Przykładowa implementacja obliczania użyteczności zadania.
        Zwraca przykładową wartość - należy dostosować do specyfiki problemu.
        """
        # Przykład: użyteczność może być związana z kosztem alokacji
        return self.cost_matrix[task_index][resource_index]

    def calculate_utility_after_reallocation(self, task_index, resource_index):
        """
        Przykładowa implementacja obliczania użyteczności po realokacji.
        Zwraca przykładową wartość - należy dostosować do specyfiki problemu.
        """
        # Przykład: użyteczność po realokacji - załóżmy, że jest to zawsze lepsza
        return self.cost_matrix[task_index][resource_index] - 1  # Przykładowe zmniejszenie kosztu

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
        original_resource = -1
        for j in range(self.num_resources):
            if self.allocation_matrix[task_index][j] == 1:
                original_resource = j
                break

        self.allocation_matrix[task_index][original_resource] = 0
        self.allocation_matrix[task_index][resource_index] = 1

        # Obliczenie całkowitej użyteczności po realokacji
        total_utility = self.calculate_total_utility()

        # Przywrócenie oryginalnej alokacji
        self.allocation_matrix[task_index][resource_index] = 0
        self.allocation_matrix[task_index][original_resource] = 1

        return total_utility

    def run(self):
        # Uruchomienie początkowej optymalizacji
        self.initial_optimization()

        # Uruchomienie algorytmu minimalizacji SPELR
        self.minimize_splr()

        # Uruchomienie algorytmu minimalizacji GELR
        self.minimize_gelr()

        # Uruchomienie optymalizacji ewolucyjnej
        self.evolutionary_optimization()

        # Sprawdzenie, czy osiągnięto równowagę Nasha
        if self.find_nash_equilibrium():
            print("Znaleziono równowagę Nasha.")
        else:
            print("Nie znaleziono równowagi Nasha.")


if __name__ == '__main__':
    # Przykładowe dane wejściowe
    num_tasks = 5
    num_resources = 5
    cost_matrix = [[random.randint(1, 10) for _ in range(num_resources)] for _ in range(num_tasks)]

    # Utworzenie instancji klasy i uruchomienie procesu optymalizacji
    allocation_system = CloudResourceAllocation(num_tasks, num_resources, cost_matrix)
    allocation_system.run()
