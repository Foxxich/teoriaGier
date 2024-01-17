import pulp
import random
import itertools

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

    def random_initialization(self):
        for i in range(self.num_tasks):
            # Losowe przypisanie zadania do zasobu
            resource = random.randint(0, self.num_resources - 1)
            self.allocation_matrix[i][resource] = 1

    def evolutionary_optimization(self):
        i = 0  # Rozpocznij od pierwszego zadania (indeks 0 w indeksacji od zera)
        flag = True
        while flag:  # Kontynuuj, dopóki nie zostaną dokonane żadne zmiany w pełnym przebiegu
            flag = False  # Zresetuj flagę dla bieżącego przebiegu
            while True:
                ms = self.obtain_multiplexing_resource_vector(i)  # Pobierz wektor zasobów dla zadania
                for j in ms:
                    q = self.MinGlobal(i, j)  # Znajdź nową alokację zasobów
                    if q != -1:  # Jeśli nowa alokacja jest możliwa
                        p = self.MinSingle(i, j)  # Znajdź optymalną pojedynczą alokację
                        if p != -1 and p != q:  # Jeśli znaleziono lepszą alokację i jest ona różna od obecnej
                            self.execute_reallocation(i, j, p)  # Dokonaj realokacji
                            flag = True  # Ustaw flagę na True, ponieważ dokonano zmiany
                            print(f"Zadanie {i + 1} - zasób {j} realokacja do {p} wykonana.")

                if i == self.num_tasks - 1:  # Sprawdź, czy osiągnięto ostatnie zadanie
                    if not flag:
                        break  # Żadne realokacje nie zostały wykonane w tym przebiegu, więc przerwij pętlę
                    else:
                        i = 0  # Zresetuj indeks zadania na 0, aby rozpocząć nowy przebieg
                else:
                    i += 1  # Przejdź do następnego zadania

        print("Optymalizacja ewolucyjna zakończona.")

    def tij(self, resource_j):
        # Ta funkcja powinna zwracać pewną metrykę związaną z zadaniem i zasobem j
        # Jako przykład, zwracam losową wartość. Należy ją zastąpić rzeczywistą logiką.
        return random.random()

    def obtain_multiplexing_resource_vector(self, task_i):
        # Zakładając, że każde zadanie może być przypisane tylko do jednego zasobu na raz
        current_resource = self.get_current_resource(task_i)
        return [j for j in range(self.num_resources) if j != current_resource]

    def MinGlobal(self, task_i, resource_j):
        # Znajdź zasób minimalizujący globalny koszt dla zadania task_i, wykluczając bieżący zasób resource_j
        min_cost = float('inf')
        min_resource_index = -1
        for r in range(self.num_resources):
            if r != resource_j:
                cost = self.calculate_global_cost(task_i, r)  # Należy zdefiniować tę metodę na podstawie problemu
                if cost < min_cost:
                    min_cost = cost
                    min_resource_index = r
        return min_resource_index if min_cost < self.calculate_global_cost(task_i, resource_j) else -1

    def MinSingle(self, task_i, resource_j):
        # Znajdź zasób minimalizujący pojedynczy koszt zadania task_i, wykluczając bieżący zasób resource_j
        min_cost = float('inf')
        min_resource_index = -1
        for r in range(self.num_resources):
            if r != resource_j:
                cost = self.calculate_single_task_cost(task_i, r)  # Należy zdefiniować tę metodę na podstawie problemu
                if cost < min_cost:
                    min_cost = cost
                    min_resource_index = r
        return min_resource_index if min_cost < self.calculate_single_task_cost(task_i, resource_j) else -1

    def calculate_single_task_cost(self, task_i, resource_j):
        """
        Oblicz koszt przypisania zadania task_i do zasobu resource_j.
        Przykład używa prostego modelu kosztów opartego na czasach przetwarzania i macierzy kosztów.
        """
        # Zakładamy, że processing_times to lista, gdzie processing_times[i] to czas przetwarzania zadania i
        # oraz cost_matrix to macierz, gdzie cost_matrix[i][j] to czynnik kosztu przypisania zadania i do zasobu j
        task_cost = self.processing_times[task_i] * self.cost_matrix[task_i][resource_j]
        return task_cost

    def calculate_global_cost(self, task_i, resource_j):
        """
        Oblicz globalny koszt przypisania zadania task_i do zasobu resource_j, biorąc pod uwagę wszystkie zadania i zasoby.
        Przykład zakłada, że cost_matrix to macierz kosztów alokacji, gdzie cost_matrix[i][j] to koszt
        przypisania zadania i do zasobu j.
        """
        if task_i < 0 or task_i >= self.num_tasks or resource_j < 0 or resource_j >= self.num_resources:
            raise ValueError("Indeks task_i lub resource_j poza zakresem")

        current_global_cost = 0
        for t in range(self.num_tasks):
            if t != task_i:
                current_resource = self.get_current_resource(t)
                if current_resource is not None:
                    current_global_cost += self.cost_matrix[t][current_resource]
                else:
                    pass

        # Dodaj koszt przypisania zadania task_i do zasobu resource_j
        new_global_cost = current_global_cost + self.cost_matrix[task_i][resource_j]

        return new_global_cost

    def execute_reallocation(self, task_i, resource_j, resource_p):
        """
        Realokuje zadanie (task_i) z obecnego zasobu (resource_j)
        do nowego zasobu (resource_p).
        """
        if task_i < 0 or task_i >= self.num_tasks or resource_p < 0 or resource_p >= self.num_resources:
            return  # Either task_i or resource_p is out of range, so do nothing

        current_resource = self.get_current_resource(task_i)
        if current_resource is not None and current_resource != resource_p:
            self.perform_reallocation(task_i, current_resource, resource_p)

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
        if task_index < 0 or task_index >= self.num_tasks:
            return None  # task_index is out of range

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

    def calculate_total_cost(self):
        """
        Oblicza całkowity koszt alokacji zadań do zasobów.
        """
        total_cost = 0
        for i in range(self.num_tasks):
            for j in range(self.num_resources):
                if self.allocation_matrix[i][j] == 1:
                    total_cost += self.processing_times[i] * self.cost_matrix[i][j]
        return total_cost


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