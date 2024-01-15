import random


def generate_cost_matrix(num_tasks, num_resources):
    """
    Generuje macierz kosztów dla zadań i zasobów.

    Parametry:
    num_tasks (int): Liczba zadań do przypisania.
    num_resources (int): Liczba dostępnych zasobów.

    Zwraca:
    list[list[int]]: Macierz kosztów, gdzie każdy element macierzy reprezentuje
                     koszt przypisania danego zadania do danego zasobu.
                     Wymiary macierzy to num_tasks x num_resources.
    """
    return [[random.randint(1, 10) for _ in range(num_resources)] for _ in range(num_tasks)]


def generate_processing_times(num_tasks):
    """
    Generuje losowe czasy przetwarzania dla każdego zadania.

    Parametry:
    num_tasks (int): Liczba zadań.

    Zwraca:
    list[int]: Lista czasów przetwarzania dla każdego zadania.
               Długość listy równa jest liczbie zadań (num_tasks).
    """
    return [random.randint(1, 10) for _ in range(num_tasks)]