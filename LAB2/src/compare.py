import time
import random
import matplotlib.pyplot as plt

# ---------------------------- Sorting Algorithms ---------------------------- #

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)

def _merge(left, right):
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def heapsort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        _heapify(arr, i, 0)
    return arr

def _heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        _heapify(arr, n, largest)

def radix_sort(arr):
    if not arr:
        return arr
    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        _counting_sort(arr, exp)
        exp *= 10
    return arr

def _counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    for num in arr:
        index = (num // exp) % 10
        count[index] += 1
    for i in range(1, 10):
        count[i] += count[i-1]
    for i in range(n - 1, -1, -1):
        num = arr[i]
        index = (num // exp) % 10
        output[count[index] - 1] = num
        count[index] -= 1
    for i in range(n):
        arr[i] = output[i]

# ------------------------- Timing and Data Generation ------------------------ #

def time_sorting_algorithm(algorithm, arr):
    start_time = time.time()
    algorithm(arr.copy())
    end_time = time.time()
    return (end_time - start_time) * 1000  # Convert to milliseconds

def generate_sorted_array(size):
    return list(range(size))

def generate_reverse_sorted_array(size):
    return list(range(size, 0, -1))

def generate_nearly_sorted_array(size, swaps=10):
    arr = list(range(size))
    for _ in range(swaps):
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

def generate_random_array(size, lower=0, upper=10000):
    return [random.randint(lower, upper) for _ in range(size)]

# ----------------------- Displaying Results & Plotting ----------------------- #

def print_results_table(results, algorithms, sizes, distributions):
    alg_names = list(algorithms.keys())
    header = f"{'Size':<8}| {'Distribution':<15}" + "".join([f"| {name:<12}" for name in alg_names])
    print(header)
    print("-" * len(header))
    
    for size in sizes:
        for dist in distributions:
            row = f"{size:<8}| {dist:<15}"
            for alg_name in alg_names:
                time_taken = results[dist][size][alg_name]
                row += f"| {time_taken:>10.3f} ms"
            print(row)
        print()

def plot_results(results, sizes, distributions, algorithms):
    alg_names = list(algorithms.keys())
    num_dists = len(distributions)
    
    # Create a subplot for each distribution (2 rows x 2 cols layout if 4 distributions)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()
    
    for i, dist in enumerate(distributions):
        ax = axs[i]
        for alg_name in alg_names:
            # Gather timings for each size for the current distribution
            y_values = [results[dist][size][alg_name] for size in sizes]
            ax.plot(sizes, y_values, marker="o", label=alg_name)
        ax.set_title(f"{dist} Array")
        ax.set_xlabel("Input Size (n)")
        ax.set_ylabel("Time (ms)")
        ax.grid(True)
        ax.legend()
    
    fig.suptitle("Sorting Algorithm Performance Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def run_experiments():
    # Define array sizes and distributions to test
    sizes = [100, 1000, 5000, 10000]
    distributions = ["Sorted", "ReverseSorted", "NearlySorted", "Random"]
    
    # Define the sorting algorithms to test
    algorithms = {
        "Merge Sort": merge_sort,
        "Quick Sort": quick_sort,
        "Heapsort": heapsort,
        "Radix Sort": radix_sort
    }
    
    # Dictionary to hold results: results[distribution][size][algorithm] = time (ms)
    results = {dist: {sz: {} for sz in sizes} for dist in distributions}
    
    # Generate arrays and time each algorithm for each array type and size
    for size in sizes:
        sorted_arr = generate_sorted_array(size)
        reverse_sorted_arr = generate_reverse_sorted_array(size)
        nearly_sorted_arr = generate_nearly_sorted_array(size)
        random_arr = generate_random_array(size)
        
        for alg_name, alg_func in algorithms.items():
            results["Sorted"][size][alg_name] = time_sorting_algorithm(alg_func, sorted_arr)
            results["ReverseSorted"][size][alg_name] = time_sorting_algorithm(alg_func, reverse_sorted_arr)
            results["NearlySorted"][size][alg_name] = time_sorting_algorithm(alg_func, nearly_sorted_arr)
            results["Random"][size][alg_name] = time_sorting_algorithm(alg_func, random_arr)
    
    # Print results in a table-like format
    print_results_table(results, algorithms, sizes, distributions)
    
    # Plot the results for visual comparison
    plot_results(results, sizes, distributions, algorithms)

if __name__ == "__main__":
    run_experiments()
