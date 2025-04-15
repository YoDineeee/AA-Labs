import time
import random
import matplotlib.pyplot as plt



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
    for i in range(n-1, 0, -1):
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
    for i in range(n-1, -1, -1):
        num = arr[i]
        index = (num // exp) % 10
        output[count[index] - 1] = num
        count[index] -= 1
    for i in range(n):
        arr[i] = output[i]



def time_sorting_algorithm(algorithm, arr):
    start_time = time.time()
    algorithm(arr.copy())  
    end_time = time.time()
    return end_time - start_time

def plot_performance():
    sizes = [100, 1000, 5000, 10000, 20000]  
    algorithms = {
        "Merge Sort": merge_sort,
        "Quick Sort": quick_sort,
        "Heapsort": heapsort,
        "Radix Sort": radix_sort
    }
    
    # Generate a large random array once (to avoid bias)
    max_size = max(sizes)
    test_array = [random.randint(0, 10000) for _ in range(max_size)]
    
    results = {name: [] for name in algorithms}
    
    for size in sizes:
        arr = test_array[:size]  # Slice to get subset
        for name, algorithm in algorithms.items():
            time_taken = time_sorting_algorithm(algorithm, arr)
            results[name].append(time_taken)
            print(f"{name} (n={size}): {time_taken:.4f} sec")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for name, timings in results.items():
        plt.plot(sizes, timings, marker='o', label=name)
    
    plt.xlabel("Input Size (n)")
    plt.ylabel("Time (seconds)")
    plt.title("Sorting Algorithm Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_performance()