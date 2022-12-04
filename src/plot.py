import numpy as np
from matplotlib import pyplot as plt
from polynomial_code import polynomial_matmul
from redundant_code import redundant_matmul

def generate_error_vs_threshold_plot():
    r = 32
    s = 32
    t = 32
    errors_1, errors_2 = [], []
    n_vals = np.array([1, 2, 2, 4, 4, 8])
    m_vals = np.array([1, 1, 2, 2, 4, 4])
    nm_vals = np.multiply(n_vals, m_vals)
    for i in range(6):
        A = np.random.random([s, r])
        B = np.random.random([s, t])
        avg_err_1, _, _ = polynomial_matmul(A, B, s, r, t, 1, m_vals[i], n_vals[i], nm_vals[i], None, 1, 1, 1, 1, 1)
        errors_1.append(avg_err_1)
        avg_err_2, _, _ = polynomial_matmul(A, B, s, r, t, 1, m_vals[i], n_vals[i], nm_vals[i], 4, 1, 1, 1, 1, 1)
        errors_2.append(avg_err_2)
        print(f"Threshold = {nm_vals[i]}, Error = {avg_err_1}, {avg_err_2}")
    plt.figure()
    plt.plot(nm_vals, errors_1, '--bo', label='Chebyshev nodes')
    plt.plot(nm_vals, errors_2, '--go', label='Equally spaced nodes 1 to N')
    plt.yscale('log')
    plt.xlabel("Threshold = pmn + p - 1")
    plt.ylabel("Relative Error (logarithmic scale)")
    plt.title("Error with entangled polynomial code for two 32x32 matrices")
    plt.legend()
    plt.savefig('error_vs_threshold.png', bbox_inches='tight')

def generate_time_vs_matsize_plot():
    matsizes = 512 * np.array([1, 2, 4, 8])
    polynomial_times = []
    polynomial_pre_times = []
    redundant_times = []
    for matsize in matsizes:
        s = matsize
        r = matsize
        t = matsize
        A = np.random.random([s, r])
        B = np.random.random([s, t])
        _, avg_time_1, avg_pre_time_1 = polynomial_matmul(A, B, s, r, t, 1, 4, 4, 32, avg_over=2)
        polynomial_times.append(avg_time_1)
        polynomial_pre_times.append(avg_pre_time_1)
        _, avg_time_2 = redundant_matmul(A, B, s, r, t, 4, 4, 32, avg_over=1)
        redundant_times.append(avg_time_2)
        print(f"Matrix dimension = {matsize}, Polynomial code time = {avg_time_1}, Redundant time: {avg_time_2}, Preproc time: {avg_pre_time_1}, Time except preproc: {avg_time_1 - avg_pre_time_1}")
    plt.figure()
    plt.plot(matsizes, polynomial_times, '--bo', label='Entangled polynomial code (total)')
    plt.plot(matsizes, polynomial_pre_times, '--ro', label='Entangled polynomial code (preproc)')
    plt.plot(matsizes, np.array(polynomial_times) - np.array(polynomial_pre_times), '--ko', label='Entangled polynomial code (except preproc)')
    plt.plot(matsizes, redundant_times, '--go', label='Redundant code')
    plt.xlabel("Matrix dimension")
    plt.ylabel("Time taken (seconds)")
    plt.title("Time taken for 32 workers")
    plt.legend()
    plt.savefig('time_vs_matsize.png', bbox_inches='tight')

# def generate_time_vs__plot():
#     matsizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
#     polynomial_times = []
#     redundant_times = []
#     for matsize in matsizes:
#         r = matsize
#         s = matsize
#         t = matsize
#         A = np.random.random([s, r])
#         B = np.random.random([s, t])
#         _, avg_time_1 = polynomial_matmul(A, B, s, r, t, 1, 4, 4, 32, avg_over=3)
#         polynomial_times.append(avg_time_1)
#         _, avg_time_2 = redundant_matmul(A, B, s, r, t, 4, 4, 32, avg_over=3)
#         redundant_times.append(avg_time_2)
#         print(f"Matrix dimension = {matsize}, Polynomial code time = {avg_time_1}, Redundant time: {avg_time_2}")
#     plt.figure()
#     plt.plot(matsizes, polynomial_times, '--bo')
#     plt.plot(matsizes, redundant_times, '--go')
#     plt.xlabel("Matrix dimension")
#     plt.ylabel("Time taken (seconds)")
#     plt.title("Time taken for 32 workers")
#     plt.legend()
#     plt.savefig('time_vs_matsize.png', bbox_inches='tight')

def main():
    #generate_error_vs_threshold_plot()
    generate_time_vs_matsize_plot()

if __name__ == "__main__":
    main()