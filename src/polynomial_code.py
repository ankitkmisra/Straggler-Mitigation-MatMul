import multiprocessing
import numpy as np
from scipy.interpolate import lagrange
import argparse
import os
import time
import pickle
import warnings

warnings.simplefilter('ignore', np.RankWarning)

def matrix_multiplication(A, B, i, f, lamda1, lamda2):
    C = np.matmul(np.transpose(A), B)
    if np.random.random() <= f:
        time.sleep(np.random.exponential(lamda2))
    else:
        time.sleep(np.random.exponential(lamda1))
    with open(f"{i}_polynomial.pkl", "wb") as outfile:
        pickle.dump(C, outfile)

def polynomial_matmul(A, B, s, r, t, p, m, n, N, eval_points_type=None, interpolation_method=None, f=0.5, lamda1=2, lamda2=10, avg_over=5, alpha=None, beta=None, theta=None, verbose=False):
    req_threshold = p * m * n + p - 1
    total_time = 0
    total_pre_time = 0
    total_err = 0

    if alpha is None:
        alpha = 1
    if beta is None:
        beta = p
    if theta is None:
        theta = p*m

    for i in range(N):
        if os.path.exists(f"{i}_polynomial.pkl"):
            os.remove(f"{i}_polynomial.pkl")

    for iter_no in range(avg_over):
        start = time.time()

        start2 = time.time()

        A_j = np.tile(np.transpose(np.array([np.linspace(0, p - 1, num = p, dtype = int)])), [1, m])
        A_k = np.tile(np.array([np.linspace(0, m - 1, num = m, dtype = int)]), [p, 1])
        A_mult = (alpha * A_j) + (beta * A_k)
        mult_A = np.zeros([p, m, s // p, r // m])
        for i in range(p):
            for j in range(m):
                mult_A[i, j, :, :] = A_mult[i, j]

        B_j = np.tile(np.transpose(np.array([np.linspace(0, p - 1, num = p, dtype = int)])), [1, n])
        B_k = np.tile(np.array([np.linspace(0, n - 1, num = n, dtype = int)]), [p, 1])
        B_mult = ((p - 1 - B_j) * alpha) + (theta * B_k)
        mult_B = np.zeros([p, n, s // p, t // n])
        for i in range(p):
            for j in range(n):
                mult_B[i, j, :, :] = B_mult[i, j]

        block_A = np.zeros([p, m, s // p, r // m])
        block_B = np.zeros([p, n, s // p, t // n])

        for i in range(p):
            for j in range(m):
                block_A[i,j] = A[i*(s//p):(i+1)*(s//p),j*(r//m):(j+1)*(r//m)]
            for j in range(n):
                block_B[i,j] = B[i*(s//p):(i+1)*(s//p),j*(t//n):(j+1)*(t//n)]

        X = None
        if eval_points_type == 2:
            X = np.arange(start = 1, stop = N + 1, dtype = int) / N
            # Starts failing at large N since separation tends to 0 and now small errors actually cause an issue.
        elif eval_points_type == 3:
            polynomial_evaluation_delta = 4 * np.arctan(1) / N
            X = np.linspace(start = polynomial_evaluation_delta, stop = N * polynomial_evaluation_delta, num = N)
            X = np.exp(X * 1j)
        elif eval_points_type == 4:
            X = np.arange(start = 1, stop = N + 1, dtype = int)
        else:
            X = np.cos((2 * np.arange(start = 1, stop = N + 1) - 1) * np.pi/(2*N))
        
        processes = [None] * N
        As = [None] * N
        Bs = [None] * N
        for i in range(N):
            x = X[i]
            As[i] = np.sum(block_A * np.resize(x**mult_A, new_shape = block_A.shape), axis = (0, 1))
            Bs[i] = np.sum(block_B * np.resize(x**mult_B, new_shape = block_B.shape), axis = (0, 1))

        end2 = time.time()

        for i in range(N):
            processes[i] = [i, multiprocessing.Process(target = matrix_multiplication, args = (As[i], Bs[i], i, f, lamda1, lamda2))]
            processes[i][1].start()
        
        finished = 0
        finished_ids = [None] * req_threshold
        workers_remaining = N
        while finished < req_threshold:
            i = 0
            while i < workers_remaining and finished < req_threshold:
                if not processes[i][1].is_alive():
                    workers_remaining -= 1
                    finished_ids[finished] = processes[i][0]
                    finished += 1
                    tmp = processes[i]
                    processes[i] = processes[workers_remaining]
                    processes[workers_remaining] = tmp
                else:
                    i += 1
        
        outputs = [None] * req_threshold
        for i in range(req_threshold):
            with open(f"{finished_ids[i]}_polynomial.pkl", "rb") as infile:
                outputs[i] = [pickle.load(infile), finished_ids[i]]
        
        output_X = np.array([X[q[1]] for q in outputs])
        output_C = np.array([q[0] for q in outputs])
        C = np.zeros([r, t])
        h = np.array([np.reshape(output_C[i, :, :], [(r // m) * (t // n)]) for i in range(req_threshold)])

        power = np.arange(req_threshold)
        raised_X = output_X[None, :]**power[:, None]

        coefficients = None
        if interpolation_method == 2:
            coefficients = np.linalg.solve(np.transpose(raised_X), h)
            coefficients = np.reshape(np.transpose(coefficients), [r // m, t // n, req_threshold])
        elif interpolation_method == 3:
            coefficients = lagrange(output_X, h).coef
        else:
            coefficients = np.flip(np.transpose(np.polyfit(output_X, h, deg = p * m * n + p  - 2, rcond = 2e-50)), 1)
            coefficients = np.reshape(coefficients, [r // m, t // n, req_threshold])

        i = np.repeat(np.arange(r), t)
        j = np.tile(np.arange(t), r)
        k_1 = i // (r // m)
        k_2 = j // (t // n)
        i_2 = i % (r // m)
        j_2 = j % (t // n)
        C[i,j] = coefficients[i_2, j_2, p - 1 + (k_1 * p) + (k_2 * p * m)]

        end = time.time()

        for proc in processes:
            while proc[1].is_alive():
                proc[1].kill()
            proc[1].close()

        for i in range(N):
            if os.path.exists(f"{i}_polynomial.pkl"):
                os.remove(f"{i}_polynomial.pkl")

        # Relative Frobenius norm of (actual multiplication - decoded multiplication)
        iter_err = np.linalg.norm((np.transpose(A) @ B) - C, ord = 'fro') / np.linalg.norm((np.transpose(A) @ B), ord = 'fro')
        iter_time = end - start
        iter_pre_time = end2 - start2

        if verbose:
            print(f"Iteration {iter_no}: Error = {iter_err}, Time = {iter_time}, Preprocessing Time = {iter_pre_time}")

        total_err += iter_err
        total_time += iter_time
        total_pre_time += iter_pre_time

    avg_err = total_err / avg_over
    avg_time = total_time / avg_over
    avg_pre_time = total_pre_time / avg_over

    return avg_err, avg_time, avg_pre_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("s", help = "First Dimension of A and B", type = int)
    parser.add_argument("r", help = "Second Dimension of A", type = int)
    parser.add_argument("t", help = "Second Dimension of B", type = int)
    parser.add_argument("p", help = "First Dimension of submatrices", type = int)
    parser.add_argument("m", help = "Second Dimension of submatrices of A", type = int)
    parser.add_argument("n", help = "Second Dimension of submatrices of B", type = int)
    parser.add_argument("N", help = "Number of workers", type = int)
    parser.add_argument("-f", help = "Fraction of nodes which take a long time", type = float, default = 0.5)
    parser.add_argument("-lamda1", help = "Average delay of a fast node", type = float, default = 2)
    parser.add_argument("-lamda2", help = "Average delay of a slow node", type = float, default = 10)
    parser.add_argument("-avg", help = "Number of runs", type = int, default = 5)
    args = parser.parse_args()
    s, r, t, p, m, n, N, f, lamda1, lamda2, avg_over = args.s, args.r, args.t, args.p, args.m, args.n, args.N, args.f, args.lamda1, args.lamda2, args.avg
    if s <= 0 or r <= 0 or t <= 0 or p <= 0 or m <= 0 or n <= 0 or N <= 0 or lamda1 <= 0 or lamda2 <= 0 or f <= 0 or avg_over <= 0:
        print("All the values must be greater than 0!")
        exit()
    if s % p != 0 or r % m != 0 or t % n != 0:
        print("Submatrices must be of the same size!")
        exit()
    alpha = 1
    beta = p
    theta = p * m
    if N < (p * m * n + p - 1):
        N = p * m * n + p - 1
    
    A = np.random.random([s, r])
    B = np.random.random([s, t])

    avg_err, avg_time, avg_pre_time = polynomial_matmul(A, B, s, r, t, p, m, n, N, 1, 1, f, lamda1, lamda2, avg_over, alpha, beta, theta, verbose=True)

    print("Average error:", avg_err, "Average time:", avg_time, "Average preprocessing time:", avg_pre_time)
    
if __name__ == "__main__":
    main()