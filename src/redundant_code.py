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
    C = np.matmul(A, B)
    if np.random.random() <= f:
        time.sleep(np.random.exponential(lamda2))
    else:
        time.sleep(np.random.exponential(lamda1))
    with open(f"{i}_redundant.pkl", "wb") as outfile:
        pickle.dump(C, outfile)

def redundant_matmul(A, B, s, r, t, r1, r2, N, f=0.5, lamda1=2, lamda2=10, avg_over=5, verbose=False):
    req_threshold = r1 * r2
    total_time = 0
    total_err = 0

    for i in range(N):
        if os.path.exists(f"{i}_redundant.pkl"):
            os.remove(f"{i}_redundant.pkl")

    for iter_no in range(avg_over):
        start = time.time()

        finished = np.zeros([r1 * r2], dtype = bool)
        processes = [None] * N
        
        for i in range(N):
            i1 = i % (r1 * r2)
            j = i1 // r2
            k = i1 % r2
            processes[i] = [i, multiprocessing.Process(target = matrix_multiplication, args = (A[j*(r//r1):(j+1)*(r//r1),:], B[:,k*(t//r2):(k+1)*(t//r2)], i, f, lamda1, lamda2))]
            processes[i][1].start()
        
        unique_results = 0
        workers_finished = 0
        workers_remaining = N
        file_ids = [None] * req_threshold

        while unique_results < req_threshold:
            i = 0
            while i < workers_remaining and unique_results < req_threshold:
                if not processes[i][1].is_alive():
                    j = processes[i][0] % (r1 * r2)
                    if not finished[j]:
                        file_ids[unique_results] = processes[i][0]
                        unique_results += 1
                        finished[j] = True
                    workers_remaining -= 1
                    workers_finished += 1
                    tmp = processes[i]
                    processes[i] = processes[workers_remaining]
                    processes[workers_remaining] = tmp
                else:
                    i += 1
        
        C = np.zeros([r, t])

        for file_id in file_ids:
            with open(f"{file_id}_redundant.pkl", "rb") as infile:
                mat = pickle.load(infile)
            i = file_id
            i = i % (r1 * r2)
            if finished[i]:
                finished[i] = False
                j = i // r2
                k = i % r2
                C[j*(r//r1):(j+1)*(r//r1),k*(t//r2):(k+1)*(t//r2)] = mat

        end = time.time()

        for proc in processes:
            while proc[1].is_alive():
                proc[1].kill()
            proc[1].close()

        for i in range(N):
            if os.path.exists(f"{i}_redundant.pkl"):
                os.remove(f"{i}_redundant.pkl")

        iter_err = np.linalg.norm((A @ B) - C, ord = 'fro') / np.linalg.norm((A @ B), ord = 'fro')
        iter_time = end - start

        if verbose:
            print(f"Iteration {iter_no}: Error = {iter_err}, Time taken = {iter_time}")
        
        total_err += iter_err
        total_time += iter_time
    
    avg_err = total_err / avg_over
    avg_time = total_time / avg_over

    return avg_err, avg_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("s", help = "Second Dimension of A and First Dimension of B", type = int)
    parser.add_argument("r", help = "Second Dimension of A", type = int)
    parser.add_argument("t", help = "Second Dimension of B", type = int)
    parser.add_argument("r1", help = "Number of divisions of A", type = int)
    parser.add_argument("r2", help = "Number of divisions of B", type = int)
    parser.add_argument("N", help = "Number of workers", type = int)
    parser.add_argument("-f", help = "Fraction of slow nodes", type = float, default = 0.5)
    parser.add_argument("-lamda1", help = "Average delay of a fast node", type = float, default = 2)
    parser.add_argument("-lamda2", help = "Average delay of a slow node", type = float, default = 10)
    parser.add_argument("-avg", help = "Number of runs to average over", type = int, default = 5)
    args = parser.parse_args()
    s, r, t, r1, r2, N, f, lamda1, lamda2, avg_over = args.s, args.r, args.t, args.r1, args.r2, args.N, args.f, args.lamda1, args.lamda2, args.avg
    if s <= 0 or r <= 0 or t <= 0 or r1 <= 0 or r2 <= 0 or N <= 0 or lamda1 <= 0 or lamda2 <= 0 or avg_over <= 0:
        print("All the values must be greater than 0!")
        exit()
    if r % r1 != 0 or t % r2 != 0:
        print("Submatrices must be of same size!")
        exit()
    if N % (r1 * r2) != 0:
        print("Number of workers must be a multiple of r1 and r2")
        N = (int)(r1 * r2 * np.ceil(N / (r1 * r2)))
    
    A = np.random.random([r, s])
    B = np.random.random([s, t])

    avg_err, avg_time = redundant_matmul(A, B, s, r, t, r1, r2, N, f, lamda1, lamda2, avg_over, True)

    print("Average error:", avg_err, "Average time:", avg_time)
    
if __name__ == "__main__":
    main()