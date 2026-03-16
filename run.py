import subprocess
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Compile the CUDA code
    subprocess.run(["nvcc", "-o", "mlp", "mlp.cu"], check=True)

    batched = []
    N_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    
    # Run with command line arguments from 32 to 32k in powers of 2
    for N in N_sizes:
        print(f"Running with N={N}")
        subprocess.run(["./mlp", str(N)], check=True)
        output = subprocess.check_output(["./mlp", str(N)])
        # print(output.decode())
        elapsedTime = float(output.decode().strip())
        print(f"Elapsed time for N={N}: {elapsedTime} ms")
        batched.append(elapsedTime)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(N_sizes, batched, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('N (log scale)')
    plt.ylabel('Elapsed Time (ms, log scale)')
    plt.title('Elapsed Time vs N for MLP CUDA Implementation')
    plt.grid(True, which="both", ls="--")
    plt.savefig('mlp_performance.png')
    plt.show()
