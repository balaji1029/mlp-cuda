import subprocess
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Compile the CUDA code
    # subprocess.run(["nvcc", "-o", "mlp", "mlp.cu"], check=True)

    batched = []
    N_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    
    # Run with command line arguments from 32 to 32k in powers of 2
    # for N in N_sizes:
    #     print(f"Running with N={N}")
    #     subprocess.run(["./a.out", str(N)], check=True)
    #     output = subprocess.check_output(["./a.out", str(N)])
    #     # print(output.decode())
    #     elapsedTime = float(output.decode().strip())
    #     print(f"Elapsed time for N={N}: {elapsedTime} ms")
    #     batched.append(elapsedTime)

    # Running with N=32
    # 5.01146
    # Elapsed time for N=32: 2.98202 ms
    # Running with N=64
    # 3.1703
    # Elapsed time for N=64: 3.38221 ms
    # Running with N=128
    # 3.92909
    # Elapsed time for N=128: 8.28614 ms
    # Running with N=256
    # 6.47475
    # Elapsed time for N=256: 3.86765 ms
    # Running with N=512
    # 5.42413
    # Elapsed time for N=512: 5.5296 ms
    # Running with N=1024
    # 11.3531
    # Elapsed time for N=1024: 13.9993 ms
    # Running with N=2048
    # 33.7715
    # Elapsed time for N=2048: 33.3957 ms
    # Running with N=4096
    # 121.431
    # Elapsed time for N=4096: 115.688 ms
    # Running with N=8192
    # 976.844
    # Elapsed time for N=8192: 993.196 ms
    # Running with N=16384
    # 3.34336
    # Elapsed time for N=16384: 3.21024 ms
    # Running with N=32768
    # 4.79453
    # Elapsed time for N=32768: 4.4137 ms

    # Running with N=32
    # 221.295
    # Elapsed time for N=32: 200.587 ms
    # Running with N=64
    # 206.02
    # Elapsed time for N=64: 218.729 ms
    # Running with N=128
    # 209.157
    # Elapsed time for N=128: 203.09 ms
    # Running with N=256
    # 207.72
    # Elapsed time for N=256: 209.687 ms
    # Running with N=512
    # 211.85
    # Elapsed time for N=512: 202.253 ms
    # Running with N=1024
    # 213.955
    # Elapsed time for N=1024: 207.529 ms
    # Running with N=2048
    # 219.193
    # Elapsed time for N=2048: 208.704 ms
    # Running with N=4096
    # 213.324
    # Elapsed time for N=4096: 210.013 ms
    # Running with N=8192
    # 219.804
    # Elapsed time for N=8192: 220.587 ms
    # Running with N=16384
    # 225.462
    # Elapsed time for N=16384: 230.059 ms
    # Running with N=32768
    # 231.121
    # Elapsed time for N=32768: 219.851 ms

    batched = [2.98202, 3.38221, 8.28614, 3.86765, 5.5296, 13.9993, 33.3957, 115.688, 993.196, 3210.24, 4413.7]
    streamed = [200.587, 218.729, 203.09, 209.687, 202.253, 207.529, 208.704, 210.013, 220.587, 230.059, 219.851]
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(N_sizes, batched, marker='o')
    plt.plot(N_sizes, streamed, marker='o')
    plt.legend(['Batched', 'Streamed'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('N (log scale)')
    plt.ylabel('Elapsed Time (ms, log scale)')
    plt.title('Elapsed Time vs N for MLP CUDA Implementation')
    plt.grid(True, which="both", ls="--")
    plt.savefig('mlp_performance.png')
    plt.show()
