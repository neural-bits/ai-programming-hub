
import os
import benchmark
from algorithm import vectorize
from memory import memset_zero
from random import rand, random_float64
from algorithm import parallelize

alias type = DType.float32
alias M = 1024
alias N = 1024
alias K = 1024

struct Matrix[rows: Int, cols: Int]:
    var data: DTypePointer[type]

    # Initialize zeroeing all values
    fn __init__(inout self):
        self.data = DTypePointer[type].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, data: DTypePointer[type]):
        self.data = data

    # Initialize with random values
    @staticmethod
    fn rand() -> Self:
        var data = DTypePointer[type].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(data)

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: Scalar[type]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
        return self.data.store[width=nelts](y * self.cols + x, val)

@always_inline
fn bench[
    func: fn (Matrix, Matrix, Matrix) -> None]() -> Float64:
    var C = Matrix[M, N]()
    var A = Matrix[M, K].rand()
    var B = Matrix[K, N].rand()

    @always_inline
    @parameter
    fn test_fn():
        _ = func(C, A, B)

    var secs = benchmark.run[test_fn](max_runtime_secs=1).mean()

    A.data.free()
    B.data.free()
    C.data.free()

    var gflops = ((2 * M * N * K) / secs) / 1e9

    return gflops


## Vectorized-0 : Setting the static width of the SIMD vector (single instruction multiple device) ##
alias nelts = simdwidthof[DType.float32]() * 2
fn matmul_vectorized_0(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for nv in range(0, C.cols - nelts + 1, nelts):
                C.store(m, nv, C.load[nelts](m, nv) + A[m, k] * B.load[nelts](k, nv))

            # Handle remaining elements with scalars.
            for n in range(nelts * (C.cols // nelts), C.cols):
                C[m, n] += A[m, k] * B[k, n]

## Vectorized-1 : Parallelizing over the M dimension, making it multi-threaded##
fn matmul_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):
            @parameter
            fn dot[nelts : Int](n : Int):
                C.store[nelts](m,n, C.load[nelts](m,n) + A[m,k] * B.load[nelts](k,n))
            vectorize[dot, nelts, size = C.cols]()
    parallelize[calc_row](C.rows, C.rows)


fn main():
    var v0 = bench[matmul_vectorized_0]()
    var v1 = bench[matmul_parallelized]()
    
    print("[MatMul][Advanced][SIMD] Mojo: ", v0, "GFLOP/s")
    print("[MatMul][Advanced][Parallel] Mojo: ", v1, "GFLOP/s")

