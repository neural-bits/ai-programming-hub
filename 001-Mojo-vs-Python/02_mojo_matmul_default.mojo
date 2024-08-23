import benchmark
from memory import memset_zero
from random import rand, random_float64

## Exact MatMul Implementation but in Mojo Code ##
def matmul_untyped(C, A, B):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]

## Methods to create and manipulate a matrix ##
fn matrix_getitem(self: object, i: object) raises -> object:
    return self.value[i]


fn matrix_setitem(self: object, i: object, value: object) raises -> object:
    self.value[i] = value
    return None


fn matrix_append(self: object, value: object) raises -> object:
    self.value.append(value)
    return None


fn matrix_init(rows: Int, cols: Int) raises -> object:
    var value = object([])
    return object(
        Attr("value", value), Attr("__getitem__", matrix_getitem), Attr("__setitem__", matrix_setitem),
        Attr("rows", rows), Attr("cols", cols), Attr("append", matrix_append),
    )

## Benchmarking the MatMul ##
def benchmark_matmul_untyped(M: Int, N: Int, K: Int):
    C = matrix_init(M, N)
    A = matrix_init(M, K)
    B = matrix_init(K, N)
    for i in range(M):
        c_row = object([])
        b_row = object([])
        a_row = object([])
        for j in range(N):
            c_row.append(0.0)
            b_row.append(random_float64(-5, 5))
            a_row.append(random_float64(-5, 5))
        C.append(c_row)
        B.append(b_row)
        A.append(a_row)

    @parameter
    fn test_fn():
        try:
            _ = matmul_untyped(C, A, B)
        except:
            pass

    var secs = benchmark.run[test_fn](max_runtime_secs=0.5).mean()
    _ = (A, B, C)
    var gflops = ((2*M*N*K)/secs) / 1e9
    print("[MatMul] Mojo:",gflops % 10e10, "GFLOP/s")

fn main():
    try:
        benchmark_matmul_untyped(128, 128, 128)
    except:
        pass