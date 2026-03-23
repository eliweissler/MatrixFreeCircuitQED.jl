using Pkg
Pkg.activate(".")
using QuantumToolbox
using LinearAlgebra

# Timing utilities
function timed_runs(f::Function; trials::Int=3)
    # Compilation time is not included in timings.
    f()
    times = [@elapsed f() for _ in 1:trials]
    return minimum(times), sum(times) / length(times)
end

function print_timing(label::AbstractString, min_t::Float64, avg_t::Float64)
    println(label, " | min: ", round(min_t * 1e3, digits=3), " ms, avg: ", round(avg_t * 1e3, digits=3), " ms")
end

function phase_aligned_relative_error(a, b)
    ov = dot(a, b)
    b_aligned = abs(ov) == 0 ? b : b * (ov / abs(ov))
    return norm(a - b_aligned) / max(norm(a), eps(Float64))
end

# Helpers for building the coupling terms
function polynomial_coupling_operator(x::AbstractMatrix, order::Int)
    C = zeros(eltype(x), size(x))
    term = Matrix(x)
    for _ in 1:order
        C .+= term
        term = term * x
    end
    return C
end
function polynomial_coupling_operator_qobj(x, order::Int)
    C = zero(x)
    term = x
    for _ in 1:order
        C += term
        term = term * x
    end
    return C
end

# Define the Custom Matrix-Free Array Type for 3 Oscillators
mutable struct MatrixFree3Oscillators{TC, TW} <: AbstractMatrix{TC}
    d1::Int
    d2::Int
    d3::Int
    w1::TW
    w2::TW
    w3::TW
    g12::TW
    g23::TW
    n1::Matrix{TC} # Number operator for mode 1
    n2::Matrix{TC}
    n3::Matrix{TC}
    x1::Matrix{TC} # Position operator for mode 1
    x2::Matrix{TC}
    x3::Matrix{TC}
    thread_strategy::Symbol
    X_buf::Array{TC, 3}
    Y_buf::Array{TC, 3}
end

function MatrixFree3Oscillators(
    d1::Int,
    d2::Int,
    d3::Int,
    w1,
    w2,
    w3,
    g12,
    g23,
    n1::AbstractMatrix{TC},
    n2::AbstractMatrix{TC},
    n3::AbstractMatrix{TC},
    x1::AbstractMatrix{TC},
    x2::AbstractMatrix{TC},
    x3::AbstractMatrix{TC},
    ;
    thread_strategy::Symbol=:kj,
) where {TC}
    TW = promote_type(typeof(w1), typeof(w2), typeof(w3), typeof(g12), typeof(g23))
    if thread_strategy != :k && thread_strategy != :kj
        throw(ArgumentError("thread_strategy must be :k or :kj"))
    end
    X_buf = zeros(TC, d3, d2, d1)
    Y_buf = zeros(TC, d3, d2, d1)

    return MatrixFree3Oscillators{TC, TW}(
        d1,
        d2,
        d3,
        convert(TW, w1),
        convert(TW, w2),
        convert(TW, w3),
        convert(TW, g12),
        convert(TW, g23),
        Matrix(n1),
        Matrix(n2),
        Matrix(n3),
        Matrix(x1),
        Matrix(x2),
        Matrix(x3),
        thread_strategy,
        X_buf,
        Y_buf,
    )
end

# Satisfy the AbstractArray interface -- size and getindex
Base.size(H::MatrixFree3Oscillators) = (H.d1 * H.d2 * H.d3, H.d1 * H.d2 * H.d3)
function Base.getindex(H::MatrixFree3Oscillators, I::Int, J::Int)
    dims = (H.d3, H.d2, H.d1)
    ci_I = CartesianIndices(dims)[I]
    ci_J = CartesianIndices(dims)[J]

    i1, i2, i3 = ci_I[3], ci_I[2], ci_I[1]
    j1, j2, j3 = ci_J[3], ci_J[2], ci_J[1]

    val = zero(eltype(H))

    # --- Evaluate local terms ---
    if i2 == j2 && i3 == j3
        val += H.w1 * H.n1[i1, j1]
    end
    if i1 == j1 && i3 == j3
        val += H.w2 * H.n2[i2, j2]
    end
    if i1 == j1 && i2 == j2
        val += H.w3 * H.n3[i3, j3]
    end

    # --- Evaluate coupling terms ---
    if i3 == j3
        val += H.g12 * H.x1[i1, j1] * H.x2[i2, j2]
    end
    if i1 == j1
        val += H.g23 * H.x2[i2, j2] * H.x3[i3, j3]
    end

    return val
end

# Implement a low-overhead matrix-free mat-vec kernel
@inline function _site_action(H::MatrixFree3Oscillators{TC}, X::Array{TC, 3}, k::Int, j::Int, i::Int) where {TC}
    val = zero(TC)

    # Local term on mode 1
    for a in 1:H.d1
        val += H.w1 * H.n1[i, a] * X[k, j, a]
    end

    # Local term on mode 2
    for b in 1:H.d2
        val += H.w2 * H.n2[j, b] * X[k, b, i]
    end

    # Local term on mode 3
    for c in 1:H.d3
        val += H.w3 * H.n3[k, c] * X[c, j, i]
    end

    # Coupling x1 x2
    for a in 1:H.d1
        x1ia = H.x1[i, a]
        for b in 1:H.d2
            val += H.g12 * x1ia * H.x2[j, b] * X[k, b, a]
        end
    end

    # Coupling x2 x3
    for b in 1:H.d2
        x2jb = H.x2[j, b]
        for c in 1:H.d3
            val += H.g23 * x2jb * H.x3[k, c] * X[c, b, i]
        end
    end

    return val
end

function _mul_thread_k!(Y::Array{TC, 3}, H::MatrixFree3Oscillators{TC}, X::Array{TC, 3}) where {TC}
    Threads.@threads for k in 1:H.d3
        @inbounds for j in 1:H.d2
            for i in 1:H.d1
                Y[k, j, i] = _site_action(H, X, k, j, i)
            end
        end
    end
end

function _mul_thread_kj!(Y::Array{TC, 3}, H::MatrixFree3Oscillators{TC}, X::Array{TC, 3}) where {TC}
    Threads.@threads for kj in 1:(H.d3 * H.d2)
        k = ((kj - 1) ÷ H.d2) + 1
        j = ((kj - 1) % H.d2) + 1
        @inbounds for i in 1:H.d1
            Y[k, j, i] = _site_action(H, X, k, j, i)
        end
    end
end

function LinearAlgebra.mul!(y::AbstractVector, H::MatrixFree3Oscillators, x::AbstractVector)
    # Use preallocated buffers stored on the operator to avoid scratch allocation.
    copyto!(H.X_buf, reshape(x, H.d3, H.d2, H.d1))
    X = H.X_buf
    Y = H.Y_buf

    fill!(Y, zero(eltype(y)))

    if H.thread_strategy == :k
        _mul_thread_k!(Y, H, X)
    else
        _mul_thread_kj!(Y, H, X)
    end

    copyto!(y, reshape(Y, :))

    return y
end
Base.:*(H::MatrixFree3Oscillators, x::AbstractVector) = mul!(similar(x), H, x)

# Direct Approach (Explicit Tensor Products - Sparse by default in QuantumToolbox)
function build_direct_hamiltonian(d1, d2, d3, w1, w2, w3, g12, g23; coupling_order::Int=min(d1, d2, d3))
    # Create standard bosonic operators
    a1, a2, a3 = destroy(d1), destroy(d2), destroy(d3)
    n1, n2, n3 = a1'*a1, a2'*a2, a3'*a3
    x1, x2, x3 = a1+a1', a2+a2', a3+a3'

    c1 = polynomial_coupling_operator_qobj(x1, coupling_order)
    c2 = polynomial_coupling_operator_qobj(x2, coupling_order)
    c3 = polynomial_coupling_operator_qobj(x3, coupling_order)
    
    I1, I2, I3 = qeye(d1), qeye(d2), qeye(d3)

    H = w1 * tensor(n1, I2, I3) +
        w2 * tensor(I1, n2, I3) +
        w3 * tensor(I1, I2, n3) +
        g12 * tensor(c1, c2, I3) +
        g23 * tensor(I1, c2, c3)

    return H
end

if abspath(PROGRAM_FILE) == @__FILE__

    # Setup the system parameters
    d1, d2, d3 = 5, 5, 5 
    w1, w2, w3 = 1.0, 1.1, 1.2
    g12, g23 = 0.2, 0.2
    coupling_order = d1
    thread_strategy = :kj  # toggle: :k (outer only) or :kj (first two dimensions)

    # Build direct Hamiltonian
    H_direct_op = build_direct_hamiltonian(d1, d2, d3, w1, w2, w3, g12, g23; coupling_order=coupling_order)


    # Build Matrix-Free Hamiltonian
    a1, a2, a3 = destroy(d1), destroy(d2), destroy(d3)
    n1_mat, n2_mat, n3_mat = Array((a1'*a1).data), Array((a2'*a2).data), Array((a3'*a3).data)
    c1 = polynomial_coupling_operator_qobj(a1 + a1', coupling_order)
    c2 = polynomial_coupling_operator_qobj(a2 + a2', coupling_order)
    c3 = polynomial_coupling_operator_qobj(a3 + a3', coupling_order)
    x1_mat = Array(c1.data)
    x2_mat = Array(c2.data)
    x3_mat = Array(c3.data)
    H_mf_array = MatrixFree3Oscillators(d1, d2, d3, w1, w2, w3, g12, g23, n1_mat, n2_mat, n3_mat, x1_mat, x2_mat, x3_mat; thread_strategy=thread_strategy)
    H_mf = QuantumObject(H_mf_array, type=Operator(), dims=(d1, d2, d3))

    # --- TEST 1: getindex Verification ---
    dense_diff = norm(H_direct_op.data - [H_mf_array[i,j] for i in 1:(d1*d2*d3), j in 1:(d1*d2*d3)])
    dense_rel = dense_diff / max(norm(H_direct_op.data), eps(Float64))
    println("getindex matches dense matrix (rel<1e-12): ", dense_rel < 1e-12, " [rel=", dense_rel, "]")

    # --- TEST 2: Operator Action ---
    ψ0 = normalize(tensor(fock(d1, 0), fock(d2, 1), fock(d3, 0)) + 0.5 * tensor(fock(d1, 1), fock(d2, 0), fock(d3, 1)))
    out_direct = H_direct_op * ψ0
    out_mf = H_mf * ψ0
    action_diff = norm(out_direct - out_mf)
    action_rel = action_diff / max(norm(out_direct), eps(Float64))
    println("Matrix-free action matches direct product (rel<1e-12): ", action_rel < 1e-12, " [rel=", action_rel, "]")

    
    # --- TEST 3: Time Evolution ---
    tlist = range(0, 5, length=50)
    sol_direct = sesolve(H_direct_op, ψ0, tlist)
    sol_mf = sesolve(H_mf, ψ0, tlist)
    final_abs = norm(sol_direct.states[end] - sol_mf.states[end])
    final_rel = phase_aligned_relative_error(sol_direct.states[end], sol_mf.states[end])
    max_rel = maximum(phase_aligned_relative_error(sd, sm) for (sd, sm) in zip(sol_direct.states, sol_mf.states))
    te_tol = 1e-9
    println("Time evolution match (max phase-aligned rel<", te_tol, "): ", max_rel < te_tol,
        " [final_abs=", final_abs, ", final_rel=", final_rel, ", max_rel=", max_rel, "]")

    println("\n--- Timing: sesolve ---")
    min_direct_se, avg_direct_se = timed_runs(() -> sesolve(H_direct_op, ψ0, tlist); trials=3)
    min_mf_se, avg_mf_se = timed_runs(() -> sesolve(H_mf, ψ0, tlist); trials=3)
    print_timing("sesolve (direct H)", min_direct_se, avg_direct_se)
    print_timing("sesolve (matrix-free H)", min_mf_se, avg_mf_se)

    # --- TEST 4: Diagonalization (KrylovKit) ---
    println("\n--- Timing: Ground-State Solve ---")
    min_direct_eig, avg_direct_eig = timed_runs(() -> eigsolve(H_direct_op, k=1, which=:SR, ishermitian=true); trials=3)
    min_mf_eig, avg_mf_eig = timed_runs(() -> eigsolve(H_mf, k=1, which=:SR, ishermitian=true); trials=3)
    print_timing("eigsolve (sparse H)", min_direct_eig, avg_direct_eig)
    print_timing("eigsolve (matrix-free H)", min_mf_eig, avg_mf_eig)

end