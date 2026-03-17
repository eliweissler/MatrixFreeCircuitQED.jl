using Pkg
Pkg.activate(".")
include("basic_integration_test.jl")
using Plots
using LinearAlgebra

# Fixed physics parameters for the 3 coupled oscillators
w1, w2, w3 = 1.0, 1.1, 1.2
g12, g23 = 0.2, 0.2

# We will sweep the local dimension d (number of energy levels per mode)
d_range = 5:5:50
times_direct_single = Float64[]
times_mf_single     = Float64[]

for d in d_range
    println("Timing d = $d (Total Hilbert Space: $(d^3))...")

    # Extract dense local operators for the current dimension d
    a = destroy(d)
    n_mat = Array((a'*a).data)
    x_base = Array((a+a').data)
    x_mat = polynomial_coupling_operator(x_base, d)

    # Build direct Hamiltonian (Sparse by default in QuantumToolbox)
    H_direct = build_direct_hamiltonian(d, d, d, w1, w2, w3, g12, g23; coupling_order=d)

    # Build Matrix-Free Hamiltonian
    H_mf_arr = MatrixFree3Oscillators(d, d, d, w1, w2, w3, g12, g23, n_mat, n_mat, n_mat, x_mat, x_mat, x_mat)
    H_mf     = QuantumObject(H_mf_arr, type=Operator(), dims=(d, d, d))

    # Use a dense random state so sparse-state structure does not skew results.
    ψ_dense_vec = normalize(randn(ComplexF64, d^3))
    y_direct = similar(ψ_dense_vec)
    y_mf = similar(ψ_dense_vec)

    # Single mat-vec timing at array level (no QuantumObject wrapper overhead).
    t_direct_single, _ = timed_runs(() -> mul!(y_direct, H_direct.data, ψ_dense_vec); trials=5)
    t_mf_single, _ = timed_runs(() -> mul!(y_mf, H_mf_arr, ψ_dense_vec); trials=5)

    push!(times_direct_single, t_direct_single * 1e3)  # convert to ms
    push!(times_mf_single, t_mf_single * 1e3)
end

d_vals = collect(d_range)
p = plot(d_vals, times_direct_single;
    label     = "Direct | single mat-vec (dense state)",
    marker    = :circle,
    yaxis     = :log10,
    xlabel    = "Local Dimension (d)",
    ylabel    = "Time (ms)",
    title     = "Single Mat-Vec Timing vs d (3 Oscillators)",
    legend    = :topleft,
)
plot!(p, d_vals, times_mf_single; label = "Matrix-free | single mat-vec (dense state)", marker = :square)

savefig(p, "mat_vec_product_timing_3osc.png")
println("Saved mat_vec_product_timing_3osc.png")
display(p)