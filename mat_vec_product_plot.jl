using Pkg
Pkg.activate(".")
include("basic_integration_test.jl")
using Plots
using LinearAlgebra
using Statistics

function timing_mean_std(f::Function; trials::Int=10)
    # Exclude compilation from the timing statistics.
    f()
    samples = [@elapsed f() for _ in 1:trials]
    mean_t = mean(samples)
    std_t = length(samples) > 1 ? std(samples) : 0.0
    return mean_t, std_t
end

# Parameters for the 3 coupled oscillators
w1, w2, w3 = 1.0, 1.1, 1.2
g12, g23 = 0.2, 0.2

# Sweep the number of levels per mode
d_range = 5:5:90
times_direct_single = Float64[]
times_mf_single     = Float64[]
std_direct_single = Float64[]
std_mf_single     = Float64[]

for d in d_range
    println("Timing d = $d (Total Hilbert Space: $(d^3))...")
    
    # Build direct Hamiltonian (Sparse by default in QuantumToolbox)
    H_direct = build_direct_hamiltonian(d, d, d, w1, w2, w3, g12, g23; coupling_order=d)

    # Build Matrix-Free Hamiltonian
    a = destroy(d)
    n_mat = Array((a'*a).data)
    x_base = Array((a+a').data)
    x_mat = polynomial_coupling_operator(x_base, d)
    H_mf_arr = MatrixFree3Oscillators(d, d, d, w1, w2, w3, g12, g23, n_mat, n_mat, n_mat, x_mat, x_mat, x_mat)
    H_mf     = QuantumObject(H_mf_arr, type=Operator(), dims=(d, d, d))

    # Use a random state so sparse-state structure does not skew results.
    ψ_dense_vec = normalize(randn(ComplexF64, d^3))
    y_direct = similar(ψ_dense_vec)
    y_mf = similar(ψ_dense_vec)

    # Single mat-vec timing at array level (no QuantumObject wrapper overhead).
    t_direct_single, std_direct = timing_mean_std(() -> mul!(y_direct, H_direct.data, ψ_dense_vec); trials=10)
    t_mf_single, std_mf = timing_mean_std(() -> mul!(y_mf, H_mf_arr, ψ_dense_vec); trials=10)

    push!(times_direct_single, t_direct_single * 1e3)  # convert to ms
    push!(times_mf_single, t_mf_single * 1e3)
    push!(std_direct_single, std_direct * 1e3)
    push!(std_mf_single, std_mf * 1e3)
end

d_vals = collect(d_range)
speedup =  times_direct_single ./ times_mf_single
all_times = vcat(times_direct_single, times_mf_single)
emin = floor(Int, log10(minimum(all_times)))
emax = ceil(Int, log10(maximum(all_times)))
decade_ticks = 10.0 .^ (emin:emax)

p = plot(d_vals, times_direct_single;
    label     = "direct",
    marker    = :circle,
    yerror    = std_direct_single,
    yaxis     = :log10,
    yticks    = decade_ticks,
    xlabel    = "Local Dimension (d)",
    ylabel    = "Time (ms)",
    title     = "Single Mat-Vec for 3 oscillators ($(Threads.nthreads()) threads)",
    legend    = :topleft,
    size      = (900, 600),
    right_margin = 14Plots.mm,
)
plot!(p, d_vals, times_mf_single;
    label = "matrix-free",
    marker = :square,
    yerror = std_mf_single,
)

# Right axis: linear speedup ratio.
p_right = twinx(p)
plot!(p_right, d_vals, speedup;
    label  = "",
    color  = :black,
    marker = :diamond,
    yaxis  = :identity,
    ylabel = "Speedup (direct / mat-free)",
)

# Proxy series so the speedup styling appears in the main legend.
plot!(p, [NaN], [NaN]; label = "speedup", color = :black, marker = :diamond)

savefig(p, "mat_vec_product_timing_$(Threads.nthreads())_threads.png")
println("Saved")

# `display` can fail in headless or constrained render contexts even when savefig succeeds.
if isinteractive()
    display(p)
end