include("state.jl")

using KrylovKit: eigsolve


function harm_osc(v::State)

    apply_template!(s, p_temp!, 2, 1)
    apply_template!(s, x_temp!, 2, 1)
    reset_data!(v)

end


x0 = State(10)

vals, vecs, info = eigsolve(harm_osc, x0, 5)