
using Pkg
Pkg.activate(".")

include("state.jl")

using KrylovKit: eigsolve


function harm_osc(v::State)

    m = 1
    w = 1

    apply_template!(v, p_temp!, 2, 1/(2*m) * (m*w/2))
    apply_template!(v, x_temp!, 2, (m*(w^2)/2) * 1/(2*m*w))
    reset_data!(v)

    v

end


function harm_osc2(v::State)

    m = 1
    w = 1

    apply_template!(v, [a_temp!, a_dag_temp!], 1, w)
    apply_template!(v, id!, 1, w/2)
    reset_data!(v)

    v

end

function harm_osc3(v::State)

    m = 1
    w = 1

    apply_template!(v, N_temp!, 1, w)
    apply_template!(v, id!, 1, w/2)
    reset_data!(v)

    # v

end


x0 = State(10, 3)

# harm_osc3(x0)

vals, vecs, info = eigsolve(harm_osc, x0, 5, ishermitian=true)