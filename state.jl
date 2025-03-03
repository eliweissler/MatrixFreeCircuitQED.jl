
import Base: size, getindex, setindex!, iterate, length
import KrylovKit: add!!, apply

dtype = ComplexF64

function one_one(N, n)
    v = zeros(dtype, N)
    v[n] = 1
    return v
end


mutable struct State{T} <: AbstractVector{T}
    data::Vector{T}
    buffer::Vector{Vector{T}}
    new_data::Vector{T}
    target::Int
    source::Int
end

# For KrylovKit
function add!!(y::State, x::State ; α::Number = 1, β::Number = 1)
    y.data *= β
    y.data += α*x.data
    y
end
function apply(operator, x::State, α₀, α₁)
    og = State(x.data, x.buffer, x.new_data, x.target, x.source)
    y = apply(operator, x)
    if α₀ != zero(α₀) || α₁ != one(α₁)
        y = add!!(y, og, α₀, α₁)
    end
    return y
end

# Constructor
State(N, n) = State(one_one(N, n), [zeros(dtype, N), zeros(dtype, N)], zeros(dtype, N), 1, 2)

# Constructor
State(N) = State(N, 1)


# Define size
size(v::State) = (length(v.data),)

# Define length
length(v::State) = length(v.data)

# Define indexing
getindex(v::State, i::Int) = v.data[i]

# Define setting values
setindex!(v::State, val, i::Int) = (v.data[i] = val)

# Define iteration
iterate(v::State, state=1) = state > length(v.data) ? nothing : (v.data[state], state + 1)


function swap_target!(v::State)
    old_target = v.target
    v.target = v.source
    v.source = old_target
end

function add_buffer!(v::State, i::Integer)
    v.new_data += v.buffer[i]
end

# function reset_buffer!(v::State)
#     v.buffer[1] = 
# end

function reset_data!(v::State)
    v.data = v.new_data
    v.new_data *= 0
end

function reset_new_data!(v::State)
    v.new_data = zeros(length(v))
end


## TODO: Add to this
# Define operations on the state
function x_temp!(v::State, source::Vector, target::Vector)
    target[1] = source[2]
    n = length(v)
    # sqrt coefficients are index from 0, not 1
    for i in 2:n-1
        target[i] = sqrt(i-1)*source[i-1] + sqrt(i)*source[i+1]
    end
    target[n] = sqrt(n-1)*source[n-1]
end


function p_temp!(v::State, source::Vector, target::Vector)
    target[1] = -1.0im*source[2]
    n = length(v)
    # sqrt coefficients are index from 0, not 1
    for i in 2:n-1
        target[i] = 1.0im*(sqrt(i-1)*source[i-1] - sqrt(i)*source[i+1])
    end
    target[n] = 1.0im*sqrt(n-1)*source[n-1]
end

# lowering operator
function a_temp!(v::State, source::Vector, target::Vector)
    target[1] = -1.0im*source[2]
    n = length(v)
    # sqrt coefficients are index from 0, not 1
    for i in 2:n-1
        target[i] = sqrt(i)*source[i+1]
    end
    target[n] = 0
end

# raising operator
function a_dag_temp!(v::State, source::Vector, target::Vector)
    target[1] = 0
    n = length(v)
    # sqrt coefficients are index from 0, not 1
    for i in 2:n-1
        target[i] = sqrt(i-1)*source[i-1]
    end
    target[n] = sqrt(n-1)*source[n-1]
end

# number operator
function N_temp!(v::State, source::Vector, target::Vector)
    # coefficients are index from 0, not 1
    for i in 1:length(v)
        target[i] = (i-1)*source[i]
    end
end

function id!(v::State, source::Vector, target::Vector)
    for i in 1:length(v)
        target[i] = source[i]
    end
end


function apply_template!(v::State, templates::Vector, n::Int64, coeff::Number)

    # data -> buffer
    copyto!(v.buffer[1], v.data)
    v.source = 1
    v.target = 2

    for app in 1:n
        for temp in templates
            temp(v, v.buffer[v.source], v.buffer[v.target])
            swap_target!(v)
        end
    end

    v.new_data += coeff*v.buffer[v.source]

    # template(v, v.data, v.buffer[1])

    # # Subsequent applications, alternate buffer1 <--> buffer2
    # for app in 2:n
    #     source = 1 + mod(app, 2)
    #     target = 3 - source
    #     template(v, v.buffer[source], v.buffer[target])
    # end

    # Add to new data
    # v.new_data += coff*v.buffer[1 + mod(n+1, 2)]

end

function apply_template!(v::State, template::Function, n::Integer, coeff::Number)

    apply_template!(v, [template], n, coeff)

    # # First application, data -> buffer
    # template(v, v.data, v.buffer[1])

    # # Subsequent applications, alternate buffer1 <--> buffer2
    # for app in 2:n
    #     source = 1 + mod(app, 2)
    #     target = 3 - source
    #     template(v, v.buffer[source], v.buffer[target])
    # end

    # # Add to new data
    # v.new_data += coeff*v.buffer[1 + mod(n+1, 2)]

end

# s = State(10)
# apply_template!(s, p_temp!, 2, 1)