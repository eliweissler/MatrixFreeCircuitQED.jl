
import Base: size, getindex, setindex!, iterate, length

dtype = ComplexF32

function one_one(N, n)
    v = zeros(dtype, N)
    v[n] = 1
    return v
end


mutable struct State{T} <: AbstractVector{T}
    data::Vector{T}
    buffer::Vector{Vector{T}}
    new_data::Vector{T}
end

# Constructor
State(N, n) = State(one_one(N, n), [zeros(dtype, N), zeros(dtype, N)], zeros(dtype, N))

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



function add_buffer!(v::State, i::Integer)
    v.new_data += v.buffer[i]
end

function reset_buffer!(v::State)
    v.buffer = zeros(length(v))
end

function reset_data!(v::State)
    v.data = v.new_data
    v.new_data = zeros(length(v))
end

function reset_new_data!(v::State)
    v.new_data = zeros(length(v))
end


## TODO: Add to this
# Define operations on the state
function x_temp!(v::State, source::Vector, target::Vector)
    target[1] = sqrt(2)*source[2]
    n = length(v)
    for i in 2:n-1
        target[i] = sqrt(i)*source[i-1] + sqrt(i+1)*source[i+1]
    end
    target[n] = sqrt(n)*source[n-1]
end
function p_temp!(v::State, source::Vector, target::Vector)
    target[1] = -1.0im*sqrt(2)*source[2]
    n = length(v)
    for i in 2:n-1
        target[i] = 1.0im*(sqrt(i)*source[i-1] - sqrt(i+1)*source[i+1])
    end
    target[n] = 1.0im*sqrt(n)*source[n-1]
end

function apply_template!(v::State, template::Function, n::Integer, coeff::Number)

    # First application, data -> buffer
    template(v, v.data, v.buffer[1])

    # Subsequent applications, alternate buffer1 <--> buffer2
    for app in 2:n
        source = 1 + mod(app, 2)
        target = 3 - source
        template(v, v.buffer[source], v.buffer[target])
    end

    # Add to new data
    v.new_data += (coeff^n)*v.buffer[1 + mod(n+1, 2)]

end

# s = State(10)
# apply_template!(s, p_temp!, 2, 1)