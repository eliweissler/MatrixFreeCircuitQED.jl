using Pkg
Pkg.activate(".")

using QuantumOptics
using LinearAlgebra
using ClassicalOrthogonalPolynomials

N = 100
b = FockBasis(N)

a = destroy(b)
adag = create(b)

function x_n(n, a, adag, x, y)
    op = 0*a
    for m in 0:Int(floor(n/2))
        factor = (x*y)^m * factorial(n) / (factorial(m)*factorial(n-2m)*2^m)
        for r in 0:(n - 2*m)
            op += factor*binomial(n-2*m, r) * x^r * y^(n-2*m-r) * adag^(n - 2*m - r) * a^r
        end
    end
    return op
end

function x_n_coeff(n, x, y)
    # Factor in terms of coefficient of adag (+1)
    coeffs = zeros(n+1, n+1)
    for m in 0:Int(floor(n/2))
        factor = (x*y)^m * factorial(n) / (factorial(m)*factorial(n-2m)*2^m)
        for r in 0:(n - 2*m)
            exp1 = n - 2*m - r
            exp2 = r
            coeffs[exp1+1, exp2+1] += factor*binomial(n-2*m, r) * x^r * y^(n-2*m-r)
        end
    end
    return coeffs
end




test = x_n(2, a, adag, 1, 1)
act = (a + adag)^2

d = displace(b, 1)


function displace_mn(alpha, m, n)
    if m < n 
        # Swap m, n & take -alpha conj
        og_m = m
        m = n 
        n = og_m
        alpha = -alpha'
    end
    return sqrt(factorial(n)/factorial(m)) * alpha^(m-n) * exp(-abs(alpha)^2 / 2) * laguerrel(n, m-n, abs(alpha)^2)
end


function displace_mn_log(alpha, m, n)
    # if m < n 
    #     # Swap m, n & take -alpha conj
    #     og_m = m
    #     m = n 
    #     n = og_m
    #     alpha = -alpha'
    # end
    return (1/2)*(sum(log.(1:n)) - sum(log.(1:m))) + (m-n)*log(alpha) - abs(alpha)^2 / 2 + log(laguerrel(n, m-n, abs(alpha)^2))
end

dmn = displace_mn(1, 10, 9)

