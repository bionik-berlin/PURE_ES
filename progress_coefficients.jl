"""
    Progress_Coefficients
Calculate the progress coefficients for (1,λ) and (μ/μᵢ.λ) ES
Rechenberg, I. (1994): Evolutionsstrategie '94.
Beyer, H.-G. (2001): The theory of evolution strategies.
"""
module Progress_Coefficients
using LinearAlgebra
using QuadGK
using Statistics
using StatsFuns
using SpecialFunctions
using Distributions

export c_1_c_λ,get_i_table,get_list,c_μ_c_λ,P_μ_d_λ


"""
    c_1_c_λ(λ)
Progress coefficient for (1,λ)-ES
"""
function c_1_c_λ(λ)
    if λ==1
        return 0.0
    end
    f(x)=sqrt(2/π)*λ/BigFloat(2)^BigFloat(λ-1) * BigFloat(x) * exp(-BigFloat(x)^2)*BigFloat(1+erf(BigFloat(x)))^BigFloat(λ-1)
    a=-10
    while isnan(f(a)) || abs(f(a))<1e-100
        a+=0.01
    end
    b=a+1.0
    while abs(f(b))>1e-100 && !isnan(f(b))
        b+=0.01
    end

   res=quadgk(f,a,b)
   Float64(res[1])
end


"""
    P_μ_d_λ(μ,λ,x)
Probability distribution for (μ/μᵢ,λ) on a plane.
"""
function P_μ_d_λ(μ,λ,x)
    if λ==1
        return 0.0
    end
    d=Distributions.Normal(BigFloat(0),BigFloat(1));
    g(x)=binomial(BigInt(λ),BigInt(μ))*μ*pdf(d,BigFloat(x))*(cdf(d,BigFloat(x)))^(μ-1)*(1-cdf(d,BigFloat(x)))^(λ-μ)
   res=g(x)
 Float64(res[1])
end


"""
    c_μ_c_λ(λ)
Progress coefficient for (μ/μᵢ,λ)-ES
"""
function c_μ_c_λ(μ,λ)
    if (μ≥λ) || (λ==1)
        return 0.0
    end
    f(x)=exp(-BigFloat(x)^2)*(λ-μ)/(2*π)*binomial(BigInt(λ),μ)*BigFloat(normcdf(x))^BigFloat(λ-μ-1) * BigFloat(1-BigFloat(normcdf(x)))^BigFloat(μ-1)
    a=-10
    while isnan(f(a)) || abs(f(a))<1e-100
        a+=0.01
    end
    b=a+1.0
    while abs(f(b))>1e-100 && !isnan(f(b))
        b+=0.01
    end

   res=quadgk(f,a,b)
   Float64(res[1])
end


"""
    get_list(λ_in...)
Probuce list of progress coefficients for
    λ from 1 to λ_in
or
    λ from λ_in[1] to λ_in[2]
"""
function get_list(λ_in...)
    if length(λ_in)==1
        λ=1:λ_in[1]
    elseif length(λ_in)==2
        λ=λ_in[1]:λ_in[2]
    else
        return []
    end
    c_1_c_λ.(λ)
end

"""
    get_i_table(λ_b,λ_e)
Produce table of progress coefficients for (μ/μ,λ)-ES for λ from λ_b to λ_e
and μ from 1 to λ.
"""
function get_i_table(λ_b,λ_e)
    [[c_μ_c_λ.(μ,λ)  for μ=1:λ] for λ=λ_b:λ_e]
end


end
