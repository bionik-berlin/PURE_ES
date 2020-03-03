using LinearAlgebra
using Statistics
using Plots
using Printf
using Dates
# Adapted from
# Beyer, Hans-Georg; Sendhoff, Bernhard (2008): Covariance Matrix Adaptation Revisited – The CMSA Evolution Strategy –.
# In: Günter Rudolph (Hg.): Parallel problem solving from nature - PPSN X.
#  10th international conference, Dortmund, Germany, September 13-17, 2008 : proceedings /
# Günter Rudolph … [et al.] (eds.), Bd. 5199. Berlin:
# Springer (Lecture notes in computer science, 0302-9743, 5199), S. 123–132.l
# isk 03.03.2020

"""
Implementation of the (μ/μ,λ)-CMSA-ES (Beyer 2008)
cmsaes(f ,xᵢ ,σ [,λ ,μ ,fevalₛ])
* f::Function - Fitness function to optimize
* yᵢ::AbstractVector - Initial point of search
* σ::Union{Number,AbstractVector} - Initial stepsize(s)
* λ::Integer - Number of offspring
* μ::Integer - Number of parents
* τf::Float64 - Factor for learning parameter
* fevalₛ::Integer - Maximal number of fitness function evaluations
* dis - [:plot , :text, :none]
for details consult:
H. Beyer ,B. Sendhof: Covariance Matrix Adaptation Revisited – the CMSA Evolution Strategy – (2008)
http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.7929&rank=1
# Retruns
(fmin,xmin,γ,feval,N)
* fmin  - Fitness of best last offspring
* xmin - Best last offspring
* γ - Number of generations
* feval - Number of function evaluations
* N - Problem dimension

# Examples
    res=cmsaes(x->sum(x.^2),ones(10),0.1)
"""
function cmsaes(f::Function, 𝒚ᵢ::AbstractVector, 𝝈::Union{AbstractVector,Number}; λ::Integer=0,μ::Integer=0,τf::Float64=1.0,fevalₛ::Integer=0,dis=[:none])

    N = length(𝒚ᵢ)

    if typeof(dis)==Symbol
        dis=[dis]
    elseif typeof(dis)!==Array{Symbol,1}
        dis=[:none]
    end
    if :plot ∈ dis
        dat_q,dat_σ = [],[]
        dat_d=[[] for id=1:N]
        dat_x=[[] for id=1:N]
    end
    if :text ∈ dis
        start_time=now()
    end

    if (length(𝝈)!=1) && (length(𝝈) != length(𝐱ᵢ))
        error("σ and x₀ have different dimension!")
    end

    fᵢ = f(𝒚ᵢ)
    if(! (typeof(fᵢ) <: AbstractFloat) )
        error("f should return a scalar Float")
    end

    𝒚= 𝒚ᵢ
    fₛ = 1e-10  # stop if fitness < stopfitness (minimization)
    if(fevalₛ==0) fevalₛ= 100*N^2+0*1e3*N^2 end # stop after fevalₛ number of function evaluations

    #########
    # Strategy parameter setting: Selection
    if(λ==0)
        λ = convert(Integer,4+floor(3*log(N)))  # population size, offspring number
    end

    if (μ==0)
        μ =λ/4
        μ = convert(Integer,floor(μ))
    end

    #########
    # Strategy parameter setting: Adaptation

    τ=τf/sqrt(2*N)                  # Eq. 1
    τc=1+τf *N*(N+1)/(2*μ)          # Eq. 2
    τp=sqrt(N)
    cu = floor(Integer,τc)
    if cu > N/2
      cu = floor(Integer,N/2);
    end

    #########
    if (length(𝝈)==1)
        σ = 𝝈
        𝑪= diagm(ones(N))
    else
        σ = maximum(𝝈)
        𝑪 = diagm(𝝈/σ)
    end

    #init a few things
    𝝈⁽ᵍ⁺¹⁾= zeros(λ)
    𝒔⁽ᵍ⁺¹⁾= zeros(N,λ)
    𝒛⁽ᵍ⁺¹⁾= zeros(N,λ)
    𝒚⁽ᵍ⁺¹⁾= zeros(N,λ)
    𝒇⁽ᵍ⁺¹⁾= zeros(λ)
    idx = zeros(λ)
    𝑪½=transpose(sqrt(𝑪))
    𝑪½=det(𝑪½)^(1/N)*𝑪½

    γ = 0           # Generation count
    feval = 0       # function evaluation count

    # -------------------- Generation Loop --------------------------------
    while feval < fevalₛ
        if (γ % cu) == 0
            𝑪=(𝑪+transpose(𝑪))/2.0;
            𝑪½=transpose(sqrt(𝑪))
            𝑪½=det(𝑪½)^(-1/N)*𝑪½
        end
        for l=1:λ  # Generate and evaluate λ offspring
            𝝈⁽ᵍ⁺¹⁾[l]=σ*exp(τ*randn())             # Eq. R1
            𝒔⁽ᵍ⁺¹⁾[:,l] = 𝑪½*randn(N)              # Eq. R2
            𝒛⁽ᵍ⁺¹⁾[:,l] = 𝝈⁽ᵍ⁺¹⁾[l] * 𝒔⁽ᵍ⁺¹⁾[:,l]  # Eq. R3
            𝒚⁽ᵍ⁺¹⁾[:,l] = 𝒚 + 𝒛⁽ᵍ⁺¹⁾[:,l]          # Eq. R4
            𝒇⁽ᵍ⁺¹⁾[l] = f( 𝒚⁽ᵍ⁺¹⁾[:,l] )           # Eq. R5
            feval+=1
        end
        γ = γ+1

        idx = sortperm(𝒇⁽ᵍ⁺¹⁾)     # index of sorted obj. values
        𝒇⁽ᵍ⁺¹⁾ = 𝒇⁽ᵍ⁺¹⁾[idx]       # sort obj. values


        σ=mean(𝝈⁽ᵍ⁺¹⁾[idx[1:μ]])
        𝒛=mean(𝒛⁽ᵍ⁺¹⁾[:,idx[1:μ]],dims=2)
        𝒚=𝒚+𝒛                                     # Eq. R6

        𝑪ₜ=mean([𝒔⁽ᵍ⁺¹⁾[:,k] * transpose(𝒔⁽ᵍ⁺¹⁾[:,k]) for k in idx[1:μ]])
        𝑪= (1-1/τc)*𝑪 + 1/τc * 𝑪ₜ                # Eq. R7

        if 𝒇⁽ᵍ⁺¹⁾[1] <= fₛ   # Stop if obj. fun below threshold
            break;
        end
        if sqrt(maximum(eigen(𝑪).values)) > 1e7 * sqrt(minimum(eigen(𝑪).values))
            println("Condition error !")
            break;
        end
        if  𝒇⁽ᵍ⁺¹⁾[1] ==  𝒇⁽ᵍ⁺¹⁾[ceil(Integer,0.7*λ)]
            σ = σ * exp(0.2)
            println("Flat fitness, consider reformulating the objective")
        end

        if :plot ∈ dis # Collect information during the optimization
            push!(dat_q, 𝒇⁽ᵍ⁺¹⁾[1])
            push!(dat_σ, σ)
            for id=1:N
                push!(dat_d[id],1e5*sqrt(eigen(𝑪).values[id]))
                push!(dat_x[id],𝒚[id])
            end
        end
        if :text  ∈ dis # Print status every 5 seconds
            if (now()-start_time).value > 5e3
                @printf("%d \t %e \t %e \t %e \t %e\n" ,feval,𝒇⁽ᵍ⁺¹⁾[1],σ,
                                                σ*sqrt(maximum(eigen(𝑪).values)),
                                                cond(𝑪½))
                start_time=now()
            end
        end
    end # while, end generation loop

    if :plot ∈ dis   # Plot optimization data
        default(legend=false)
        pl1=plot([dat_q,dat_σ,dat_d],yaxis=:log);
        pl2=plot(dat_x);
        display(plot(pl1,pl2,layout=(2,1)));
    end
    return  (fmin=𝒇⁽ᵍ⁺¹⁾[1],xmin=𝒚⁽ᵍ⁺¹⁾[:, idx[1]],γ=γ,feval=feval,N=N)

end



sphere = y-> sum(y.^2)
schwefel = y -> sum([ sum(y[1:i])^2 for i=1:length(y)])
rosenbrock = y -> sum([100.0*(y[i]^2 -  y[i+1])^2+(y[i]-1.0)^2 for i=1:(length(y)-1)])
rastrigin = y -> 10.0*length(y)+sum([(y[i]^2-10.0*cos(2*π*y[i])) for i=1:length(y)])
disc = 𝐱-> 1e6*𝐱[1]^2+sum(𝐱[2:end].^2)
cigar = 𝐱-> 𝐱[1]^2+1e6*sum(𝐱[2:end].^2)
elli = x -> sum([ x[i]^2 * 1e6.^((i-1.0)/(length(x)-1.0)) for i=1:length(x)] )
twoaxes = x->  sum([x[i]^2 for i=1:floor(Integer,length(x)/2)])+1e6 * sum([x[i]^2 for i=floor(Integer,length(x)/2)+1:length(x)])
n=[2 ;3 ;5 ;10; 20; 40; 80 ;160]
l=[n->8;n->4*n;n->4*n*n ]
