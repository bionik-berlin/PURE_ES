using LinearAlgebra
using Plots
using Printf
using Dates
# Adapted from
# Hansen, N. :The CMA Evolution Strategy: A Tutorial. 2016
# https://arxiv.org/pdf/1604.00772.pdf
# from http://cma.gforge.inria.fr/purecmaes.m
# Adapted from https://github.com/Staross/JuliaCMAES/blob/master/cmaes.jl

"""
Implementation of the (μ/μ,λ)-CMA-ES (Hansen 2016)
cmaes_d(f ,xᵢ ,σ [ ,λ ,μ ,fevalₛ])
* f::Function - Fitness function to optimize
* xᵢ::AbstractVector - Initial point of search
* σ::Union{Number,AbstractVector} - Initial stepsize(s)
* λ::Integer - Number of offspring
* μ::Integer - Number of parents
* fevalₛ::Integer - Maximal number of fitness function evaluations
* dis - [:plot , :text, :none]
for details consult:
N. Hansen: The CMA Evolution Strategy: A Tutorial. 2016
https://arxiv.org/pdf/1604.00772.pdf
# Retruns
# Examples
    res=cmaes(x->sum(x.^2),ones(10),0.1)
"""
function cmaes(f::Function, 𝒙ᵢ::AbstractVector, 𝝈::Union{AbstractVector,Number}; λ::Integer=0,μ::Integer=0,fevalₛ::Integer=0,dis=[:none])

    N = length(𝒙ᵢ)

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

    fᵢ = f(𝒙ᵢ)
    if(! (typeof(fᵢ) <: AbstractFloat) )
        error("Q should return a scalar Float")
    end

    𝒎 = 𝒙ᵢ
    fₛ = 1e-10  # stop if fitness < stopfitness (minimization)
    if(fevalₛ==0) fevalₛ= 100*N^2+0*1e3*N^2 end # stop after fevalₛ number of function evaluations

    #########
    # Strategy parameter setting: Selection
    if(λ==0)
        λ = convert(Integer,4+floor(3*log(N)))  # population size, offspring number
    end

    if (μ==0)
        μ =λ/2                   # number of parents/points for recombination
        𝒘 = log(μ+1/2).-log.(1:μ) # muXone array for weighted recombination
        μ = convert(Integer,floor(μ))
        𝒘 = 𝒘/sum(𝒘)     # normalize recombination weights array
        μₑ=sum(𝒘)^2/sum(𝒘.^2) # variance-effectiveness of sum w_i x_i
    else
        𝐰 = ones(μ)/μ
        μₑ=μ
    end

    #########
    # Strategy parameter setting: Adaptation

    cC = (4 + μₑ/N) / (N+4 + 2*μₑ/N) # time constant for cumulation for C
    cσ = (μₑ+2) / (N+μₑ+5)  # t-const for cumulation for sigma control

    c1 = 2 / ((N+1.3)^2+μₑ)   # learning rate for rank-one update of C
    cμ = min(1-c1, 2 * (μₑ-2+1/μₑ) / ((N+2)^2+μₑ) )   # and for rank-mu update
    dσ = 1.0 + 2.0*max(0.0, sqrt((μₑ-1)/(N+1))-1) + cσ # damping for sigma, usually close to 1
    ΧN=sqrt(N)*(1-1/(4*N)+1/(21*N^2))  # expectation of  ||N(0,I)|| == norm(randn(N,1))


    #########
    𝑩=I
    if (length(𝝈)==1)
        σ = 𝝈
        𝑫=diagm(ones(N))
    else
        σ = maximum(𝝈)
        𝑫 = diagm(sqrt.(𝝈/σ))
    end

    𝑪 = (𝑩 * 𝑫) * transpose(𝑩 * 𝑫)      # covariance matrix == BD*(BD)'

    #init a few things
    𝒙⁽ᵍ⁺¹⁾= zeros(N,λ)
    𝒚⁽ᵍ⁺¹⁾= zeros(N,λ)
    𝒛⁽ᵍ⁺¹⁾= zeros(N,λ)
    𝐪⁽ᵍ⁺¹⁾= zeros(λ)
    idx = zeros(λ)

    γ = 0           # Generation count
    feval = 0       # function evaluation count
    eigeneval=0     # counter for 𝑩 and 𝑫 evaluation
    𝐩C = zeros(N)   # Path for covariance
    𝐩σ = zeros(N)   # Path general stepsize

    # -------------------- Generation Loop --------------------------------
    while feval < fevalₛ
        for l=1:λ  # Generate and evaluate λ offspring
            𝒛⁽ᵍ⁺¹⁾[:,l] = randn(N)
            𝒚⁽ᵍ⁺¹⁾[:,l] = 𝑩 * 𝑫 * 𝒛⁽ᵍ⁺¹⁾[:,l]         # Eq. 39
            𝒙⁽ᵍ⁺¹⁾[:,l] = 𝒎 + σ * 𝒚⁽ᵍ⁺¹⁾[:,l];       # Eq. 40
            𝐪⁽ᵍ⁺¹⁾[l] = f( 𝒙⁽ᵍ⁺¹⁾[:,l] )              # objective function call
            feval+=1
        end
        γ = γ+1

        idx = sortperm(𝐪⁽ᵍ⁺¹⁾)     # index of sorted obj. values
        𝐪⁽ᵍ⁺¹⁾ = 𝐪⁽ᵍ⁺¹⁾[idx]       # sort obj. values


        𝒎ₒ = 𝒎                    # record old mean
        𝒎 = 𝒙⁽ᵍ⁺¹⁾[:,idx[1:μ]]*𝒘  # recombination  Eq. 42
        𝒛ₘ= 𝒛⁽ᵍ⁺¹⁾[:,idx[1:μ]]*𝒘  #  𝑫⁻¹* 𝑩ᵀ * (𝒎-𝒎ₘ)/σ
        𝒚ₘ= 𝒚⁽ᵍ⁺¹⁾[:,idx[1:μ]]*𝒘  # 𝑩 * D * 𝒛ₘ

        # Cumulation: Update evolution paths
        𝐩σ = (1-cσ) * 𝐩σ + sqrt(cσ*(2-cσ)*μₑ) * 𝑩 * 𝒛ₘ  ;  # Eq. 43
        hσ=   (sum(𝐩σ.^2)/(1-(1-cσ)^(2*γ)) )/N  <  (2 + 4/(N+1)) ? 1 : 0

        𝐩C = (1-cC) * 𝐩C + hσ * sqrt(cC*(2-cC)*μₑ) * 𝒚ₘ;

        # Adapt covariance matrix C # Eq. 47
        𝑪 =( (1-c1-cμ) * 𝑪                       # old 𝑪
           + c1 * (𝐩C * transpose(𝐩C)            # rank one update
           +(1-hσ) * cC * (2-cC) * 𝑪)            # magic correction
           +cμ * 𝒚⁽ᵍ⁺¹⁾[:,idx[1:μ]] * diagm(𝒘) * transpose(𝒚⁽ᵍ⁺¹⁾[:,idx[1:μ]])
           )                                    # rank μ update

        # Adapt step size σ
        σ = σ  * exp((cσ/dσ)*(norm(𝐩σ)/ΧN - 1)); # Eq. 44

        # Update B and D from C
        if feval - eigeneval > λ/(c1+cμ)/N/10  # to achieve O(N^2)
            eigeneval = feval
            𝑪 = triu(𝑪) + transpose(triu(𝑪,1))  # enforce symmetry
            𝑫,𝑩=eigen(𝑪)
            𝑫 = diagm(sqrt.(𝑫))        # D contains standard deviations now
        end


        if 𝐪⁽ᵍ⁺¹⁾[1] <= fₛ   # Stop if obj. fun below threshold
            break;
        end
        if maximum(diag(𝑫)) > 1e7 * minimum(diag(𝑫))
            println("Condition error !")
            break;
        end
        if  𝐪⁽ᵍ⁺¹⁾[1] ==  𝐪⁽ᵍ⁺¹⁾[ceil(Integer,0.7*λ)]
            σ = σ * exp(0.2+cσ/dσ)
            println("Flat fitness, consider reformulating the objective")
        end

        if :plot ∈ dis # Collect information during the optimization
            push!(dat_q, 𝐪⁽ᵍ⁺¹⁾[1])
            push!(dat_σ, σ)
            for id=1:N
                push!(dat_d[id],1e5*diag(𝑫)[id])
                push!(dat_x[id],𝒎[id])
            end
        end
        if :text  ∈ dis # Print status every 5 seconds
            if (now()-start_time).value > 5e3
                @printf("%d \t %e \t %e \t %e \t %e\n" ,feval,𝐪⁽ᵍ⁺¹⁾[1],σ,
                                                σ*sqrt(maximum(diag(𝑪))),
                                            maximum(diag(𝑫))/minimum(diag(𝑫)))
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
    return  (qmin=𝐪⁽ᵍ⁺¹⁾[1],xmin=𝒙⁽ᵍ⁺¹⁾[:, idx[1]],γ=γ,feval=feval,N=N)

end



sphere = 𝐱-> sum(𝐱.^2)
disc = 𝐱-> 1e6*𝐱[1]^2+sum(𝐱[2:end].^2)
cigar = 𝐱-> 𝐱[1]^2+1e6*sum(𝐱[2:end].^2)
rosenbrock = 𝐱 -> sum((1 .- 𝐱[1:end-1]).^2)+100.0*sum((𝐱[2:end]-(𝐱[1:end-1]).^2).^2)
ssphere = x -> sqrt(sum(x.^2))
schwefel = x -> sum([ sum(x[1:i])^2 for i=1:length(x)])
cigtab = x -> x[1]^2 + 1e8*x[end]^2 + 1e4*sum(x[2:(end-1)].^2)
elli = x -> sum([ x[i]^2 * 1e6.^((i-1.0)/(length(x)-1.0)) for i=1:length(x)] )
elli100 = x -> sum([ x[i]^2 * 1e4.^((i-1.0)/(length(x)-1.0)) for i=1:length(x)] )
plane =x -> x[1]
twoaxes = x->  sum([x[i]^2 for i=1:floor(Integer,length(x)/2)])+1e6 * sum([x[i]^2 for i=floor(Integer,length(x)/2)+1:length(x)])
parabR = x ->  -x[1] + 100.0*sum(x[2:end].^2)
sharpR = x ->  -x[1] + 100.0*norm(x[2:end])
diffpow = x -> sum([abs(x[i])^(2+10.0*(i-1.0)/(length(x)-1.0)) for i=1:length(x)])
rastrigin10 = x -> 10.0*length(x)+sum([ ((10.0^((i-1.0)/(length(x)-1.0))*x[i])^2-10.0*cos(2*π*10.0^((i-1.0)/(length(x)-1.0))*x[i])) for i=1:length(x)])
