using DataFrames
using CSV
using Random
function q(x)
    sum(x.^2)
end

function es_msr(x₀,λ,σ₀,qₜ,N)
    xₑ=x₀
    σ=σ₀
    γ=0
    qₑ=q(xₑ)
    while (qₑ>qₜ)
        σₙ=[σ*exp(rand([-1.0,1.0])) for l=1:λ]
        xₙ=[xₑ+σₙ[l]*randn(N) for l=1:λ]
        qₙ=q.(xₙ)
        idx=sortperm(qₙ)
        xₑ=xₙ[idx[1]]
        σ=σₙ[idx[1]]
        qₑ=qₙ[idx[1]]
        γ=γ+1
    end
    (γ,xₑ,qₑ,σ)
end

REP=250
N=100
G1=[]
G2=[]
for r=1:REP
    res1=es_msr(ones(N),10,1/sqrt(N),1.0e-10,N)
    res2=es_msr(ones(N),27,1/sqrt(N),1.0e-10,N)
#    @show[res[1]]
    push!(G1,res1[1])
    push!(G2,res2[1])
end
df = DataFrame(id = 1:length(G1), x1 = G1, x2=G2)
if (pwd()=="/home/iskl")
    cd("github/PURE_ES")
end
CSV.write("./test1.csv", df)
