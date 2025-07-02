using Plots
using StatsBase

function Precipitating_Automata(nt::Int, L::Int, θ::Vector{Int}, IC::Matrix{Int64})
    θ1 , θ2 = θ
    domain = zeros(Int, nt, L, L)
    domain[1, :, :] .= IC
    
    for t = 1:nt-1 , i = 2:L-1 , j = 2:L-1
        σ = sum(domain[t, i-1:i+1, j-1:j+1]) - domain[t, i, j]
        if domain[t, i, j] == 0 && σ <= θ2 && σ >= θ1 || domain[t , i , j] == 1
            domain[t+1, i, j] = 1
        else
            domain[t+1, i, j] = 0
        end
    end
    return domain
end

function Precipitating_Automata_Random(nt::Int, L::Int, θ::Vector{Int} , IC::Matrix{Int64} , p::Float64 = .5)
    θ1 , θ2 = θ
    domain = zeros(Int, nt, L, L)
    domain[1, :, :] .= IC
    
    for t = 1:nt-1 , i = 2:L-1 , j = 2:L-1
        σ = sum(domain[t, i-1:i+1, j-1:j+1]) - domain[t, i, j]
        if rand() < p && domain[t, i, j] == 0 && σ <= θ2 && σ >= θ1 || domain[t , i , j] == 1
            domain[t+1, i, j] = 1
        else
            domain[t+1, i, j] = 0
        end
    end
    return domain
end

function makefig_precipitating(nt::Int , L::Int , θ1::Int , θ2::Int , IC::Matrix{Int64} , p::Float64 = .5 , single::Bool = true)
    domain1 = Precipitating_Automata(nt, L, [θ1, θ2], IC)
    domain2 = Precipitating_Automata_Random( nt, L, [θ1, θ2], IC , p )

    p1 = heatmap( domain1[end,:,:] , color=:bukavu , title = "Deterministic Rule")
    p2 = heatmap( domain2[end,:,:] , color=:bukavu , title = "Random Rule , p = $(p)")
    p3 = plot( sum( domain1 , dims=(2,3))[:] / L^2 , label = "Deterministic" ,ylims=(0,1))
    plot!( sum( domain2 , dims=(2,3))[:] / L^2  , label = "Random Rule" ,ylims=(0,1))
    
    plot( p1 , p2 , p3 , size = (1800,600) , layout=(1,3) , colorbar = false , suptitle = "θ1=$(θ1) , θ2=$(θ2)")
    if single
        savefig("Precipitation_Single/single_$(θ1)$(θ2).png")
    else
        savefig("Precipitation_Rand/rand_$(θ1)$(θ2).png")
    end  
end

function Association_Automata(nt::Int, L::Int, θ::Vector{Int} , δ::Vector{Int}, IC::Matrix{Int64})
    θ1 , θ2 = θ
    δ1 , δ2 = δ
    domain = zeros(Int, nt, L, L)
    domain[1, :, :] .= IC
    
    for t = 1:nt-1 , i = 2:L-1 , j = 2:L-1
        σ = sum(domain[t, i-1:i+1, j-1:j+1]) - domain[t, i, j]
        if (domain[t, i, j] == 0 && σ <= θ2 && σ >= θ1) || (domain[t , i , j] == 1 && σ <= δ2 && σ >= δ1)
            domain[t+1, i, j] = 1
        else
            domain[t+1, i, j] = 0
        end
    end
    return domain
end

function Association_Automata_Random(nt::Int, L::Int, θ::Vector{Int} , δ::Vector{Int}, IC::Matrix{Int64} , p = .5)
    θ1 , θ2 = θ
    δ1 , δ2 = δ
    domain = zeros(Int, nt, L, L)
    domain[1, :, :] .= IC
    
    for t  = 1:nt-1 , i = 2:L-1 , j = 2:L-1
        σ = sum(domain[t, i-1:i+1, j-1:j+1]) - domain[t, i, j]
        if rand() < p && (domain[t, i, j] == 0 && σ <= θ2 && σ >= θ1) || (domain[t , i , j] == 1 && σ <= δ2 && σ >= δ1)
            domain[t+1, i, j] = 1
        else
            domain[t+1, i, j] = 0
        end
    end
    return domain
end

function makefig_association(nt::Int , L::Int , θ::Vector{Int} , δ::Vector{Int} , IC::Matrix{Int64} , p::Float64 = .5 , single::Bool = true)
    domain1 = Association_Automata(nt, L, θ , δ, IC)
    domain2 = Association_Automata_Random( nt, L, θ , δ , IC , .5 )

    p1 = heatmap( domain1[end,:,:] , color=:bukavu , title = "Deterministic Rule")
    p2 = heatmap( domain2[end,:,:] , color=:bukavu , title = "Random Rule , p = $(p)")
    p3 = plot( sum( domain1 , dims=(2,3))[:] / L^2 , label = "Deterministic" ,ylims = (0,1))
    plot!( sum( domain2 , dims=(2,3))[:] / L^2 , label = "Random Rule" ,ylims = (0,1))
    
    plot( p1 , p2 , p3 , size = (1800,600) , layout=(1,3) , colorbar = false , suptitle = "θ=$(θ) , δ=$(δ)")
    if single
        savefig("Association_Single/single_$(θ[1])$(θ[2])_$(δ[1])$(δ[2]).png")
    else
        savefig("Association_Rand/rand_$(θ[1])$(θ[2])_$(δ[1])$(δ[2]).png")
    end  
end