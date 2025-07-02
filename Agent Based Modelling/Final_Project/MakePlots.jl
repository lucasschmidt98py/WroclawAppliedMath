using Base.Threads
include("ReactionDiffusion.jl")

nt = 700
L = 201
p = .5

IC1 = zeros(Int,L, L)
IC1[L÷2+1 , L÷2+1] = 1 
IC2 = rand( [0,1] , L , L )

#=@sync for θ2 = 1:2 
    for θ1 = 1:θ2
        @spawn begin
            makefig_precipitating(nt,L,θ1,θ2,copy(IC1),p,false) 
        end
    end
end

@sync for θ2 = 1:9
    for θ1 = 1:θ2
        @spawn begin
            makefig_precipitating(nt,L,θ1,θ2,copy(IC2),p,false) 
        end
    end
end=#


for θ2 in 1:9
    for θ1 in 1:θ2
        for δ2 in 1:9
            for δ1 in 1:δ2
                @spawn begin
                    makefig_association(nt, L, [θ1, θ2], [δ1, δ2], copy(IC1), p, true)
                end
            end
        end
    end
end