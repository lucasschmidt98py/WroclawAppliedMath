using Random
using Statistics
using Base.Threads

function sigmoid(z)
    return 1.0/ (1.0 + exp(-z))
end

function sigmoid_derivative(z)
    return sigmoid(z) * (1.0 - sigmoid(z))
end

mutable struct NeuralNetwork
    neurons::Vector{Int}
    w1::Matrix{Float64}
    w2::Matrix{Float64}
    w3::Matrix{Float64}
    b1::Matrix{Float64}
    b2::Matrix{Float64}
    b3::Matrix{Float64}
    λ::Float64
    epochs::Int
    Loss::Vector{Float64}
end

function NeuralNetwork(sizes::Vector{Int}, λ::Float64=10e-3)
    n_input, n_neurons_1, n_neurons_2, n_output, n_epochs = sizes
    neurons = [n_input,n_neurons_1,n_neurons_2,n_output]
    w1 = randn(n_neurons_1, n_input) 
    w2 = randn(n_neurons_2, n_neurons_1)
    w3 = randn(n_output, n_neurons_2)
    b1 = zeros(n_neurons_1, 1)
    b2 = zeros(n_neurons_2, 1)
    b3 = zeros(n_output, 1)

    return NeuralNetwork(neurons,w1, w2, w3, b1, b2, b3, λ, n_epochs, zeros(Float64, n_epochs))
end

function Loss_L2(y, y_hat)
    return 0.5 * sum((y_hat .- y) .^ 2)
end

function Train_2Layers(nn::NeuralNetwork, training_data::Vector{Tuple{Vector{Float64}, Vector{Float64}}})
    for epoch in 1:nn.epochs
        for (x, y) in training_data
            z1 = nn.w1 * x .+ nn.b1
            a1 = sigmoid.(z1)

            z2 = nn.w2 * a1 .+ nn.b2
            a2 = sigmoid.(z2)

            z3 = nn.w3 * a2 .+ nn.b3
            y_hat = sigmoid.(z3)

            nn.Loss[epoch] += Loss_L2(y, y_hat) / length(training_data)

            delta3 = (y_hat .- y) .* sigmoid_derivative.(z3)
            dLdW3 = delta3 * a2'
            dLdB3 = delta3

            delta2 = nn.w3' * delta3 .* sigmoid_derivative.(z2)
            dLdW2 = delta2 * a1'
            dLdB2 = delta2

            delta1 = nn.w2' * delta2 .* sigmoid_derivative.(z1)
            dLdW1 = delta1 * x'
            dLdB1 = delta1

            nn.w3 .-= nn.λ * dLdW3
            nn.w2 .-= nn.λ * dLdW2
            nn.w1 .-= nn.λ * dLdW1

            nn.b3 .-= nn.λ * sum(dLdB3, dims=2)
            nn.b2 .-= nn.λ * sum(dLdB2, dims=2)
            nn.b1 .-= nn.λ * sum(dLdB1, dims=2)
        end
    end
end

function Y_hat(nn::NeuralNetwork, x::Vector{Float64})
    z1 = nn.w1 * x .+ nn.b1
    a1 = sigmoid.(z1)

    z2 = nn.w2 * a1 .+ nn.b2
    a2 = sigmoid.(z2)

    z3 = nn.w3 * a2 .+ nn.b3
    return sigmoid.(z3)
end

function Validate_2Layer(nn::NeuralNetwork, validation_data::Vector{Tuple{Matrix{Float64}, Matrix{Float64}}})
    total_loss = 0.0
    accuracy = 0.0
    for (x, y) in validation_data
        y_hat = Y_hat(nn, x)
        total_loss += Loss_L2(y, y_hat) / length(validation_data)
        if argmax(y_hat)[1] == argmax(y)-1
            accuracy += 1.0 / length(validation_data)
        end
    end
    return total_loss, accuracy
end 

function Test_2Layer(nn::NeuralNetwork, test_data::Vector{Tuple{Vector{Float64}, Vector{Float64}}})
    count = [ argmax( Y_hat(nn,x) )[1] == argmax( y )  for (x,y) in test_data ]
    return sum(count) / length(count)
end
