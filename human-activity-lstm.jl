# Init project
using Pkg

Pkg.activate(".")
Pkg.instantiate()

# Load raw data
dir_train = "data/UCI HAR Dataset/train/Inertial Signals"
dir_test = "data/UCI HAR Dataset/test/Inertial Signals"

feature_names = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z"
]

using Flux: onehotbatch
include("data-processing.jl")

X_train = loadFeatureData(dir_train, feature_names, "train")
X_test = loadFeatureData(dir_test, feature_names, "test")

y_labels_train = loadLabelData("data/UCI HAR Dataset/train/y_train.txt")
y_labels_test = loadLabelData("data/UCI HAR Dataset/test/y_test.txt")
y_train = onehotbatch(y_labels_train, 1:6)
y_test = onehotbatch(y_labels_test, 1:6)

# Prepare batch loader for train data
using Flux.Data: DataLoader

train_loader = DataLoader((X_train, y_train), batchsize=32, shuffle=true)

# Build model
using Flux
using Flux: @epochs
using Flux.Losses
using Statistics: mean

model = Chain(
    LSTM(9, 32),
    Dense(32, 16, relu),
    Dense(16, 6),
    softmax
)

apply(model, x) = last(map(model, [view(x, :, t, :) for t in 1:128]))

accuracy(y_pred, y) = mean(Flux.onecold(y_pred) .== Flux.onecold(y))

function evalcb()
    y_pred = apply(model, X_train)
    @info "Loss: $(crossentropy(y_pred, y_train)), Accuracy: $(accuracy(y_pred, y_train))"
    Flux.reset!(model)
end

function loss(x, y)
    y_pred = apply(model, x)
    l = crossentropy(y_pred, y)
    Flux.reset!(model)
    l
end

function train()
    opt = ADAM()
    Flux.reset!(model)
    @epochs 10 @time Flux.train!(loss, params(model), train_loader, opt, cb = Flux.throttle(evalcb, 5))
end

train()

# Final evaulation and minimum accuracy needed to beat "guess only most often class"
hist = reshape(sum(y_train, dims=2), :)
print("Minimum needed accuracy: $(maximum(hist)/sum(hist))")

y_pred = apply(model, X_test)
print("Actual accurcy $(accuracy(y_pred, y_test))")
