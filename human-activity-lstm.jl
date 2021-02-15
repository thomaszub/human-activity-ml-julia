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

y_train = loadLabelData("data/UCI HAR Dataset/train/y_train.txt")
y_test = loadLabelData("data/UCI HAR Dataset/test/y_test.txt")
yh_train = onehotbatch(y_train, 1:6)
yh_test = onehotbatch(y_test, 1:6)

# Prepare batch loader for train data
using Flux.Data: DataLoader

train_loader = DataLoader((X_train, yh_train), batchsize=32, shuffle=true)

# Build model

using Flux
using Flux: @epochs
using Flux.Losses

model = Chain(
    LSTM(9, 9),
    Dense(9, 6),
    softmax
)

opt = ADAM()

function loss(x, y)
    yh = last(map(model, [view(x, :, t, :) for t in 1:128]))
    l = logitcrossentropy(yh, y)
    Flux.reset!(model)
    l
end


#@epochs 2
Flux.train!(loss, params(model), train_loader, opt)
