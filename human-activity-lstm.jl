# Init project
using Pkg

Pkg.activate(".")
Pkg.instantiate()

# Load data

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

include("data-processing.jl")

X_train = loadData(dir_train, feature_names, "train")

using Flux.Data: DataLoader

loader_X_train = DataLoader(X_train, batchsize=32)

for x_batch in loader_X_train
    print(size(x_batch))
end
