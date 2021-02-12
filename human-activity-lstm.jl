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

include("data-processing.jl")

X_train = loadFeatureData(dir_train, feature_names, "train")
X_test = loadFeatureData(dir_test, feature_names, "test")

y_train = loadLabelData("data/UCI HAR Dataset/train/y_train.txt")
y_test = loadLabelData("data/UCI HAR Dataset/test/y_test.txt")

# Prepare batch loader for train data
using Flux.Data: DataLoader

train_loader = DataLoader((X_train, y_train), batchsize=32, shuffle=true)
