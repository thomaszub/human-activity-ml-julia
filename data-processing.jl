using DelimitedFiles

function transform(line::String)
    splitted = split(strip(line), ' ', keepempty=false)
    map(s -> parse(Float32, s), splitted)
end

function loadFile(file::String)
    open(file) do data
        [transform(line) for line in eachline(data)]
    end
end

function loadFeatureData(directory::String, feature_names::Array{String,1}, type::String)
    data = [hcat(loadFile(string(directory, "/", feature, "_", type, ".txt"))...) for feature in feature_names]
    permutedims(reshape(vcat(data...), 128, size(feature_names)..., :), (2, 1, 3))
end

function loadLabelData(file::String)
    toInt(line) = parse(Int32, line)
    open(file) do data
        [toInt(line) for line in eachline(data)]
    end
end
