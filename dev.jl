using HTTP, JSON
#using Plots

const API_KEY = "08333c23ecfae8c3874ac3fd49b41e67"

struct TvSeries
    name::String
    adult::Bool
    episode_run_time::Int
    genres::Vector{String}
    first_air_date::String
    last_air_date::String
    networks::Vector{String}
    number_of_episodes::Int
    number_of_seasons::Int
    status::String
    popularity::Float64
end

function fetch_tv_show_ids(page::Int)
    url = "https://api.themoviedb.org/3/discover/tv?include_null_first_air_dates=false&language=tr-TR&page=$page&sort_by=primary_release_date.desc&with_original_language=tr&api_key=$API_KEY"
    response = HTTP.get(url)
    if HTTP.status(response) == 200
        response_data = JSON.parse(String(response.body))
        results = get(response_data, "results", [])
        tv_show_ids = [get(result, "id", nothing) for result in results]
        return tv_show_ids
    else
        error("HTTP Request failed with status: $(HTTP.status(response))")
    end
end

function fetch_tv_series(tv_show_id::Int)
    url = "https://api.themoviedb.org/3/tv/$tv_show_id?language=tr-TR&api_key=08333c23ecfae8c3874ac3fd49b41e67"
    response = HTTP.get(url)
    if HTTP.status(response) == 200
        response_data = JSON.parse(String(response.body))
        return TvSeries(
            response_data["name"],
            response_data["adult"],
            get(response_data["episode_run_time"], 1, 0),
            [get(genre, "name", "") for genre in response_data["genres"]],
            get(response_data, "first_air_date", ""),
            get(response_data, "last_air_date", ""),
            [get(network, "name", "") for network in response_data["networks"]],
            get(response_data, "number_of_episodes", 0),
            get(response_data, "number_of_seasons", 0),
            get(response_data, "status", ""),
            get(response_data, "popularity", 0.0)
        )
    else
        error("HTTP Request failed with status: $(HTTP.status(response))")
    end
end


all_tv_show_ids = []
for page in 2:2
    tv_show_ids = fetch_tv_show_ids(page)
    append!(all_tv_show_ids, tv_show_ids)
end
sayi = length(all_tv_show_ids)
series = TvSeries[]
for x in 1:sayi
    push!(series, fetch_tv_series(all_tv_show_ids[x]))
end

# Tüm çekilen veriyi sırala
println("Sunucudan alınan tüm diziler:")
for i in eachindex(series)
    println(series[i])
end






# PCA Test
using MultivariateStats
using StatsBase

# Veri setini hazırla (sayısal özellikleri kullan)
numeric_features = hcat([series[i].episode_run_time for i in eachindex(series)],
    [series[i].number_of_episodes for i in eachindex(series)],
    [series[i].number_of_seasons for i in eachindex(series)],
    [series[i].popularity for i in eachindex(series)])

# Veriyi normalize et
#https://youtu.be/ZWyoSZk-Uq0?t=500
# row = feature, column = sample
normalized_features = zscore(numeric_features, 1)
normalized_features = normalized_features'
println("normalized_features:")
display("text/plain", normalized_features)

# PCA modelini oluştur
pca_model = fit(PCA, normalized_features; maxoutdim=2)

# Boyut azaltılmış veriyi al
reduced_features = transform(pca_model, normalized_features)

# Sonucu göster
println("\nBoyut Azaltılmış Veri Seti:")
println(length(reduced_features), " tane indexsi ve $(axes(reduced_features, 2)) tane sütunu var. reduced_features[] array'i (2x20)'lik bir matrix yani loop 20 kere tekrarlanıcak.")
# axes(reduced_features, 2) fonksiyonu sütun sayısını döndürür. (1=satır, 2=sütun, 3...=diğer boyutlar) (axes = eksenler (x,y,z gibi))
for i in axes(reduced_features, 2)
    println("TV Serisi $(i): ", reduced_features[:, i])
end

println("\nreduced_features matrix'i böyle:")
display("text/plain", reduced_features)



#plotlyjs(size = (360, 360))