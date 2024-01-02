using HTTP, JSON

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
for page in 20:24
    tv_show_ids = fetch_tv_show_ids(page)
    append!(all_tv_show_ids, tv_show_ids)
end
series = TvSeries[]
for x in eachindex(all_tv_show_ids)
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
# Ayrıca Julia'da öncelik sütun'da olmalı. Diğer dillerden farklı olarak bu dilde satırda özellikler ve sütunda örnekler olmalı.
# Bu yüzden bu matrixin transpose'unu alıyoruz.
# Eğer yanlış biliyorsam söyleyin düzeltelim ama böyle program normal çalışıyor.
numeric_features = hcat([series[i].episode_run_time for i in eachindex(series)],
    [series[i].number_of_episodes for i in eachindex(series)],
    [series[i].number_of_seasons for i in eachindex(series)],
    [series[i].popularity for i in eachindex(series)])'
# Veriyi normalize et
#https://youtu.be/ZWyoSZk-Uq0?t=500
# row = feature, column = sample
normalized_features = zscore(numeric_features)
# PCA modelini oluştur
pca_model = fit(PCA, normalized_features; maxoutdim=2)
# Boyut azaltılmış veriyi al
reduced_features = transform(pca_model, normalized_features)
# Sonucu göster
println("\nBoyut Azaltılmış Veri Seti:")
#println(length(reduced_features), " tane indexsi ve $(size(reduced_features, 2)) tane sütunu var. reduced_features[] array'i (3x20)'lik bir matrix yani loop 20 kere tekrarlanıcak.")
# axes(reduced_features, 2) fonksiyonu sütun sayısını döndürür. (1=satır, 2=sütun, 3...=diğer boyutlar) (axes = eksenler (x,y,z gibi))
for i in axes(reduced_features, 2)
    println("TV Serisi $(i): ", reduced_features[:, i])
end
#=
println("\nreduced_features matrix'i böyle:")
display("text/plain", reduced_features)
# Veriyi geri elde etme(?)
reversed_reduced_features = reconstruct(pca_model, reduced_features)
println("\nreversed_reduced_features matrix'i böyle:")
display("text/plain", reversed_reduced_features)
println("\nAsıl (PCA uygulanmamış) matrix böyleydi:")
display("text/plain", normalized_features)
=#
#K-Means Test
using Clustering
using Plots
# Küme sayısı
k = 5
#Model oluştur
kmeans_model = kmeans(reduced_features, k)
#Kümeleri tahmin etme
cluster_assignments = assignments(kmeans_model)

#Küme merkezlerini bulma
cluster_centers = kmeans_model.centers
#Sonuçları print etme
kume1=[]
kume2=[]
kume3=[]
kume4=[]
kume5=[]

println("\nK-Means Kümeleme Sonuçları:")
for i in eachindex(cluster_assignments)
    if(cluster_assignments[i]==1)
        push!(kume1, series[i].number_of_seasons)
    
    elseif(cluster_assignments[i]==2)
        push!(kume2, series[i].number_of_seasons)
    
    elseif(cluster_assignments[i]==3)
        push!(kume3, series[i].number_of_seasons)
    
    elseif(cluster_assignments[i]==4)
        push!(kume4, series[i].number_of_seasons)

    elseif(cluster_assignments[i]==5)
        push!(kume5, series[i].number_of_seasons)
    end
    println("TV Serisi $i küme: ", cluster_assignments[i])
end
#Merkezleri print etme
println("\nKüme Merkezleri:")
for i in axes(cluster_centers, 2)
    println("Küme $i Merkezi: ", cluster_centers[:, i])
end


using Statistics
println("Küme-1:", mean(kume1))
println("Küme-2:", mean(kume2))
println("Küme-3:", mean(kume3))
println("Küme-4:", mean(kume4))
println("Küme-5:", mean(kume5))


#Plot olarak gösterir
scatter(reduced_features[1, :], reduced_features[2, :], color=cluster_assignments, marker=:auto, xlabel="Principal Component 1", ylabel="Principal Component 2", zlabel="Principal Component 3", legend=false)
scatter!(cluster_centers[1, :], cluster_centers[2, :], color=:red, markersize=8, label="Cluster Centers")
# Bazı verilerde hata var, bu yüzden çok dağınık görülüyorlar. Bizim bu hatalı verileri ayıklamamız gerek.
