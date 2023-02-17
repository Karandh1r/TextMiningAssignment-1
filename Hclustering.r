library(imputeTS)
library(tidyr)
library(stringr)
library(NbClust)
library(cluster)
library(mclust)
library(factoextra)
library(amap)


set.seed(786)
file_loc <- "ombddata.csv"
movie_df <- read.csv("ombddata.csv",sep=",",header=FALSE)
names(movie_df) <- c("MovieId","Title","Year","Rated","Released","Runtime","Genre","Director","Writer","Actors","Plot","Language","CountryAwards","Metascore","imdbRating","imdbVotes","BoxOffice")
selected_df <- data.frame(movie_df$Metascore,movie_df$imdbRating)
selected_df$movie_df.Metascore <- str_replace(selected_df$movie_df.Metascore, "N/A", "0")
Record_3D_DF<-selected_df

selected_df$movie_df.Metascore <- as.integer(selected_df$movie_df.Metascore)
selected_df$movie_df.imdbRating <- as.integer(selected_df$movie_df.imdbRating)

dist_mat <- dist(selected_df, method = 'euclidean')
Record_3D_DF_Norm <- as.data.frame(apply(selected_df[,1:2 ], 2,
                                         function(x) (x - min(x))/(max(x)-min(x))))
Dist_norm <- dist(Record_3D_DF_Norm, method = "minkowski", p=2)
kmeans_3D_1 <- NbClust::NbClust(Record_3D_DF_Norm, 
                                min.nc=2, max.nc=5, method="kmeans")
table(kmeans_3D_1$Best.n[1,])

barplot(table(kmeans_3D_1$Best.n[1,]), 
        xlab="Numer of Clusters", ylab="",
        main="Number of Clusters")

fviz_nbclust(Record_3D_DF_Norm, method = "silhouette", 
  FUN = hcut, k.max = 5)

kmeans_3D_1_Result <- kmeans(Record_3D_DF, 2, nstart=25)
print(kmeans_3D_1_Result)
aggregate(Record_3D_DF, by=list(cluster=kmeans_3D_1_Result$cluster), mean)

cbind(selected_df, cluster = kmeans_3D_1_Result$cluster)
fviz_cluster(kmeans_3D_1_Result, selected_df, main="Euclidean")

My_Kmeans_3D_E<-Kmeans(Record_3D_DF_Norm, centers=2 ,method = "euclidean")
fviz_cluster(My_Kmeans_3D_E, selected_df, main="Euclidean")

My_Kmeans_SmallCorp3<-Kmeans(Record_3D_DF_Norm, centers=3 ,method = "spearman")
fviz_cluster(My_Kmeans_SmallCorp3, selected_df, main="Spearman", repel = TRUE)

My_Kmeans_SmallCorp4<-Kmeans(Record_3D_DF_Norm, centers=4 ,method = "spearman")
fviz_cluster(My_Kmeans_SmallCorp4,selected_df, main="Spearman", repel = TRUE)


Novels_DF_DT <- as.data.frame(as.matrix(Record_3D_DF_Norm))
My_novels_m <- (as.matrix(Novels_DF_DT))
CosineSim <- My_novels_m / sqrt(rowSums(My_novels_m * My_novels_m))
CosineSim <- na.omit(CosineSim)
CosineSim <- CosineSim %*% t(CosineSim)
D_Cos_Sim <- as.dist(1-CosineSim)

HClust_Ward_CosSim_SmallCorp2 <- hclust(D_Cos_Sim, method="ward.D2")
plot(HClust_Ward_CosSim_SmallCorp2, cex=.7, hang=-11,main = "Cosine Sim")
rect.hclust(HClust_Ward_CosSim_SmallCorp2, k=4)