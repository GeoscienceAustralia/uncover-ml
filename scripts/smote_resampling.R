library('rgdal')
library('UBL')
library('MASS')
library('ks')
library('ggplot2')


shp <- readOGR(dsn="/home/sudipta/Documents/nci/0_50centimeters_T_full_group_v1.shp")
data <- shp@data
data$longitude <- shp$coords.x1
data$latitude <- shp$coords.x2


# using a relevance function provided by the user
rel <- matrix(0, ncol = 3, nrow = 0)
rel <- rbind(rel, c(0., 0.1, 0))
rel <- rbind(rel, c(0.5, 1, 0))
rel <- rbind(rel, c(1, 1, 0))
rel <- rbind(rel, c(1.5, 1, 0))
rel <- rbind(rel, c(2, 1, 0))
rel <- rbind(rel, c(3, 1, 0))
rel <- rbind(rel, c(4, 1, 0))
rel <- rbind(rel, c(5, 1, 0))
rel <- rbind(rel, c(6, 1, 0))
rel <- rbind(rel, c(7, 1, 0))
rel <- rbind(rel, c(8, 1, 0))
rel <- rbind(rel, c(9, 1, 0))
rel <- rbind(rel, c(10, 1, 0))



smoteBalan.shp <- SmoteRegress(con~., data, rel=rel, dist = "HEOM", C.perc = "extreme")

# check number of points in smote output
dim(smoteBalan.shp)

# plot(sort(data$con))
# plot(sort(smoteBalan.shp$con))

coordinates(smoteBalan.shp)=~longitude+latitude
proj4string(smoteBalan.shp)<- shp@proj4string

# write smote.shp on disc
writeOGR(smoteBalan.shp, dsn=".", layer="smote", driver="ESRI Shapefile")

# plot histograms
ggplot(smoteBalan.shp@data, aes(x=con)) + geom_histogram()
ggplot(data, aes(x=con)) + geom_histogram()
