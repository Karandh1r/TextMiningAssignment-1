library(rlang)
library(usethis)
library(devtools)
library(base64enc)
library(RCurl)
library(tokenizers)
library(dplyr)
library(arules)
library(arulesViz)


movie_plot = 'ombddata.csv'
movies_df <- read.csv('ombddata.csv',header=TRUE,sep=",")
MovieDF <- read.csv(movie_plot,header = FALSE, sep = ",")
MovieDF<-MovieDF %>%
  mutate_all(as.character)
(str(MovieDF))

MovieDF[MovieDF == "t.co"] <- ""
MovieDF[MovieDF == "rt"] <- ""
MovieDF[MovieDF == "http"] <- ""
MovieDF[MovieDF == "https"] <- ""

MyDF<-NULL
MyDF2<-NULL
for (i in 1:ncol(MovieDF)){
  MyList=c() 
  MyList2=c()
  MyList=c(MyList,grepl("[[:digit:]]", MovieDF[[i]]))
  MyDF<-cbind(MyDF,MyList)
  MyList2=c(MyList2,(nchar(MovieDF[[i]])<4 | nchar(MovieDF[[i]])>11))
  MyDF2<-cbind(MyDF2,MyList2) 
}
MovieDF[MyDF] <- ""
MovieDF[MyDF2] <- ""
(head(MovieDF,10))

write.table(MovieDF, file = "UpdatedMovieFile.csv", col.names = FALSE, 
            row.names = FALSE, sep = ",")
MovieTrans <- read.transactions("UpdatedMovieFile.csv", sep =",", 
                                format("basket"),  rm.duplicates = TRUE)



MovieTrans_rules = arules::apriori(MovieTrans, 
                                   parameter = list(support=.05, conf=1, minlen=2))

View(MovieTrans_rules)
SortedRules_conf <- sort(MovieTrans_rules, by="confidence", decreasing=TRUE)
inspect(SortedRules_conf[1:15])

SortedRules_sup <- sort(MovieTrans_rules, by="support", decreasing=TRUE)
inspect(SortedRules_sup[1:15])

SortedRules_lift <- sort(MovieTrans_rules, by="lift", decreasing=TRUE)
inspect(SortedRules_lift[1:15])

print(SortedRules_sup[1:30])
plot(SortedRules_sup[1:37],method="graph",engine='interactive', shading="confidence") 
plot(SortedRules_conf[1:37],method="graph",engine='interactive',shading="confidence")

Rules_DF2<-DATAFRAME(MovieTrans_rules, separate = TRUE)
(head(Rules_DF2))
str(Rules_DF2)

Rules_DF2$LHS<-as.character(Rules_DF2$LHS)
Rules_DF2$RHS<-as.character(Rules_DF2$RHS)

Rules_DF2[] <- lapply(Rules_DF2, gsub, pattern='[{]', replacement='')
Rules_DF2[] <- lapply(Rules_DF2, gsub, pattern='[}]', replacement='')

## USING LIFT
Rules_L<-Rules_DF2[c(1,2,5)]
names(Rules_L) <- c("SourceName", "TargetName", "Weight")
head(Rules_L,30)

## USING SUP
Rules_S<-Rules_DF2[c(1,2,3)]
names(Rules_S) <- c("SourceName", "TargetName", "Weight")
head(Rules_S,30)

## USING CONF
Rules_C<-Rules_DF2[c(1,2,4)]
names(Rules_C) <- c("SourceName", "TargetName", "Weight")
head(Rules_C,30)

Rules_Sup<-Rules_L


############################### BUILD THE NODES & EDGES ####################################
(edgeList<-Rules_Sup)
(MyGraph <- igraph::simplify(igraph::graph.data.frame(edgeList, directed=TRUE)))

nodeList <- data.frame(ID = c(0:(igraph::vcount(MyGraph) - 1)), 
                       # because networkD3 library requires IDs to start at 0
                       nName = igraph::V(MyGraph)$name)
## Node Degree
(nodeList <- cbind(nodeList, nodeDegree=igraph::degree(MyGraph, 
                                                       v = igraph::V(MyGraph), mode = "all")))

## Betweenness
BetweenNess <- igraph::betweenness(MyGraph, 
                                   v = igraph::V(MyGraph), 
                                   directed = TRUE) 

(nodeList <- cbind(nodeList, nodeBetweenness=BetweenNess))

## This can change the BetweenNess value if needed
#BetweenNess<-BetweenNess/100

getNodeID <- function(x){
  which(x == igraph::V(MyGraph)$name) - 1  #IDs start at 0
}
## UPDATE THIS !! depending on # choice
(getNodeID("salary")) 

edgeList <- plyr::ddply(
  Rules_Sup, .variables = c("SourceName", "TargetName" , "Weight"), 
  function (x) data.frame(SourceID = getNodeID(x$SourceName), 
                          TargetID = getNodeID(x$TargetName)))

head(edgeList)
nrow(edgeList)


D3_network_Tweets <- networkD3::forceNetwork(
  Links = edgeList, # data frame that contains info about edges
  Nodes = nodeList, # data frame that contains info about nodes
  Source = "SourceID", # ID of source node 
  Target = "TargetID", # ID of target node
  Value = "Weight", # value from the edge list (data frame) that will be used to value/weight relationship amongst nodes
  NodeID = "nName", # value from the node list (data frame) that contains node description we want to use (e.g., node name)
  Nodesize = "nodeBetweenness",  # value from the node list (data frame) that contains value we want to use for a node size
  Group = "nodeDegree",  # value from the node list (data frame) that contains value we want to use for node color
  height = 700, # Size of the plot (vertical)
  width = 900,  # Size of the plot (horizontal)
  fontSize = 20, # Font size
  linkDistance = networkD3::JS("function(d) { return d.value*1000; }"), # Function to determine distance between any two nodes, uses variables already defined in forceNetwork function (not variables from a data frame)
  linkWidth = networkD3::JS("function(d) { return d.value*5; }"),# Function to determine link/edge thickness, uses variables already defined in forceNetwork function (not variables from a data frame)
  opacity = 5, # opacity
  zoom = TRUE, # ability to zoom when click on the node
  opacityNoHover = 5, # opacity of labels when static
  linkColour = "red"   ###"edges_col"red"# edge colors
) 

# Plot network
#D3_network_Tweets

# Save network as html file
networkD3::saveNetwork(D3_network_Tweets, 
                       "NetD3_DCR2019_worldNewsL_2021.html", selfcontained = TRUE)