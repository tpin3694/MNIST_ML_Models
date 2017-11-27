install.packages("Rtsne")
install.packages("rgl")
library(rgl)
library(Rtsne)
#2D
train<- read.csv("train.csv") 
class(train$label)

Labels<-train$label
train$label<-as.factor(train$label)

colours = rainbow(length(unique(train$label)))
names(colours) = unique(train$label)

tsne <- Rtsne(train[,-1], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)

plot(tsne$Y, t='n', main="tsne")
text(tsne$Y, labels=train$label, col=colours[train$label])

train2D<-cbind(train[,1:784],tsne$Y,train[,785])

#3D
tsne3d <- Rtsne(train[,-1], dims = 3, perplexity=30, verbose=TRUE, max_iter = 500)
plot3d(tsne3d$Y, col=colours[train$label])


