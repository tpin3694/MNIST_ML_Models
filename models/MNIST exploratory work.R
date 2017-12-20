train<-read.csv("train.csv")
class(train$label)
train$label <- as.factor(train$label)


train8 <- train[train$label == 8, ]
flip <- function(matrix){
  apply(matrix, 2, rev)
}


par(mfrow=c(1,5))
for (i in 500:504){
  digit <- flip(matrix(rev(as.numeric(train8[i,-c(1, 786)])), nrow = 28)) #look at one digit
  image(digit, col = grey.colors(255))
}


train <- read.csv("train.csv", header=TRUE)
train<-as.matrix(train)

colors<-c('gray33','white')
cus_col<-colorRampPalette(colors=colors)

par(mfrow=c(2,5),pty='s',mar=c(1,1,1,1),xaxt='n',yaxt='n')
all_img<-array(dim=c(10,28*28))
for(di in 0:9)
{
  print(di)
  all_img[di+1,]<-apply(train[train[,1]==di,-1],2,sum)
  all_img[di+1,]<-all_img[di+1,]/max(all_img[di+1,])*255
  
  z<-array(all_img[di+1,],dim=c(28,28))
  z<-z[,28:1] ##right side up
  image(1:28,1:28,z,main=di,col=cus_col(256))
}

