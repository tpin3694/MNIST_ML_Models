tsne_3d <- read.csv("Downloads/tsne/tsne3d.csv")
library(dplyr)
colnames(tsne_3d)
tsne_2d_write <- tsne_3d %>% 
  select(label, X1, X2, X3)
write.csv(tsne_2d_write,
          "Documents/university/MSc/dataMining/group_work/emotion_recognition/data/mnist/tsne/train2d.csv",
          row.names = F)
