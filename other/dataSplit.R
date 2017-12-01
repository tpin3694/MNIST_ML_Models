setwd("/home/tpin3694/Documents/university/MSc/dataMining/group_work/emotion_recognition/data/mnist/tsne/")
digits <- read.csv("tsne3D.csv")

smp_size <- floor(0.7*nrow(digits))
set.seed(123)

train_ind <- sample(seq_len(nrow(digits)), size = smp_size)
train <- digits[train_ind, ]
test <- digits[-train_ind, ]

write.csv(train, "3d_train.csv", row.names = F)
write.csv(test, "3d_test.csv", row.names = F)