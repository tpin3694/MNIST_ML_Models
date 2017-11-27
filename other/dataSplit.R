digits <- read.csv("Documents/university/MSc/dataMining/group_work/emotion_recognition/data/mnist/train.csv")


smp_size <- floor(0.7*nrow(digits))
set.seed(123)

train_ind <- sample(seq_len(nrow(digits)), size = smp_size)
train <- digits[train_ind, ]
test <- digits[-train_ind, ]

write.csv(train, "Documents/university/MSc/dataMining/group_work/emotion_recognition/data/mnist/split/train.csv",
          row.names = F)
write.csv(test, "Documents/university/MSc/dataMining/group_work/emotion_recognition/data/mnist/split/test.csv",
          row.names = F)