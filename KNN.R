library(e1071)
library(caret)
library(dplyr)
library(class)

train<-read.csv('train.csv')
test<-read.csv('test.csv')


train[] <- lapply(train, function(x) as.numeric(as.integer(x)))
test[] <- lapply(test, function(x) as.numeric(as.integer(x)))

train$label<-as.factor(train$label)

porcentaje<-1

corte <- sample(nrow(train),nrow(train)*porcentaje)
cortest <- sample(nrow(test),nrow(test)*porcentaje)

train<-train[corte,]
test<-train[cortest,]
nrow(test)
nrow(train)

train[is.na(train)] <- 0
test[is.na(test)] <- 0


modeloSVM_L<-svm(label~., data=train, cost=0.5, kernel="linear")#95%
prediccionL<-predict(modeloSVM_L,newdata=test[,-1])

confusionMatrix(test$label,prediccionL)
