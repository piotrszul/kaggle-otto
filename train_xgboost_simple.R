library(xgboost)
library(caret)


multiLogLoss <-function (data,label) {
    print("multiLogLoss")
    sumProbs <- apply(data,1, sum)
    classProb <- apply(cbind(as.numeric(label), data),1,function(x) {x[x[1]+1]})
    scaledProb <- classProb/sumProbs
    adjustedProb <- pmax(pmin(scaledProb,1-(1e-15)),1e-15)
    logLoss = - mean(log(adjustedProb))
    print(logLoss)
    c(LOGLOSS=logLoss)
}

train_data <- read.csv('target/train.csv')
test_data <- read.csv('target/test.csv')

print('train data')
summary(train_data)
print('test data')
summary(test_data)

set.seed(100)
cov_and_res  = subset(train_data, select=-id)

trainSet <- cov_and_res
trainMatrix <- as.matrix(trainSet[,1:93])
storage.mode(trainMatrix) <- "double"
trainData <- xgb.DMatrix(trainMatrix, label = as.numeric(trainSet$target)-1)
params <- list(max.depth = 14,  nthread=16, subsample = 0.75, colsample_bytree = 0.5,
           eta = 0.01, min_child_weight=6,  objective = "multi:softprob", 
           eval_metric ="mlogloss", 
           num_class=length(levels(trainSet$target)))

nrounds <-2246
bst <- xgb.train(params = params, data = trainData,nrounds = nrounds, 
                 verbose=1,
                 watchlist=list(train=trainData))

test_cov_and_res  = subset(test_data, select=-id)
testSet <- test_cov_and_res
testMatrix <- as.matrix(testSet[,1:93])
storage.mode(testMatrix) <- "double"
pred <- predict(bst, testMatrix)
predMatrix <- t(matrix(pred, 9, nrow(testMatrix)))
final_prob <- data.frame(id = test_data$id, predMatrix)
names(final_prob) <-c("id",paste("Class_", 1:9, sep=''))
write.csv(final_prob,"target/subtmit-prob-xgboost.csv", row.names=FALSE, quote=FALSE )


