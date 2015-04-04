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
trainIndex = createDataPartition(cov_and_res$target, p=0.75, list= FALSE)

trainSet <- cov_and_res[trainIndex,]
testSet <- cov_and_res[-trainIndex,]


trainMatrix <- as.matrix(trainSet[,1:93])
storage.mode(trainMatrix) <- "double"

bst <- xgboost(data = trainMatrix, label = as.numeric(trainSet$target)-1, max.depth = 10,  nthread=4,
               eta = 0.05, nrounds = 600,objective = "multi:softprob", 
               eval_metric ="mlogloss", 
               verbose=1, num_class=length(levels(trainSet$target)))

testMatrix <- as.matrix(testSet[,1:93])
storage.mode(testMatrix) <- "double"

pred <- predict(bst, testMatrix)
predMatrix <- t(matrix(pred, 9, nrow(testMatrix)))
multiLogLoss(predMatrix, testSet$target)


