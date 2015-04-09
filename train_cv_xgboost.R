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

trainData <- xgb.DMatrix(trainMatrix, label = as.numeric(trainSet$target)-1)
testMatrix <- as.matrix(testSet[,1:93])
storage.mode(testMatrix) <- "double"
testData <-  xgb.DMatrix(testMatrix, label = as.numeric(testSet$target)-1)
params <- list(max.depth = 6,  nthread=16,
           eta = 0.05, objective = "multi:softprob", subsample=0.75, colsample_bytree=0.3, 
           eval_metric ="mlogloss", 
           num_class=length(levels(trainSet$target)))

bst <- xgb.train(params = params, data = trainData,nrounds = 2500, 
                 verbose=1,
                 watchlist=list(train=trainData, test= testData))

testMatrix <- as.matrix(testSet[,1:93])
storage.mode(testMatrix) <- "double"

pred <- predict(bst, testMatrix)
predMatrix <- t(matrix(pred, 9, nrow(testMatrix)))
multiLogLoss(predMatrix, testSet$target)


