library(e1071)
library(caret)
library(doParallel)



multiLogLoss <-function (data,label) {
#    print("multiLogLoss")
    sumProbs <- apply(data,1, sum)
    classProb <- apply(cbind(as.numeric(label), data),1,function(x) {x[x[1]+1]})
    scaledProb <- classProb/sumProbs
    adjustedProb <- pmax(pmin(scaledProb,1-(1e-15)),1e-15)
    logLoss = - mean(log(adjustedProb))
#    print(logLoss)
    c(LOGLOSS=logLoss)
}

train_data <- read.csv('target/train.csv')
test_data <- read.csv('target/test.csv')

#print('train data')
#summary(train_data)
#print('test data')
#summary(test_data)

set.seed(100)
cov_and_res  = subset(train_data, select=-id)
trainIndex = createDataPartition(cov_and_res$target, p=0.75, list= FALSE)
testSet <- cov_and_res[-trainIndex,]
trainSet <- cov_and_res[trainIndex,]

cl <- makeCluster(4)
registerDoParallel(cl)

models <- foreach(1:32, .combine=cbind, .packages=c("e1071", "caret")) %dopar% {
    sampleIndex = createDataPartition(trainSet$target, p=0.25, list= FALSE)
    trainSample <- trainSet[sampleIndex,]
    svmModel <- svm(target~.,data=trainSample,type="C-classification",cost=5,probability=TRUE, kernel="radial" )
    pred <- predict(svmModel, testSet)
    predProb <- predict(svmModel, newdata=testSet, probability=TRUE)    
    data.frame(predProb=predProb)
}
stopCluster(cl)


losses <- lapply(models, FUN = function(m) {multiLogLoss(attr(m, "probabilities"), testSet$target)[[1]]})
losses
pp <-lapply(models, FUN = function(m) {attr(m, "probabilities")})
ens<- Reduce("+", pp) / length(pp)
multiLogLoss(ens,  testSet$target)

