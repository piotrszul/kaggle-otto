library(caret)
library(doMPI)

cl <- startMPIcluster(count=32)
registerDoMPI(cl)

multiLogLoss <-function (data, lev = NULL, model = NULL) {
    print("multiLogLoss")
    sumProbs <- apply(subset(data, select=-c(pred,obs)),1, sum)
    classProb <- apply(data,1,function(x) {as.numeric(x[as.numeric(substring(x[2], nchar(x[2]))) + 2])})
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

#trainSet <- cov_and_res[trainIndex,]
trainSet <- cov_and_res

tuneGrid <- expand.grid(n.trees = c(300,600), interaction.depth = c(5,10), shrinkage = c(0.1,0.05))
trControl <- trainControl(method="cv", number=10, 
                          verboseIter=TRUE,
                          summaryFunction = multiLogLoss,
                          classProbs = TRUE,                           
                          allowParallel = TRUE)

gbmModel <- train(target ~ ., 
                  data=trainSet, method="gbm",
                  metric = "LOGLOSS",
                  maximize = FALSE,
                  trControl = trControl,
                  tuneGrid = tuneGrid 
)

print(gbmModel$bestTune)
closeCluster(cl)

test_pred <- predict(gbmModel, newdata = test_data)
test_prob_pred <- predict(gbmModel, newdata = test_data, type='prob')
final_prob <- data.frame(id = test_data$id, test_prob_pred)
write.csv(final_prob,"target/subtmit-prob.csv", row.names=FALSE, quote=FALSE )



