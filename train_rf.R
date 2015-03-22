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
set.seed(100)

cov_and_res  = subset(train_data, select=-id)
trainIndex = createDataPartition(cov_and_res$target, p=0.95, list= FALSE)

#trainSet <- cov_and_res[-trainIndex,]
trainSet <- cov_and_res
trControl <- trainControl(method="cv",
                          number=10,
                    verboseIter=TRUE,
                    summaryFunction = multiLogLoss,
                    classProbs = TRUE, 
                    allowParallel = FALSE)

gbmModel <- train(target ~ ., 
                  data=trainSet, method="rf",
                  importance = TRUE,
                  ntree = 500,
                  trControl = trControl,
                  metric = "LOGLOSS",
                  maximize = FALSE,
                  tuneGrid = data.frame(mtry=c(18,20,25))
)

print(gbmModel$results)
print(gbmModel$bestTune)
closeCluster(cl)
#test_pred <- predict(gbmModel, newdata = test_data)
#result <-lapply(1:9, FUN=function(x) {as.numeric(as.numeric(test_pred) == x)})
#names(result) = lapply(1:9, FUN = function(i) {paste('Class_', i, sep='')})
#final <- data.frame(id = test_data$id, result)
#write.csv(final,"subtmit.csv", row.names=FALSE, quote=FALSE )
test_prob_pred <- predict(gbmModel, newdata = test_data, type='prob')
final_prob <- data.frame(id = test_data$id, test_prob_pred)
write.csv(final_prob,"target/subtmit-prob-rf.csv", row.names=FALSE, quote=FALSE )


