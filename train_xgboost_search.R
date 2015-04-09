library(xgboost)

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


taskId <- as.numeric(commandArgs(trailingOnly=TRUE)[1])
if (is.na(taskId)) {
    taskId <- 1
}

nrounds <- 3000
nfold <- 3

train_data <- read.csv('target/train.csv')
set.seed(100)
cov_and_res  = subset(train_data, select=-id)
trainSet <- cov_and_res
trainMatrix <- as.matrix(trainSet[,1:93])
storage.mode(trainMatrix) <- "double"
trainData <- xgb.DMatrix(trainMatrix, label = as.numeric(trainSet$target)-1)
#commandArgs

paramGrid <- expand.grid(max.depth=c(13,15,17), eta=c(0.01), subsample=c(0.75), colsample_bytree=c(0.5), min_child_weight=c(5,6,7))
#paramGrid <- expand.grid(max.depth=c(8, 10, 12), eta=c(0.01,0.008), subsample=c(1,0.75,0.5), colsample_bytree=c(1,0.75,0.5))
str(paramGrid)
taskParams <- as.list(paramGrid[taskId,])

print("Running task")
print(taskId)
print(taskParams)


techParams <- list( nthread=16,
            objective = "multi:softprob", 
            eval_metric ="mlogloss", 
            num_class=length(levels(trainSet$target)))

params <- as.list(data.frame(taskParams, techParams, stringsAsFactors=FALSE))
cv <-xgb.cv(params = params, data = trainData,nrounds = nrounds, nfold = nfold)
min_nrounds <- which.min(as.numeric(cv$test.mlogloss.mean))

result = data.frame(logloss = as.numeric(cv$test.mlogloss.mean[min_nrounds]), 
                    min_nrounds = min_nrounds, taskParams, id=taskId)         
print(result)
write.table(result, paste('target/task/out_', taskId, sep=''), row.names=FALSE, quote=FALSE, col.names=FALSE)




