library(lubridate)
library(Hmisc)
library(caret)
library(ggplot2)
library(plyr)

train_data <- read.csv('train.csv')
test_data <- read.csv('test.csv')
set.seed(100)

cov_and_res  = subset(train_data, select=-id)
trainIndex = createDataPartition(cov_and_res$target, p=0.75, list= FALSE)

#trainSet <- cov_and_res[trainIndex,]
trainSet <- cov_and_res
trControl <- trainControl(method="oob", verboseIter=TRUE)
gbmModel <- train(target ~ ., 
                  data=trainSet, method="rf",
                  importance = TRUE,
                  ntree = 200,
                  trControl = trControl,
                  tuneLength = 1
)

test_pred <- predict(gbmModel, newdata = test_data)
result <-lapply(1:9, FUN=function(x) {as.numeric(as.numeric(test_pred) == x)})
names(result) = lapply(1:9, FUN = function(i) {paste('Class_', i, sep='')})
final <- data.frame(id = test_data$id, result)
write.csv(final,"subtmit.csv", row.names=FALSE, quote=FALSE )
test_prob_pred <- predict(gbmModel, newdata = test_data, type='prob')
final_prob <- data.frame(id = test_data$id, test_prob_pred)
write.csv(final_prob,"subtmit-prob.csv", row.names=FALSE, quote=FALSE )


