library(lubridate)
library(Hmisc)
library(caret)
library(ggplot2)
library(plyr)
library(doParallel)

train_data <- read.csv('train.csv')
test_data <- read.csv('test.csv')
set.seed(100)
cl <- makeCluster(3)
registerDoParallel(cl)
cov_and_res  = subset(train_data, select=-id)
trainIndex = createDataPartition(cov_and_res$target, p=0.75, list= FALSE)

#trainSet <- cov_and_res[trainIndex,]
trainSet <- cov_and_res

tuneGrid <- expand.grid(n.trees = c(300,600), interaction.depth = c(1,5,10), shrinkage = c(0.1,0.05,0.01))
trControl <- trainControl(method="cv", number=3, 
                          verboseIter=TRUE,
                          allowParallel = TRUE)
gbmModel <- train(target ~ ., 
                  data=trainSet, method="gbm",
                  trControl = trControl,
                  tuneGrid = tuneGrid
)

print(gbmModel$bestTune)

test_pred <- predict(gbmModel, newdata = test_data)
test_prob_pred <- predict(gbmModel, newdata = test_data, type='prob')
final_prob <- data.frame(id = test_data$id, test_prob_pred)
write.csv(final_prob,"subtmit-prob.csv", row.names=FALSE, quote=FALSE )

stopCluster(cl)

