library(caret)
library(doParallel)

train_data <- read.csv('target/train.csv')
test_data <- read.csv('target/test.csv')

print('train data')
summary(train_data)
print('test data')
summary(test_data)


set.seed(100)
cl <- makeCluster(16)
registerDoParallel(cl)
cov_and_res  = subset(train_data, select=-id)
trainIndex = createDataPartition(cov_and_res$target, p=0.75, list= FALSE)

#trainSet <- cov_and_res[trainIndex,]
trainSet <- cov_and_res

tuneGrid <- expand.grid(n.trees = c(200), interaction.depth = c(5,10), shrinkage = c(0.1,0.05))
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
write.csv(final_prob,"target/subtmit-prob.csv", row.names=FALSE, quote=FALSE )

stopCluster(cl)

