library(mvtnorm)
library(plot3D)
library(caret)
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

n_dims <- 2
n_class <- 9
n_sample <- 5000


set.seed(10)
m_means <- matrix(runif(n_dims*n_class), c(n_class, n_dims))
m_sigma <- diag(n_dims)
diag(m_sigma) <- 0.01
m_samples <- matrix(runif(n_dims*n_sample), c(n_sample,n_dims))


m_densities <- t(apply(m_samples, MARGIN=1, function(x) 
    { apply(m_means, MARGIN=1, function(m) {dmvnorm(x, m, m_sigma)}) }))
m_labels <- apply(m_densities, 1, which.max)

m_noisy_labels = apply(m_densities, 1, function(p) { apply(rmultinom(1, size = 1, prob = p^8), 2, which.max)})

m_noisy_labels = m_labels
#scatter3D(m_samples[,1], m_samples[,2], apply(m_samples, 1, function(x) {dmvnorm(x, m_means[1,], m_sigma)}))

plot(m_samples, col=m_labels)
plot(m_samples, col=m_noisy_labels)
confusionMatrix(m_noisy_labels, m_labels)
multiLogLoss(m_densities, m_noisy_labels)


allData <- data.frame(m_samples, label=as.factor(paste("C",m_noisy_labels, sep='')))

trainIdx <- createDataPartition(m_noisy_labels, p=0.75, list=FALSE)

trainData <- allData[trainIdx,]
testData <- allData[-trainIdx,]

trControl <- trainControl(method="oob",
                          verboseIter=TRUE)
rfModel <- train(label ~ . , 
                 data = trainData,
                 method = 'rf',
                 ntree = 300,
                 trControl = trControl, 
                 tuneLength=3)


predLabels <- predict(rfModel, newdata = testData, type='raw')
confusionMatrix(predLabels, testData$label)
predProbs <- predict(rfModel, newdata = trainData, type='prob')
multiLogLoss(predProbs,m_noisy_labels[trainIdx])

bst <- xgboost(data = m_samples[trainIdx,], label = m_noisy_labels[trainIdx]-1, max.depth = 2,
               eta = 0.05, nrounds = 2000,objective = "multi:softprob", 
               eval_metric ="merror", 
               verbose=1, num_class=n_class)

pred <- predict(bst, m_samples[trainIdx,])
predMatrix <- t(matrix(pred,n_class, length(trainIdx)))
multiLogLoss(predMatrix, m_noisy_labels[trainIdx])


