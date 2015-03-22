print('Hello')
library(doParallel)
detectCores()

data <- read.csv('target/test.csv')
summary(data)
