rm(list=ls())
####################################################
### Functions
####################################################
installIfAbsentAndLoad <- function(neededVector) {
  for(thispackage in neededVector) {
    if( ! require(thispackage, character.only = T) )
    { install.packages(thispackage)}
    require(thispackage, character.only = T)
  }
}
##############################
### Load required packages ###
##############################
needed <- c('dplyr','plyr', 'glmnet', 'e1071', 'boot', 'ISLR', 'xgboost', 'ada', 'gbm', 'Hmisc', 'corrplot', 'verification', 'randomForest', 'rpart', 'factoextra', 'caret')  
installIfAbsentAndLoad(needed)
##########################
####    Setup         ####        
##########################
set.seed(5082)
# Reading the raw dataset
my.df.full <- read.table("online_shoppers_intention.csv", header=T, sep=',')
sum(is.na(my.df.full))

my.df <- my.df.full[sample(nrow(my.df.full), 5000), ]
str(my.df)
summary(my.df)

my.df$Month <- factor(my.df$Month)
my.df$SpecialDay <- factor(my.df$SpecialDay)
my.df$OperatingSystems <- factor(my.df$OperatingSystems)
my.df$Browser <- factor(my.df$Browser)
my.df$Region <- factor(my.df$Region)
my.df$TrafficType <- factor(my.df$TrafficType)
my.df$VisitorType <- factor(my.df$VisitorType)
my.df$Weekend <- factor(my.df$Weekend)
my.df$Revenue <- factor(my.df$Revenue)
summary(my.df)

hist(my.df$Administrative)
hist(my.df$Administrative_Duration)
hist(my.df$Informational)
hist(my.df$Informational_Duration)
hist(my.df$ProductRelated)
hist(my.df$ProductRelated_Duration)
hist(my.df$BounceRates)
hist(my.df$ExitRates)
hist(my.df$PageValues)


# my.df[, 1:9] <- sign(my.df[, 1:9]) * abs(my.df[, 1:9])^(1/3)
my.df[, 1:9] <- log(my.df[,1:9] + 1)

hist(my.df$Administrative)
hist(my.df$Administrative_Duration)
hist(my.df$Informational)
hist(my.df$Informational_Duration)
hist(my.df$ProductRelated)
hist(my.df$ProductRelated_Duration)
hist(my.df$BounceRates)
hist(my.df$ExitRates)
hist(my.df$PageValues)
summary(my.df)
str(my.df)
sum(is.na(my.df))


n <- nrow(my.df)
my.mm <- model.matrix(Revenue ~ ., my.df)[, -1]

my.mm <- scale(my.mm)
sum(is.na(my.mm))

train.indices.credit <- sample(n, .8 * n)
train.x <- my.mm[train.indices.credit, ]
test.x <- my.mm[-train.indices.credit, ]
train.y <- my.df$Revenue[train.indices.credit]
test.y <- my.df$Revenue[-train.indices.credit]
train.data <- data.frame(train.x, Revenue=train.y)
test.data <- data.frame(test.x, Revenue=test.y)


trainset <- my.df[train.indices.credit, ]
testset <- my.df[-train.indices.credit, ]
sum(is.na(train.x))

corr <- rcorr(my.mm)
corrplot(corr$r)

pca.df <- data.frame(my.mm)
sum(is.na(pca.df))
######################
####      PCA     ####        
######################
pca.fit <- prcomp(pca.df)
summary(pca.fit)

screeplot(pca.fit, type = "l", npcs = 100, main = "Screeplot of the first 100 PCs", ylim=c(0,4))
abline(h = 1, col="red", lty=5)
legend("topright", legend=c("Eigenvalue = 1"),
       col=c("red"), lty=5, cex=0.6)

######################
####    Lasso     ####        
######################
set.seed(5082)
grid = 10 ^ seq(10, -3, length=100)
sum(is.na(train.x))
mod.lasso <- glmnet(train.x, 
                    train.y, 
                    alpha=1, 
                    lambda=grid, family = 'binomial')
cv.out.class <- cv.glmnet(train.x, 
                          train.y, 
                          alpha=1, 
                          lambda=grid, family = 'binomial')
plot(cv.out.class)
(bestlam <- cv.out.class$lambda.min)

lasso.pred = predict(mod.lasso, 
                     newx=test.x, 
                     s=bestlam,
                     type='class')
(lassoTestTable <- table(actual = test.y, predicted = lasso.pred))
(lasso.test.error <- (lassoTestTable[1, 2] + lassoTestTable[2, 1])/sum(lassoTestTable))#0.118
(lassotype1 <- lassoTestTable[1,2]/sum(lassoTestTable[1,]))#0.05158265
(lassotype2 <- lassoTestTable[2,1]/sum(lassoTestTable[2,]))#0.5034014
(lassopower <- lassoTestTable[2,2]/sum(lassoTestTable[2,]))#0.4965986
(lasso.coefficients <- predict(mod.lasso, 
                               type="coefficients", 
                               s=bestlam))

lasso.prob = predict(mod.lasso, 
                     newx=test.x, 
                     s=bestlam,
                     type='response')
lasso.aucc <- roc.area(as.integer(test.data$Revenue)-1,lasso.prob)
lasso.aucc$A#0.9015799
roc.plot(as.integer(test.data$Revenue)-1,lasso.prob,main = "ROC Curve of Lasso")
legend("bottomright", bty="n",
       sprintf("AUC = %1.3f", lasso.aucc$A))

#'1-4','6-7' and 'specialday', 'weekend' are shrinkaged to 0.

train.datarm <- train.data[,c(-1, -2,-3,-4, -6,-7,-10,-11,-12,-13,-14,-69)]
test.datarm <- test.data[,c(-1, -2,-3,-4, -6,-7,-10,-11,-12,-13,-14,-69)]
trainsetrm <- trainset[,c(-1,-2,-3,-4, -6,-7, -10, -17)]
testsetrm <- testset[,c(-1,-2,-3,-4, -6,-7, -10, -17)]
##########################
####      GLM        #####        
##########################
#fit glm with all variables
#create a confusion matrix and plot ROC
summary(train.data)
set.seed(5082)
glm.all.fit <- glm(Revenue ~.,train.datarm,family = binomial)
summary(glm.all.fit)
glm.all.probs <- predict(glm.all.fit,test.datarm,type = "response")
glm.all.pred <- rep("0",nrow(testsetrm))
glm.all.pred[glm.all.probs>0.5] <- "TRUE"
glmalltable <- table(testsetrm$Revenue,glm.all.pred)
(glm.allerror <- (glmalltable[2]+glmalltable[3])/sum(glmalltable))#0.128
glmalltype1 <- glmalltable[1,2]/sum(glmalltable[1,])
glmalltype2 <- glmalltable[2,1]/sum(glmalltable[2,])
glmallpower <- glmalltable[2,2]/sum(glmalltable[2,])
glmalltable
glmalltype1#0.05627198
glmalltype2#0.5442177
testactual <- rep(FALSE,nrow(testsetrm))
testactual[test.datarm$Revenue=="TRUE"] <- 1
roc.plot(testactual,glm.all.probs,main = "ROC Curve of Logistic Regression")
glm.roc <- roc.area(testactual,glm.all.probs)
glm.roc$A #0.8856776


##########################
###Classification Tree####        
##########################
set.seed(5082)
rpart.fit <- rpart(Revenue ~ .,train.datarm, method = "class", parms = list(split = "information"), 
                   control = rpart.control(usesurrogate = 0, maxsurrogate = 0))
xerr <- rpart.fit$cptable[,"xerror"]
minxerr <- which.min(xerr)
mincp <- rpart.fit$cptable[minxerr,"CP"]
rpart.prune <- prune(rpart.fit,cp = mincp)


pred.tree <- predict(rpart.prune, test.datarm, type = "class")
pred.tree.prob <- predict(rpart.prune, test.datarm, type = "prob")
acc.tree <- mean(test.datarm$Revenue == pred.tree)
table.tree<-table(test.datarm$Revenue, pred.tree ,dnn = c("Actual", "Predicted"))

print(acc.tree) #0.891
(tree.error <- 1-acc.tree)#0.109
table.tree
(treetype1 <- table.tree[1,2]/sum(table.tree[1,]))#0.06447831
(treetype2 <- table.tree[2,1]/sum(table.tree[2,]))#0.3673469
(treepower <- table.tree[2,2]/sum(table.tree[2,]))#0.6326531
ct.aucc <- roc.area(as.integer(test.datarm$Revenue)-1,pred.tree.prob[,2])
ct.aucc$A#0.8630484
roc.plot(as.integer(test.datarm$Revenue)-1,pred.tree.prob[,2],main = "ROC Curve of Classification Tree")
legend("bottomright", bty="n",
       sprintf("AUC = %1.3f", ct.aucc$A))
##########################
####      SVM        #####        
##########################
set.seed(5082)
linear.tune.out <- tune(svm, Revenue ~ .,  data = trainsetrm, kernel="linear", 
                 ranges=list(cost=c(0.01, 0.1, 0.5, 1)))
summary(linear.tune.out$best.model)

linear.ypred <- predict(linear.tune.out$best.model, testsetrm)
(linear.testTuneOptimal <- table(actual = testsetrm$Revenue, predicted = linear.ypred))
(linear.svm.test.error <- (linear.testTuneOptimal[1, 2] + linear.testTuneOptimal[2, 1])/sum(linear.testTuneOptimal))#0.121
(lineartype1 <- linear.testTuneOptimal[1,2]/sum(linear.testTuneOptimal[1,]))#0.07033998
(lineartype2 <- linear.testTuneOptimal[2,1]/sum(linear.testTuneOptimal[2,]))#0.414966
(linearpower <- linear.testTuneOptimal[2,2]/sum(linear.testTuneOptimal[2,]))#0.585034
linear.tune.out$best.parameters



linearsvm.opt <- svm(Revenue~., data=train.datarm, 
                  kernel="linear", cost=linear.tune.out$best.parameters,
                  decision.values=T, scale = F)

linearsvm.fitted <- attributes(predict(linearsvm.opt,
                             test.datarm,
                             decision.values=T))$decision.values

linearsvm.aucc <- roc.area(as.integer(as.factor(test.datarm$Revenue))-1,-linearsvm.fitted)
linearsvm.aucc$A#0.8722396
roc.plot(as.integer(as.factor(test.datarm$Revenue))-1,-linearsvm.fitted, main = "ROC Curve of Linear Kernel SVM")
legend("bottomright", bty="n",
       sprintf("AUC = %1.3f", linearsvm.aucc$A))



set.seed(5082)#The longest execution time
radial.tune.out <- tune(svm, Revenue~., data = trainsetrm, kernel="radial", 
                 ranges=list(cost=c(0.01, 0.1, 1, 5),
                             gamma=c(0.001, 0.01, 0.1, 0.5, 1)))
summary(radial.tune.out$best.model)
radial.ypred <- predict(radial.tune.out$best.model, testsetrm)
(radial.testTuneOptimal <- table(actual = testsetrm$Revenue, predicted = radial.ypred))
(radial.svm.test.error <- (radial.testTuneOptimal[1, 2] + radial.testTuneOptimal[2, 1])/sum(radial.testTuneOptimal))#0.11
(radialtype1 <- radial.testTuneOptimal[1,2]/sum(radial.testTuneOptimal[1,]))#0.05627198
(svmtype2 <- radial.testTuneOptimal[2,1]/sum(radial.testTuneOptimal[2,]))#0.4217687
(svmpower <- radial.testTuneOptimal[2,2]/sum(radial.testTuneOptimal[2,]))#0.5782313


radial.tune.out$best.parameters
radialsvm.opt <- svm(Revenue~., data=train.datarm, 
                  kernel="linear",
                  gamma=radial.tune.out$best.parameters[2] , cost=radial.tune.out$best.parameters[1],
                  decision.values=T, scale = F)

radial.fitted <- attributes(predict(radialsvm.opt,
                             test.datarm,
                             decision.values=T))$decision.values

radial.aucc <- roc.area(as.integer(as.factor(test.datarm$Revenue))-1,-radial.fitted)
radial.aucc$A#0.8722396
roc.plot(as.integer(as.factor(test.datarm$Revenue))-1,-radial.fitted, main="ROC Curve of Radial Kernel SVM")
legend("bottomright", bty="n",
       sprintf("AUC = %1.3f", radial.aucc$A))




##########################
####  Random Forest  #####        
##########################
set.seed(5082)
rf <- randomForest(Revenue ~ .,data=trainset,ntree=500, mtry=4,
                   importance=TRUE,localImp=TRUE,replace=FALSE)

importance(rf)[order(importance(rf)[,"MeanDecreaseAccuracy"], decreasing=T),]
importance(rf)[order(importance(rf)[,"MeanDecreaseGini"], decreasing=T),]
varImpPlot(rf, main="Variable Importance in the Random Forest")

plot(rf, main="Error Rates for Random Forest")
legend("topright", c("OOB", "TypeI", "TypeII"), text.col=1:6, lty=1:3, col=1:3)

min.err <- min(rf$err.rate[,"OOB"])
min.err.idx <- which(rf$err.rate[,"OOB"]== min.err)
rf$err.rate[min.err.idx[1],]
#296
set.seed(5082)
rfb <- randomForest(formula=Revenue ~ .,data=trainset,ntree= min.err.idx[1], mtry=4,
                   importance=TRUE,localImp=TRUE,replace=FALSE)
rfb$err.rate[min.err.idx[1],]


rfb.pred <- predict(rfb, newdata=testset, type = 'prob')
rf.aucc <- roc.area(as.integer(as.factor(testset$Revenue))-1,rfb.pred[,2])
rf.aucc$A#0.922327 better to include all variables
roc.plot(as.integer(as.factor(testset$Revenue))-1,rfb.pred[,2], main="ROC Curve of Random Forest")
legend("bottomright", bty="n",
       sprintf("AUC = %1.3f", rf.aucc$A))


rf.pred <- predict(rfb, newdata=testset)

(rf.testTuneOptimal <- table(actual = testset$Revenue, predicted = rf.pred))
(rf.test.error <- (rf.testTuneOptimal[1, 2] + rf.testTuneOptimal[2, 1])/sum(rf.testTuneOptimal))#without selection 0.099
(rftype1 <- rf.testTuneOptimal[1,2]/sum(rf.testTuneOptimal[1,]))#without selection 0.04220399
(rftype2 <- rf.testTuneOptimal[2,1]/sum(rf.testTuneOptimal[2,]))#without selection 0.4285714
(rfpower <- rf.testTuneOptimal[2,2]/sum(rf.testTuneOptimal[2,]))#without selection 0.5714286


##########################
####      GBM        #####        
##########################
hyper_grid <- expand.grid(
  shrinkage = c(.01, .05, .1),
  interaction.depth = c(3, 5, 7),
  n.minobsinnode = c(5, 7, 10),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_acc = 0                     # a place to dump results
)
trainset.boost <-  trainset
testset.boost <-  testset
trainset.boost$Revenue <- as.numeric(trainset.boost$Revenue) - 1
testset.boost$Revenue <- as.numeric(testset.boost$Revenue) - 1
for(i in 1:nrow(hyper_grid)) {

  # reproducibility
  set.seed(5082)

  # train model
  gbm.tune <- gbm(
    formula = Revenue ~ .,
    distribution = "bernoulli",
    data = trainset.boost,
    n.trees = 500,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    verbose = FALSE
  )

  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)

  hyper_grid$min_acc[i] <- min(gbm.tune$valid.error)
}

hyper_grid %>% 
  dplyr::arrange(min_acc) %>%
  head(10)

gbm.boosting = gbm(Revenue~., data = trainset.boost, distribution = 'bernoulli',
                   n.trees = 88,shrinkage = .1,interaction.depth = 5, n.minobsinnode = 10, bag.fraction = 0.65)
summary(gbm.boosting)
gbm.pred <- predict(gbm.boosting, newdata = testset.boost, n.trees = 88,type="response")
gbm.all.pred <- rep("0",nrow(testset.boost))
gbm.all.pred[gbm.pred > 0.5] <- "1"
(gbmalltable <- table(testset.boost$Revenue, gbm.all.pred))
(allerror <- (gbmalltable[2]+gbmalltable[3])/sum(gbmalltable))# 0.106
(gbmalltype1 <- gbmalltable[1,2]/sum(gbmalltable[1,]))#0.05744431
(gbmalltype2 <- gbmalltable[2,1]/sum(gbmalltable[2,]))#0.3877551
(gbmallpower <- gbmalltable[2,2]/sum(gbmalltable[2,]))#0.6122449


gbm.aucc <- roc.area(as.integer(as.factor(testset$Revenue))-1,gbm.pred)
gbm.aucc$A#0.9136062

roc.plot(as.integer(as.factor(testset$Revenue))-1,gbm.pred, main="ROC Curve of GBM")
legend("bottomright", bty="n",
       sprintf("AUC = %1.3f", gbm.aucc$A))
##########################
####      XGB        #####        
##########################
# set.seed(5082)
# dtrain <- xgb.DMatrix(data = train.x,label = as.numeric(train.data$Revenue)-1) 
# dtest <- xgb.DMatrix(data = test.x,label = as.numeric(test.data$Revenue)-1) 
# # params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)
# 
# xgb.hyper_grid <- expand.grid(
#   eta = c(.01, .05, .1, .3),
#   max_depth = c(1, 3, 5, 7),
#   min_child_weight = c(1, 3, 5, 7),
#   subsample = c(.65, .8, 1), 
#   colsample_bytree = c(.8, .9, 1),
#   optimal_trees = 0,               # a place to dump results
#   min_error = 0                     # a place to dump results
# )
# for(i in 1:nrow(xgb.hyper_grid)) {
#   
#   # create parameter list
#   params <- list(
#     eta = xgb.hyper_grid$eta[i],
#     max_depth = xgb.hyper_grid$max_depth[i],
#     min_child_weight = xgb.hyper_grid$min_child_weight[i],
#     subsample = xgb.hyper_grid$subsample[i],
#     colsample_bytree = xgb.hyper_grid$colsample_bytree[i]
#   )
#   
#   # reproducibility
#   set.seed(5082)
#   
#   # train model
#   xgb.tune <- xgb.cv(
#     params = params,
#     booster = "gbtree",
#     data = dtrain,
#     label = as.numeric(train.data$Revenue)-1,
#     nrounds = 5000,
#     nfold = 10,
#     objective = "binary:logistic",
#     verbose = 0,   
#     early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
#   )
#   
#   # add min training error and trees to grid
#   xgb.hyper_grid$optimal_trees[i] <- which.min(xgb.tune$evaluation_log$test_error_mean)
#   xgb.hyper_grid$error[i] <- min(xgb.tune$evaluation_log$test_error_mean)
# }
# 
# xgb.hyper_grid %>%
#   dplyr::arrange(min_error) %>%
#   head(10)
# 
# optparams <- list(
#   eta = 0.01,
#   max_depth = 1,
#   min_child_weight = 1,
#   subsample = 0.65,
#   colsample_bytree = 0.8
# )
# xgb.opt <- xgb.train (data = dtrain, label = as.numeric(train.data$Revenue)-1, nrounds = 100, watchlist = list(val=dtest,train=dtrain), 
#                    print_every_n = 10, early_stopping_rounds = 10, maximize = F, eta=0.01, max_depth=1, min_child_wight=1,subsample=0.65)
# xgb.pred <- predict(xgb.opt, newdata = dtest,type="response")
# xgb.all.pred <- rep("FALSE",nrow(test.data))
# xgb.all.pred[xgb.pred > 0.5] <- "TRUE"
# (xgballtable <- table(test.data$Revenue, xgb.all.pred))
# (xgballerror <- (xgballtable[2]+xgballtable[3])/sum(xgballtable))# 0.119
# (xgballtype1 <- xgballtable[1,2]/sum(xgballtable[1,]))#0.07620164
# (xgballtype2 <- xgballtable[2,1]/sum(xgballtable[2,]))#0.3673469
# (xgballpower <- xgballtable[2,2]/sum(xgballtable[2,]))#0.6326531
# 
# 
# xgb.aucc <- roc.area(as.integer(as.factor(testset$Revenue))-1,xgb.pred)
# xgb.aucc$A#0.8602651
# 
# roc.plot(as.integer(as.factor(testset$Revenue))-1,xgb.pred, main="ROC Curve of XGB")
# legend("bottomright", bty="n",
#        sprintf("AUC = %1.3f", xgb.aucc$A))

#type2: the customer did buy but we predict he did not buy
#type1: the customer did not buy but we predict he buy