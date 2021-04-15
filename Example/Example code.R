
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
load("tempdata.RData")

data_weight_MM = train
set.seed(234)
CQRforest(matY = as.matrix(data_weight_MM[,namey]),
          delta = as.matrix(data_weight_MM[,status]),
          matZ = as.matrix(cbind(1,data_weight_MM[,control.z])),
          matX = as.matrix(data_weight_MM[,namex]),
          control = list(numTree = 1))

#################################################################
####### Censored Quantile Regression Forest with WW Split #######
#################################################################
Z = as.matrix(train$z)
X = train[,paste("x",1:5,sep = "")]
Y = as.matrix(train$time)
delta = as.matrix(train$status)
set.seed(123)
censor_forest = CensQRforest(Y,delta,Z,X,
                             control = list(quantile_level = as.matrix(seq(0,1,length.out = 50)[2:49]),
                                            numTree = 500,
                                            minSplitTerm = 8,maxNode = 500,
                                            mtry = 3))

cbind(censor_forest$data$.X[order(censor_forest$weights[1,],
                                  decreasing = T),1:2],
      sort(censor_forest$weights[1,],decreasing = T))[140:152,]
data_pred = data.frame(y = censor_forest$data$.Y[1:500,],
                       x = censor_forest$data$.Z[1:500,],
                       weights = censor_forest$weights[3,])
rq(y~x,data = data_pred,tau = 0.5,weights = weights)
