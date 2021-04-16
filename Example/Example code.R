
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
load("tempdata.RData")


#################################################################
####### Censored Quantile Regression Forest with WW Split #######
#################################################################
data_train = data_temp
library(randomForestSRC)
rfsrc_sim = rfsrc(Surv(time,status)~.,data = data_train,ntree = 500,
                  nodesize = 10,do.trace = T,forest.wt = T)

rfsrc_sim_surv = cbind(1,predict(rfsrc_sim,
                                 newdata = data_train)$survival)

time_interest = predict(rfsrc_sim,newdata = data_train)$time.interest
time_censor = data_train$time[data_train$status==0]
F_est = sapply(1:length(time_censor),function(i)
  1-rfsrc_sim_surv[data_train$status==0,][i,
                                          sum(time_interest<=time_censor[i])+1],
  simplify = T)

weight_censor = ifelse(F_est>=tau,1,(tau-F_est)/(1-F_est))

data_surv = data.frame(rbind(data_train[data_train$status==0,],
                             data_train[data_train$status==1,],
                             data.frame(time = 100*max(data_train$time),
                                        data_train[data_train$status==0,-1])),
                       id = c(1:nrow(data_train),1:sum(data_train$status==0)))
data_surv = data.frame(data_surv,
                       weight = c(weight_censor,rep(1,sum(data_train$status==1)),1-weight_censor))

Z = as.matrix(as.double(data_surv$z))
X = as.matrix(data_surv[,namex])
Y = as.matrix(data_surv$time)
delta = as.matrix(data_surv$status)[1:nrow(data_train),]
weight_censor = matrix(data_surv$weight,nrow = 1)
X_new = as.matrix(data_test[,namex])

set.seed(taskid)
censor_forest = .Call('_rf_CenQRForest_WW_C', PACKAGE = 'rf',
                      Z,X,Y,X_new,delta = delta,
                      tau = 0.5,
                      weight_rf = rep(1,length(Y)),
                      weight_censor = weight_censor,
                      quantile_level = as.matrix(seq(0,1,length.out = 50)[2:49]),
                      numTree = 500,minSplit1 = 8,maxNode = 500,
                      mtry = min(ceiling(sqrt(length(namex)) + 20), length(namex)-1))

