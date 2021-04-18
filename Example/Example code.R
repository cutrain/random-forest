
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
                      mtry = min(ceiling(sqrt(length(namex)) + 20),
                                 length(namex)-1))

#######################################################################
####### Quantile Process Regression Forest with rankscore Split #######
#######################################################################
rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
load("exampledataQPRF.RData")

Z = as.matrix(train[,control.z])
Y = as.matrix(train[,namey])
X = as.matrix(train[,namex])

N_test = round(200^(1/2))^2
X_test = matrix(c(rep(seq(0.05,0.95,length.out = round(200^(1/2))),
                      each = round(200^(1/2))),
                  rep(seq(0.05,0.95,length.out = round(200^(1/2))),
                      times = round(200^(1/2))),
                  runif((d-2)*N_test,0,1)),ncol = d)

set.seed(taskid)
QPR_forest = .Call('_rf_QPRForest_C', PACKAGE = 'rf',
                      Z,X,Y,X_test,taurange = c(0,1),
                      quantile_level = as.matrix(seq(0,1,length.out = 50)[2:49]),
                      max_num_tau = 2*10^3,numTree = 500,
                      minSplit1 = 8,maxNode = 500,
                      mtry = min(ceiling(sqrt(length(namex)) + 20),
                                 length(namex)-1))

mu_beta_test1 = (1+1/(1+exp(-20*(X_test[,1]-1/2))))*
  (1+1/(1+exp(-20*(X_test[,2]-1/2))))
mu_beta_test2 = (1+1/(1+exp(-20*(X_test[,1]-1/2))))*
  (1+1/(1+exp(-20*(X_test[,3]-1/2))))
mu_beta_test3 = (1+1/(1+exp(-20*(X_test[,2]-1/2))))*
  (1+1/(1+exp(-20*(X_test[,3]-1/2))))

beta_tau_test1 = beta_tau(mu_beta_test1,(1:99)/100,
                          quantile(mu_beta1,0.5))
beta_tau_test2 = beta_tau(mu_beta_test2,(1:99)/100,
                          quantile(mu_beta2,0.5))
beta_tau_test3 = beta_tau(mu_beta_test3,(1:99)/100,
                          quantile(mu_beta3,0.5))

bias1 = NULL
bias2 = NULL
bias3 = NULL
for(i in 1:nrow(X_test)){
  
  bias1 = c(bias,mean((QPR_forest$estimate[[i]][-100,2]-
                        beta_tau_test1[i,])/beta_tau_test1[i,]))
  bias2 = c(bias,mean((QPR_forest$estimate[[i]][-100,3]-
                         beta_tau_test2[i,])/beta_tau_test2[i,]))
  bias3 = c(bias,mean((QPR_forest$estimate[[i]][-100,4]-
                         beta_tau_test3[i,])/beta_tau_test3[i,]))
  
}


# data_est = data.frame(y = Y,z = Z,
#                       weights = censor_forest$weights[1,])
# qr_R = rq(y~z,data = data_est,weights = weights,tau = -1)
# 
# qr_tau_cpp_est =.Call('_rf_qr_tau_para_diff_cpp', 
#                       PACKAGE = 'rf', Z, Y, 
#                       weights = censor_forest$weights[1,], 
#                       taurange = c(0,1),tau_min = 1e-10,
#                       tol = 1e-14,maxit = 100000, 
#                       max_num_tau = 1000,
#                       use_residual = T)
