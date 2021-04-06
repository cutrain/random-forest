

data_weight_MM = train
set.seed(234)
CQRforest(matY = as.matrix(data_weight_MM[,namey]),
          delta = as.matrix(data_weight_MM[,status]),
          matZ = as.matrix(cbind(1,data_weight_MM[,control.z])),
          matX = as.matrix(data_weight_MM[,namex]),
          control = list(numTree = 1))
