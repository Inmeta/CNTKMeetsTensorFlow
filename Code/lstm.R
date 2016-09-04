rm(list = ls(all = TRUE))
source("func.r")

# simple neural network, CNTK
data <- read.table(file = "..\\Data\\LR.data.csv", sep = ",", dec = ".", header = T)
data.res <- as.data.frame(read.table("..\\cntk\\Output\\simpleNet.out.txt.HLast", header = F, sep = " ", dec = ".", stringsAsFactors = FALSE))
cntk.pred <- data.res[,1]
cntk.actual <- data$y[1:200]

res.cntk <- evaluateResults(cntk.pred, cntk.actual, 2)
plotModelResults(cntk.pred, cntk.actual, "CNTK Net")

# simple LSTM, CNTK
simple.prediction.lstm <- as.data.frame(read.table("Code\\cntk\\Examples\\MyRegression\\myLstm_out.txt.outputs", header = F, sep = " ", dec = ".", stringsAsFactors = FALSE))
simple.res.lstm <- evaluateResults(simple.prediction.lstm, simple.actual, 2)
plotModelResults(simple.prediction.lstm[, 1], simple.actual, "LSTM")