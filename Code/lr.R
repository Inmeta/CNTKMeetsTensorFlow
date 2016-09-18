source("func.r")

data <- read.table(file = "..\\Data\\LR.data.csv", sep = ",", dec = ".", header = T)
############################## predict with LR and visualize ##############################
lr.model <- lm(y ~ x1 + x2, data = data[201:1000,])
lr.pred <- predict(lr.model, data[1:200, c("x1", "x2")])
lr.actual <- data$y[1:200]

#visualize results of LR
res.lr <- evaluateResults(lr.pred, lr.actual, 2)
plotModelResults(lr.pred, lr.actual, "Linear regression")

summary(lr.model)

# simple neural network, CNTK
data <- read.table(file = "..\\Data\\LR.data.csv", sep = ",", dec = ".", header = T)
data.res <- as.data.frame(read.table("..\\cntk\\LR\\Output\\simpleNet.out.txt.HLast", header = F, sep = " ", dec = ".", stringsAsFactors = FALSE))
cntk.pred <- data.res[,1]
cntk.actual <- data$y[1:200]

res.cntk <- evaluateResults(cntk.pred, cntk.actual, 2)
plotModelResults(cntk.pred, cntk.actual, "CNTK Net")