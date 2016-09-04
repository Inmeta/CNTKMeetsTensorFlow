rm(list = ls(all = TRUE))
source("funcPlot.r")

#################################### generate test and train data #########################
x1 <- rnorm(1000, mean = 0, sd = 1)
x1 <- round(x1, 6)
x2 <- rnorm(1000, mean = 0, sd = 1)
x2 <- round(x2, 6)
bias <- 3
noise <- rnorm(1000, mean = 0, sd = 0.8)
y <- round((5 * x1 + 7 * x2 + bias + noise) / 5, 6)

# visualize x1, x2, y 
plotPairs(data)

# save to file
data <- as.data.frame(cbind(x1, x2, y) )
write.table(data, file = "..\\Data\\LR.data.csv", row.names = F, sep = ",", dec = ".")

# save in CNTK features format
data <- read.table(file = "..\\Data\\LR.data.csv", sep = ",", dec = ".", header = T)
data.txt <- as.data.frame(cbind(paste("|features", data$x1, data$x2, sep = " "),  paste("|labels", data$y, sep = " ")))

write.table(data.txt[1:200,], file = "..\\Data\\LR.test.txt", row.names = F, col.names = F,  quote = F, sep = " ")
write.table(data.txt[201:1000,], file = "..\\Data\\LR.train.txt", row.names = F, col.names = F, quote = F, sep = " ")
