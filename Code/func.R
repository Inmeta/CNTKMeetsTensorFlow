library(ggplot2)

panel.hist <- function(x, ...) {
    usr <- par("usr");
    on.exit(par(usr))
    par(usr = c(usr[1:2], 0, 1.5))
    h <- hist(x, plot = FALSE)
    breaks <- h$breaks;
    nB <- length(breaks)
    y <- h$counts;
    y <- y / max(y)
    rect(breaks[ - nB], 0, breaks[-1], y, col = "light blue", ...)
}

plotPairs <- function(d) {
    pairs(d, diag.panel = panel.hist)   
}

plotModelResults <- function(predicted, actual, caption) {
    par(mfrow = c(2, 1))
    myColor <- rgb(70, 0, 240, max = 255, alpha = 85, names = "myColor")
    # plot Predicted vs Actual
    plot(x = predicted, y = actual, type = "p", col = myColor,
        xlab = "Predicted", ylab = "Actual", main = paste(caption, "Predicted vs Actual"),
        cex.lab = 2, cex.main = 2, cex = 2, lwd = 2)
    lines(x = predicted, y = predicted, col = "black", lwd = 1, lty = 2)

    # plot standartized residuals
    library(moments)
    x <- actual - predicted
    x <- x / sd(x)
    x.norm <- rnorm(length(x))

    myColorN <- rgb(60, 160, 60, max = 255, alpha = 30, names = "myColorN")
    hist(x, col = myColor, cex.lab = 2, cex.main = 2, cex = 2, ylim = c(0,50),
        main = "Standartized Residuals distribution vs Normal distribution")
    hist(x.norm, add = T, col = myColorN)
}

evaluateResults <- function(predicted, actual, p) {
    n <- length(actual)
    mean = mean(actual)
    res <- list()

    sstot = sum((actual - mean) ^ 2)
    ssres = sum((predicted - actual) ^ 2)
    res$r2 = 100*(1 - (ssres / sstot))

    x <- actual - predicted
    x <- x / sd(x)
    res$skeweness <- skewness(x)
    res$kurtosis <- kurtosis(x)

    return(res)
}
