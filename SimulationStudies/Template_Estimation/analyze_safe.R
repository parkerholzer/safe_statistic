library(gap)

safedata = read.csv("Corrected_Pvals.csv")

qqunif(safedata$Pvals, ci=TRUE, logscale = FALSE, pch=19,
       lwd = 4)
legend(0,1,legend = c("95% Confidence Bands"))

library(ggplot2)
gg_qqplot <- function(ps, ci = 0.95, simultaneous_ci = F) {
  n  <- length(ps)
  if(simultaneous_ci){
    alpha = (1 - ci)/length(ps)
  }else{
    alpha = 1 - ci
  }
  df <- data.frame(
    observed = sort(ps),
    expected = ppoints(n),
    clower   = qbeta(p = alpha/2, shape1 = 1:n, shape2 = n:1),
    cupper   = qbeta(p = 1 - alpha/2, shape1 = 1:n, shape2 = n:1)
  )
  Pe <- expression(paste("Uniform(0,1) ", plain(P)))
  Po <- expression(paste("Observed ", plain(P)))
  ggplot(df) +
    theme_bw() +
    geom_point(aes(expected, observed), shape = 19, size = 2, colour = 'blue') +
    geom_abline(intercept = 0, slope = 1, alpha = 0.5, size = 1) +
    geom_line(aes(expected, cupper), linetype = 2, size = 1.5) +
    geom_line(aes(expected, clower), linetype = 2, size= 1.5) +
    theme(axis.title.x = element_text(size=14, face="bold"),
          axis.title.y = element_text(size=14, face="bold"),
          axis.text = element_text(size=13)) +
    xlab(Pe) + ylab(Po)
  
}
gg_qqplot(safedata$Pvals, ci=0.99)
gg_qqplot(safedata$Pvals, ci=0.95, simultaneous_ci = T)
