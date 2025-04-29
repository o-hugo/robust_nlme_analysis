# No R
setwd("/home/huguinho/Documents/GitHub/robust_nlme_analysis/data")
library(nlme)
data(Soybean)
write.csv(Soybean, "soybean_growth.csv", row.names = FALSE)
