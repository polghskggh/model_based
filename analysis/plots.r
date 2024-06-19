library(ggplot2)
library(ggpubr)
library(scales)

read <- function(env_name, alg_name)
{
    read.csv(paste0("../data/", env_name, "/", alg_name, ".csv"))
}

env <- 'breakout'

ggplot() + geom_point(data=read(env, 'simple'), aes(x=Step, y=Value))

