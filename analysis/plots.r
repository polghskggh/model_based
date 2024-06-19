library(ggplot2)
library(ggpubr)
library(scales)

read <- function(env_name, type, alg_name)
{
    read.csv(paste0("../data/", env_name, "/", type, "/", alg_name, ".csv"))
}

smooth <- function(env, type, name)
{
    geom_smooth(read(env, type, name), mapping=aes(x=Step, y=Value, color=name), method='gam', se=FALSE)

}

plotall <- function(env, type)
{
    title_string <- paste0(type, " of different algorithms in ",  env)
    ggplot() + smooth(env, type, 'ppo') + smooth(env, type, 'simple') + 
        smooth(env, type, 'simple_dones') + smooth(env, type, 'simple_dones_hybrid') +
        labs(color = "algorithm", x="frames", y=type, title = title_string) +
        theme(legend.position = c(0.8,0.2)) +
        scale_x_continuous(labels = scales::label_number(scale = 1e-6, suffix = "M")) +
        scale_color_manual(values = c("ppo" = "red", 
                                      "simple" = "blue",
                                      "simple_dones" = "yellow", 
                                      "simple_dones_hybrid" = "green"))
}
plotall('breakout', 'length')
plotall('breakout', 'returns')


