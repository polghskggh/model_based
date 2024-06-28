library(ggplot2)
library(ggpubr)
library(scales)

read <- function(env_name, type, alg_name)
{
    read.csv(paste0("../data/", env_name, "/", type, "/", alg_name, ".csv"))
}

smooth <- function(env, type, name)
{
    geom_smooth(read(env, type, name), mapping=aes(x=Step, y=Value, color=name), method='gam')

}

plotall <- function(env, type)
{
  axis <- type
    if (type == 'length') {
    axis <- "episode length"
  }
    title_string <- paste0(axis, " of different algorithms in ",  env)
    ggplot() + smooth(env, type, 'ppo') + smooth(env, type, "simple_hybrid0") +
        smooth(env, type, 'simple') + smooth(env, type, 'simple_hybrid1') +
        labs(color = "algorithm", x="frames", y=type, title = title_string) +
        theme(legend.position = c(0.8,0.2)) +
        scale_x_continuous(labels = scales::label_number(scale = 1e-6, suffix = "M")) +
        scale_color_manual(values = c("ppo" = "red",
                                      "simple" = "blue", 
                                      "simple_hybrid0" = "purple",
                                      "simple_hybrid1" = "darkgreen"))
}

plotall('breakout', 'length')
plotall('breakout', 'returns')


