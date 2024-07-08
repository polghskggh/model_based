library(ggplot2)
library(ggpubr)
library(scales)
library(tidyverse)

move_start <- function(data, name)
{
  move_val <- 0
  if (name == "dreamer") 
  {
    move_val <- 8 * 20 * 400
  } 
  if (name == "simple")
  {
    move_val <- 8 * 100 * 400
  }
  data %>% filter(Step > move_val) %>% mutate(Step = Step - move_val) %>% filter(Step < 100000)
}

read <- function(env_name, type, alg_name)
{
    read.csv(paste0("../data/", env_name, "/", type, "/", alg_name, ".csv"))
}

smooth <- function(env, type, name, filter)
{
    data <- read(env, type, name)
    if (filter) 
    {
      data <- move_start(data, name)
    }
    
    geom_smooth(data, mapping=aes(x=Step, y=Value, color=name), method='gam')

}

plotall <- function(env, type, filter=FALSE)
{
    ggplot() + smooth(env, type, 'ppo', filter) + 
        smooth(env, type, "dreamer", filter) + 
        smooth(env, type, 'simple', filter) + 
        labs(color = "algorithm", x="frames", y=type) +
        scale_x_continuous(labels = scales::label_number(scale = 1e-6, suffix = "M")) +
        scale_color_manual(values = c("ppo" = "red",
                                      "simple" = "blue", 
                                      "dreamer" = "purple"))
}

correlate_length_returns <- function(env, algorithm)
{
  cor(read(env, "length", algorithm)$Value, read(env, "returns", algorithm)$Value)
} 

interactions <- function(env, algorithm)
{
  vals <- move_start(read(env, "interactions", algorithm), algorithm)$Value
}


plt1 <- plotall('breakout', "length")
plt1 <- plt1 + theme(legend.position = "none", 
                     axis.title.x = element_blank(),
                     axis.text.x = element_blank(),
                     axis.ticks.x = element_blank(),
                     axis.line.x = element_blank()) + labs(title="Episode information upon termination in Breakout")


plt2 <- plotall('breakout', "returns")
plt2 <- plt2 + theme(legend.position = c(0.8,0.2))


plt3 <- plotall('breakout', "returns", TRUE)


ggarrange(plt1, plt2, nrow=2)
plt3
correlate_length_returns("breakout", "ppo")
interactions("breakout", "dreamer")
interactions("breakout", "simple")





