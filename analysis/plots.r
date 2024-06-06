library(ggplot2)
library(ggpubr)
library(scales)

plot <- function(env_name, alg_name, pos)
{
    data <- read.csv(paste0("../data/", env_name, "/", alg_name, ".csv"))
    plot <- ggplot(data, aes(x = Step, y = Value)) + geom_line(color = "red") +
      labs(title = alg_name, x = "frame", y = "episode return") + 
      scale_y_continuous(limits = c(0, 20)) + 
      scale_x_continuous(limits = c(0, 2000000), labels = scales::comma)
    
    if (pos == 2 || pos == 4)
    {
      plot <- plot + theme(axis.title.y = element_blank(), 
                           axis.line.y = element_blank(), 
                           axis.ticks.y = element_blank())
    }
    if (pos == 1 || pos == 2)
    {
      plot <- plot + theme(axis.title.x = element_blank(), 
                           axis.line.x = element_blank(), 
                           axis.ticks.x = element_blank())
    }
    
    plot
}


ggarrange(plot("breakout", "ppo", 1), plot("breakout", "dqn", 2),
          plot("breakout", "simple", 3), plot("breakout", "dreamer", 4),
          ncol = 2, nrow = 2)