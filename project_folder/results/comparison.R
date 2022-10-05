# Install required libraries
required_lib <- c("tidyverse", "plyr", "beeswarm")
for (package in required_lib){
  if (!requireNamespace(package, quietly = TRUE)) {
    install.packages(package, quiet = FALSE)
  }
}

# Load tidyverse library
library(tidyverse)

final_results = read_csv('final_results.csv')




# Read data ---------------------------------------------------------------------------------------------
df <- 
  readr::read_csv(
    "/MAIN/THESIS/CODE_vol15/project_folder/results/Fig3_orig.csv",
    col_names = TRUE
  ) %>% 
  dplyr::filter(
    figure_panels == "a, c, e, f"
  ) %>% 
  dplyr::select(
    -.data$figure_panels,
    -(.data$spearman_random_predictions:.data$rmse_replicates)
  )


# Set parameters for the figures ----------------------------------------------------------------------- 
size_text <- 10
size_tag <- 12
main_point_size <- 1.2
highlight_point_size <- 1.9
stroke_size <- 0.7
stroke_size_highlight <- 1.3



df_rmse_leq_2 <- df %>% dplyr::filter(.data$rmse <= 2)

# Figure ROUND 1 --------------------------------------------------------------------------------------------
df_rmse_leq_2_r1 <- df_rmse_leq_2 %>% dplyr::filter(.data$round == "Round 1")

ggplot(data = df_rmse_leq_2_r1, mapping = aes(x = rmse, y = spearman)) + 
  geom_point(
    size = main_point_size,
    color = rgb(0.59, 0.59, 0.59)
  ) +
  geom_point(
    data = df_rmse_leq_2_r1 %>% dplyr::filter(.data$baseline_model == "yes"),
    aes(x = rmse, y = spearman),
    size = highlight_point_size,
    color = "#5500ff"
  ) +
  geom_point(
    aes(x = final_results$test1[1], y = final_results$test1[2]),
    size = 7,
    shape=8,
    color = "#ff0000"
  ) +
  geom_point(
    data = df_rmse_leq_2_r1 %>% dplyr::filter(.data$highlighted_teams == "Gregory Koytiger"),
    aes(x = rmse, y = spearman),
    size = highlight_point_size,
    color = "#00b8e6"
  ) +
  labs(
    x = "RMSE", 
    y = "Spearman correlation", 
  ) + 
  ggtitle("Round 1") + 
  coord_equal(ratio = 1.3) + 
  xlim(0.89, 2) + 
  ylim(-0.19, 0.6) + 
  theme_bw() + 
  theme(
    plot.title = element_text(size = size_text,  face = "plain", hjust = 0.5),
    axis.text = element_text(size = size_text), 
    axis.title = element_text(size = size_text),
    plot.tag = element_text(size = size_tag, face = "bold")
  ) 


# Figure ROUND 2 --------------------------------------------------------------------------------------------
df_rmse_leq_2_r2 <- df_rmse_leq_2 %>% dplyr::filter(.data$round == "Round 2")

ggplot(data = df_rmse_leq_2_r2, mapping = aes(x = rmse, y = spearman)) + 
  geom_point(
    size = main_point_size,
    color =  rgb(0.59, 0.59, 0.59)
  ) +
  geom_point(
    data = df_rmse_leq_2_r2 %>% dplyr::filter(.data$baseline_model == "yes"),
    aes(x = rmse, y = spearman),
    size = highlight_point_size,
    color = "#5500ff"
  ) +
  geom_point(
    aes(x = final_results$test2[1], y = final_results$test2[2]),
    size = 7,
    color = "#ff0000",
    size = 7,
    shape=8
  ) +
  geom_point(
    data = df_rmse_leq_2_r2 %>% dplyr::filter(.data$highlighted_teams == "Gregory Koytiger"),
    aes(x = rmse, y = spearman),
    size = highlight_point_size,
    color = "#00b8e6"
  ) +
  labs(
    x = "RMSE", 
    y = "Spearman correlation", 
  ) + 
  ggtitle("Round 2") + 
  coord_equal(ratio = 1.3) + 
  xlim(0.89, 2) + 
  ylim(-0.19, 0.6) + 
  theme_bw() + 
  theme(
    plot.title = element_text(size = size_text, face = "plain", hjust = 0.5),
    axis.text = element_text(size = size_text), 
    axis.title = element_text(size = size_text),
    plot.tag = element_text(size = size_tag, face = "bold")
  ) 


# REFERNECE POINTS

ggplot() + 
  geom_point(
    size = main_point_size,
    color = rgb(0.59, 0.59, 0.59)
  ) +
  geom_point(
    aes(x = 0.75, y = 0),
    size = 7,
    shape=8,
    color = "#ff0000"
  ) + 
  geom_point(
    aes(x = 0.5, y = 0),
    size = highlight_point_size,
    color = "#00b8e6"
  ) +
  geom_point(
    aes(x = 1, y = 0),
    size = highlight_point_size,
    color = "#5500ff"
  ) + theme_classic()

