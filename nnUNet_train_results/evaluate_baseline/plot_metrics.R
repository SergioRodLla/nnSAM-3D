library(jsonlite)

setwd("/media/sergio/TOSHIBA EXT/Master/TFM/HECKTOR/baseline_unet/git_repo/nnUNet_train_results/evaluate_baseline/")
all5fold <- fromJSON("all5folds_summary.json", flatten = TRUE)
fg_metrics <- as.data.frame(all5fold$foreground_mean)
mean_tumor <- as.data.frame(all5fold$mean$`1`)
mean_nodule <- as.data.frame(all5fold$mean$`2`)

for (i in 0:4) {
  tmp <- fromJSON(paste0("fold", i, "_summary.json"), flatten = TRUE)
  fg_metrics <- rbind(fg_metrics, as.data.frame(tmp$foreground_mean))
  mean_tumor <- rbind(mean_tumor, as.data.frame(tmp$mean$`1`))
  mean_nodule <- rbind(mean_nodule, as.data.frame(tmp$mean$`2`))
}

rownames(fg_metrics) <- c("All", "F0", "F1", "F2", "F3", "F4")
rownames(mean_tumor) <- c("All", "F0", "F1", "F2", "F3", "F4")
rownames(mean_nodule) <- c("All", "F0", "F1", "F2", "F3", "F4")

write.csv(fg_metrics, "foregroundMean_metrics.csv")
write.csv(mean_tumor, "tumor_mean_metrics.csv")
write.csv(mean_nodule, "nodule_mean_metrics.csv")

# fg_metrics <- fg_metrics %>% rownames_to_column("Fold") %>%
#   pivot_longer(cols = 2:9, names_to = "Metric", values_to = "Value")
# mean_tumor <- mean_tumor %>% rownames_to_column("Fold")
# mean_nodule <- mean_nodule %>% rownames_to_column("Fold")