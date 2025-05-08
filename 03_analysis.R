## ------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------
## Script name: 03_analysis.R
##
## Purpose of script: automate analysis and generate images/plots
##
## Author: Rafael Gomes Mantovani
##
## Date Created: 2024-05-02
## ------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------

cat(" @ Loading all required files:\n")
library(ggplot2,    quietly = TRUE, warn.conflicts = FALSE)
library(reshape2,   quietly = TRUE, warn.conflicts = FALSE)
library(dplyr,      quietly = TRUE, warn.conflicts = FALSE)
library(Rtsne,      quietly = TRUE, warn.conflicts = FALSE)
library(umap,       quietly = TRUE, warn.conflicts = FALSE)

dir.create(path = "plots/", recursive=TRUE, showWarnings=FALSE)

# ---------------------------
# ---------------------------

ml.files = list.files(path = "output/machineLearning")
dl.files = list.files(path = "output/deepLearning")

ml.perf.files = ml.files[grepl(x = ml.files, pattern = "performance")]
dl.perf.files = dl.files[grepl(x = dl.files, pattern = "performance")]

seeds = c(171, 269, 289, 376, 404, 42, 51, 666, 720, 767)

# ---------------------------
# loading ML performances
# ---------------------------

aux = lapply(ml.perf.files, function(jobfile) {
	df = read.csv(file = paste0("output/machineLearning/",jobfile))
	df = as.data.frame(df)
	df$type_of_image = "rgb"
	algo = gsub(x = jobfile, pattern = "performances_|_seed_|.csv", replacement = "")
	algo = gsub(x = algo, pattern = paste(seeds, collapse="|"), replacement = "")
	df$model = algo
	return(df)
})
df.ml = do.call("rbind", aux)
df.ml$DA = FALSE

# ---------------------------
# loading DL performances
# ---------------------------

aux.dl = lapply(dl.perf.files, function(jobfile) {
	df = read.csv(file = paste0("output/deepLearning/",jobfile))
	df = as.data.frame(df)
	if(grepl(x = jobfile, pattern = "DA")) {
		df$DA = TRUE
	} else {
		df$DA = FALSE
	}
	return(df)
})
df.dl = do.call("rbind",  aux.dl)

# ---------------------------
# all performance values
# ---------------------------

df.performances = rbind(df.ml, df.dl)
df.performances$algo = "None"
ids = which(df.performances$DA)

df.performances$algo[ids]  = paste0(df.performances$model[ids], "_DA")
df.performances$algo[-ids] = df.performances$model[-ids]
df.performances$algo = paste0(df.performances$algo, "_", df.performances$type_of_image)
df.performances$algo = toupper(df.performances$algo)

# ---------------------------
# Boxplot (overall)
# ---------------------------

g = ggplot(df.performances, aes(x = reorder(algo, -fscore), y = fscore, group = algo))
g = g + geom_violin() + geom_boxplot(width = .15) + theme_bw()
g = g + geom_hline(yintercept = 0.8, linetype = "dotted", colour = "red")
g = g + labs(x = "Model", y = "FScore)")
g = g + scale_y_continuous(limits = c(0.6, 0.9))
g = g + theme(axis.text.x=element_text(angle = 90, hjust = 1, vjust = 0.5))
ggsave(g, filename = "plots/overall_boxplot.pdf", width = 6.84, height = 2.95)

# ---------------------------
# Table with aggregated measures (mean and sd)
# ---------------------------

cat(" @ Output: Average performance values \n")


algos = unique(df.performances$algo)
aux.algos = lapply(algos, function(algorithm) {
	subset = dplyr::filter(df.performances, algo == algorithm)
	ret = subset[1, 5:8]	
	ret$mean_fscore = mean(subset$fscore)
	ret$sd_fscore   = sd(subset$fscore)
	return(ret)	
})

df.algos = do.call("rbind", aux.algos)
df.algos = df.algos[order(df.algos$mean_fscore, decreasing=TRUE),]

#   type_of_image   model    DA         algo mean_fscore  sd_fscore
# 16           rgb   vgg19 FALSE    VGG19_RGB   0.8946562 0.03841407
# 5            rgb   ridge FALSE    RIDGE_RGB   0.8275398 0.03301278
# 4            rgb      rf FALSE       RF_RGB   0.8143351 0.04464497
# 1            rgb bagging FALSE  BAGGING_RGB   0.8107582 0.03493352
# 6            rgb     svm FALSE      SVM_RGB   0.8005287 0.02785584
# 11           raw   lwcnn  TRUE LWCNN_DA_RAW   0.7861326 0.03092319
# 15           rgb   vgg19  TRUE VGG19_DA_RGB   0.7834181 0.05409939
# 7            raw     cnn  TRUE   CNN_DA_RAW   0.7830159 0.03058305
# 12           rgb   lwcnn  TRUE LWCNN_DA_RGB   0.7724113 0.03976526
# 3            rgb     knn FALSE      KNN_RGB   0.7634859 0.02832514
# 13           raw   lwcnn FALSE    LWCNN_RAW   0.7626910 0.04746054
# 8            rgb     cnn  TRUE   CNN_DA_RGB   0.7519070 0.05028526
# 14           rgb   lwcnn FALSE    LWCNN_RGB   0.7431919 0.13634889
# 2            rgb      dt FALSE       DT_RGB   0.7429986 0.05160906
# 9            raw     cnn FALSE      CNN_RAW   0.7357192 0.07227774
# 10           rgb     cnn FALSE      CNN_RGB   0.6839598 0.12282515

# ---------------------------
# Statistical tests between top three models
# ---------------------------

cat(" @ Output: Statistical tests - Wilcoxon 95% \n")

vgg.perf   = df.performances[which(df.performances$algo == "VGG19_RGB"),]
ridge.perf = df.performances[which(df.performances$algo == "RIDGE_RGB"),]
rf.perf    = df.performances[which(df.performances$algo == "RF_RGB"),]

vgg_vs_ridge = wilcox.test(vgg.perf$fscore, ridge.perf$fscore, paired = TRUE)
print(vgg_vs_ridge)
cat(" @ Wilcoxon: VGG vs Ridge \n")
print(vgg_vs_ridge$p.value)

vgg_vs_rf = wilcox.test(vgg.perf$fscore, rf.perf$fscore, paired = TRUE)
print(vgg_vs_rf)
cat(" @ Wilcoxon: VGG vs RF \n")
print(vgg_vs_rf$p.value)

rf_vs_ridge = wilcox.test(rf.perf$fscore, ridge.perf$fscore, paired = TRUE)
print(rf_vs_ridge)
cat(" @ Wilcoxon: RF vs Ridge \n")
print(rf_vs_ridge$p.value)

# ---------------------------
# Plot: top 3 confusion matrices
# ---------------------------

cat(" @ Plot: Top 3 Confusion Matrices\n")

# --------------
# VGG files
# --------------

vgg.files = dl.files[grepl(x = dl.files, pattern = "predictions_vgg19")]
vgg.files = vgg.files[11:20]

aux.vgg = lapply(vgg.files, function(predfile) {
	data = read.csv(file = paste0("output/deepLearning/", predfile))
	tab  = base::table(data$predictions, data$labels)
	return(tab)
})

vgg.table = round(Reduce("+", aux.vgg)/length(aux.vgg))
df.vgg.table = melt(vgg.table)
colnames(df.vgg.table) = c("TClass", "PClass", "Y")
df.vgg.table$PClass = as.factor(df.vgg.table$PClass)
df.vgg.table$TClass = as.factor(df.vgg.table$TClass)
df.vgg.table$goodbad = "bad"
df.vgg.table$goodbad[c(1, 4)] = "good"
df.vgg.table$algo = "VGG19"

# --------------
# Ridge files
# --------------

ridge.files = ml.files[grepl(x = ml.files, pattern = "predictions_ridge")]
aux.ridge = lapply(ridge.files, function(predfile) {
	data = read.csv(file = paste0("output/machineLearning/", predfile))
	tab  = base::table(data$predictions, data$Y)
	return(tab)
})

ridge.table = round(Reduce("+", aux.ridge)/length(aux.ridge))
df.ridge.table = melt(ridge.table)
colnames(df.ridge.table) = c("TClass", "PClass", "Y")
df.ridge.table$PClass = as.factor(df.ridge.table$PClass)
df.ridge.table$TClass = as.factor(df.ridge.table$TClass)
df.ridge.table$goodbad = "bad"
df.ridge.table$goodbad[c(1, 4)] = "good"
df.ridge.table$algo = "Ridge"

# --------------
# RF files
# --------------

rf.files = ml.files[grepl(x = ml.files, pattern = "predictions_rf")]
aux.rf = lapply(rf.files, function(predfile) {
	data = read.csv(file = paste0("output/machineLearning/", predfile))
	tab  = base::table(data$predictions, data$Y)
	return(tab)
})

rf.table = round(Reduce("+", aux.rf)/length(aux.rf))
df.rf.table = melt(rf.table)
colnames(df.rf.table) = c("TClass", "PClass", "Y")
df.rf.table$PClass = as.factor(df.rf.table$PClass)
df.rf.table$TClass = as.factor(df.rf.table$TClass)
df.rf.table$goodbad = "bad"
df.rf.table$goodbad[c(1, 4)] = "good"
df.rf.table$algo = "RF"

# --------------
# --------------

df.all = rbind(df.vgg.table, df.ridge.table, df.rf.table)
df.all$PClass = factor(df.all$PClass, levels = c(1, 0))

g2 = ggplot(data =  df.all, mapping = aes(x = TClass, y = PClass, alpha = Y, fill = goodbad))
g2 = g2 + geom_tile() + geom_text(aes(label = Y), vjust = .5, fontface  = "bold", alpha = 1)
g2 = g2 + scale_fill_manual(values = c(good = "green", bad = "red"))
g2 = g2 + theme_bw() + theme(legend.position = "none") + facet_grid(~algo)
g2 = g2 + labs(x = "True", y = "Predicted")

ggsave(g2, file = "plots/top3_confusion_matrices.pdf", width = 5.12, height = 2.18)


# ---------------------------
#  Ridge PCA (tSNE) 2D plot
# ---------------------------

cat(" @ Plot: 2D PCA plot \n")

feat1 = read.csv("data/features/diagnosis_rgb.csv")
feat1$Class = "Diagnosis"
feat2 = read.csv("data/features/healthy_rgb.csv")
feat2$Class = "Healthy"

dataset.feat = rbind(feat1, feat2)

pca_res = prcomp(dataset.feat[, -c(1:4, ncol(dataset.feat))], scale. = TRUE)

df.pca = as.data.frame(pca_res$x[,1:2])
df.pca$Class = dataset.feat$Class

variance = summary(pca_res)$importance[2,]

g3 = ggplot(data = df.pca, mapping = aes(x = PC1, y = PC2, shape = Class, colour = Class))
g3 = g3 + geom_point() + theme_bw()
g3 = g3 + scale_colour_manual(values = c("black", "red"))
g3 = g3 + labs(x = paste0("1st Component (", round(variance[1] * 100, 2), "%)"), 
	y = paste0("2nd Component (",  round(variance[2] * 100, 2), "%)"))
ggsave(g3, file = "plots/pca_ridge.pdf", width = 4.24, height = 3)

# ---------------------------
cat(" @ Plot: 2D tSNE plot\n")
# ---------------------------

# Tsne versions
ids  = !duplicated(dataset.feat[, -c(1:4, ncol(dataset.feat))])
temp =  dataset.feat[ids,]
tsne_out = Rtsne(temp[,-c(1:4, ncol(dataset.feat))])
df.tsne = data.frame(x = tsne_out$Y[,1], y = tsne_out$Y[,2])
df.tsne$Class = dataset.feat$Class[ids] 

g4 = ggplot(data = df.tsne, mapping = aes(x = x, y = y, shape = Class, colour = Class))
g4 = g4 + geom_point() + theme_bw()
g4 = g4 + scale_colour_manual(values = c("black", "red"))
g4 = g4 + labs(x = "T[1]", y = "T[2]")
ggsave(g4, file = "plots/tsne_ridge.pdf", width = 4.24, height = 3)

# ---------------------------
cat(" @ Plot: 2D UMAP plot\n")
# ---------------------------

ids  = !duplicated(dataset.feat[, -c(1:4, ncol(dataset.feat))])
temp =  dataset.feat[ids,]
umap_out = umap(temp[,-c(1:4, ncol(dataset.feat))])
df.umap = data.frame(x = umap_out$layout[,1], y = umap_out$layout[,2])
df.umap$Class = dataset.feat$Class[ids] 

g5 = ggplot(data = df.umap, mapping = aes(x = x, y = y, shape = Class, colour = Class))
g5 = g5 + geom_point() + theme_bw()
g5 = g5 + scale_colour_manual(values = c("black", "red"))
g5 = g5 + labs(x = "U[1]", y = "U[2]")
ggsave(g5, file = "plots/umap_ridge.pdf", width = 4.24, height = 3)


# ------------------------------------------
# ------------------------------------------
