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


dir.create(path = "plots/", recursive=TRUE, showWarnings=FALSE)

# ---------------------------
# ---------------------------

files = list.files(path = "output/") 

performance.files = files[grepl(x = files, pattern = "performance")]

seeds = c(171, 666, 42, 51, 404, 720, 269, 289, 376, 767)

# -----------------
# loading performances
# -----------------

aux = lapply(performance.files, function(jobfile) {
	# print(jobfile)

	df = read.csv(file = paste0("output/",jobfile))
	df = as.data.frame(df)

	if(grepl(x = jobfile, pattern = "svm|ridge|dt|rf|bagging|knn")) {
		df$type_of_image = "rgb"
		algo = gsub(x = jobfile, pattern = "performances_|_seed_|.csv", replacement = "")
		algo = gsub(x = algo, pattern = paste(seeds, collapse="|"), replacement = "")
		df$model = algo
	} 
	return(df)
})


df.performances = do.call("rbind",  aux)

# -----------------
# Boxplot (overall)
# -----------------

g = ggplot(df.performances, aes(x = reorder(model, -fscore), y = fscore, group = model))
g = g + geom_violin() + geom_boxplot(width = .15) + theme_bw()
g = g + geom_hline(yintercept = 0.8, linetype = "dotted", colour = "red")
g = g + labs(x = "Model", y = "FScore)")
g = g + theme(axis.text.x=element_text(angle = 90, hjust = 1, vjust = 0.5))
ggsave(g, filename = "plots/overall_boxplot.pdf", width = 6.4, height = 3.14)


# ---------------
# Plot: top 2 confusion matrices
# ---------------


cat(" @ Plot: Confusion Matrices\n")

# VGG x Ridge

# --------------
# VGG files
# --------------

vgg.files  = files[grepl(x = files, pattern = "predictions_vgg")]

aux.vgg = lapply(vgg.files, function(predfile) {
	data = read.csv(file = paste0("output/", predfile))
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

# Do the same for Ridge
ridge.files = files[grepl(x = files, pattern = "predictions_ridge")]
aux.ridge = lapply(ridge.files, function(predfile) {
	# print(predfile)
	data = read.csv(file = paste0("output/", predfile))
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


df.all = rbind(df.vgg.table, df.ridge.table)


g2 = ggplot(data =  df.all, mapping = aes(x = TClass, y = PClass, alpha = Y, fill = goodbad))
g2 = g2 + geom_tile() + geom_text(aes(label = Y), vjust = .5, fontface  = "bold", alpha = 1)
g2 = g2 + scale_fill_manual(values = c(good = "green", bad = "red"))
g2 = g2 + theme_bw() + theme(legend.position = "none") + facet_grid(~algo)
g2 = g2 + labs(x = "True", y = "Predicted")

ggsave(g2, file = "plots/top2_confusion_matrices.pdf", width = 4.2, height = 2.18)


# -----------------
#  Ridge PCA (tSNE) 2D plot
# -----------------

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

cat(" @ Plot: 2D tSNE plto\n")

# Tsne versions
ids  = !duplicated(dataset.feat[, -c(1:4, ncol(dataset.feat))])
temp =  dataset.feat[ids,]
tsne_out = Rtsne(temp)
df.tsne = data.frame(x = tsne_out$Y[,1], y = tsne_out$Y[,2])
df.tsne$Class = dataset.feat$Class[ids] 

g4 = ggplot(data = df.tsne, mapping = aes(x = x, y = y, shape = Class, colour = Class))
g4 = g4 + geom_point() + theme_bw()
g4 = g4 + scale_colour_manual(values = c("black", "red"))
g4 = g4 + labs(x = "T[1]", y = "T[2]")
ggsave(g4, file = "plots/tsne_ridge.pdf", width = 4.24, height = 3)


# ------------------------------------------
# ------------------------------------------
