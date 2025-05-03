# ------------------------------------------
# ------------------------------------------


library("dplyr")
library("reshape2")
library("ggplot2")

# ---------------------------
# ---------------------------

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
	print(jobfile)

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
ggsave(g, filename = "plots/overall_boxplot.pdf"), width = 6.4, height = 3.14)


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




# -----------------
# Plot: problematic images
# -----------------


# ------------------------------------------
# ------------------------------------------
