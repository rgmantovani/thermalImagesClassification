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


# ------------------------------------------
# ------------------------------------------
