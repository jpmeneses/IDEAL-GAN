rm(list=ls(all=TRUE))

library(readxl)
library(pwr)
library(tidyverse)
library(ggpubr)
library(rstatix)
library(gridExtra)
#library(emmeans)

############################################################################
########################## DATA ARRANGEMENT ################################
############################################################################

model = "/TEaug-010/"
map = "PDFF"
epoch = "200"

dir = paste("C:/Users/jpmen/Documents/IDEAL-GAN/output",model,"Ep-",epoch,sep="")
setwd(dir)

TEs_suffix = c('13_21','13_22','13_23','13_24','14_21','14_22')
for (i in c(1:length(TEs_suffix)))
{
  filename = paste(c(map,'_ROIs_',TEs_suffix[i],'.xlsx'),collapse="")
  ROI_sheets = excel_sheets(filename)
  for (j in c(1:length(ROI_sheets)))
  {
    roi_data=read_excel(filename,sheet=ROI_sheets[j])
    len_ij = length(t(roi_data[,2]))
    if (i==1 & j==1)
    {
      refs = c(t(roi_data[,1]))
	meas = c(t(roi_data[,2]))
	sample_id = c(1:len_ij)
      TEs = rep(c(TEs_suffix[i]),len_ij)
    } else
    {
      refs = c(refs,t(roi_data[,1]))
      meas = c(meas,t(roi_data[,2]))
	sample_id = c(sample_id,c(1:len_ij)+(j-1)*len_ij)
      TEs = c(TEs,rep(c(TEs_suffix[i]),len_ij))
    }
  }
}
n_data = length(meas)

# Create a data frame
pdff_Data <- data.frame(
refs = c(refs),
meas = c(meas),
sample_id = factor(sample_id,
			 labels=c(paste0(LETTERS[1:len_ij],sep='-',1),
				    paste0(LETTERS[1:len_ij],sep='-',2))
			 ),
id = factor(sample_id, labels=rep(LETTERS[1:len_ij],2)),
roi = factor(c(rep(c("RHL"),len_ij),rep(c("LHL"),len_ij))),
TEs = factor(TEs)
)

# Remove zero-values
pdff_Data = subset(pdff_Data, meas > 0.0)

# Dataset summary stats
pdff_Data %>%
  group_by(sample_id) %>%
  get_summary_stats(meas, type="common")

# Precision metrics for each sample
wXs = with(pdff_Data, aggregate(meas ~ sample_id, FUN = mean))
wSDs = with(pdff_Data, aggregate(meas ~ sample_id, FUN = sd))
wXs_r = with(pdff_Data, aggregate(refs ~ sample_id, FUN = mean))
wSDs_r = with(pdff_Data, aggregate(refs ~ sample_id, FUN = sd))

res_id <- data.frame(
sample_id = wXs$sample_id,
id = factor(sample_id, labels=rep(LETTERS[1:len_ij],2)),
roi = factor(c(rep(c("RHL"),len_ij),rep(c("LHL"),len_ij))),
wXs_r = wXs_r$refs,
wSDs_r = wSDs_r$refs,
wCVs_r = wSDs_r$refs/wXs_r$refs,
wXs = wXs$meas,
wSDs = wSDs$meas,
wCVs = wSDs$meas/wXs$meas,
bias = wXs$meas - wXs_r$refs,
mean = (wXs$meas + wXs_r$refs)/2
)

# Summarized precision metrics
mean_wSD = sqrt(mean(res_id$wSDs^2))
std_wSD = sqrt(sd(res_id$wSDs^2))
mean_wCV = sqrt(mean(res_id$wCVs^2))
std_wCV = sqrt(sd(res_id$wCVs^2))
mean_RDC = 2.77*mean_wSD
std_RDC = 2.77*mean_wSD
mean_pRDC= 2.77*mean_wCV
std_pRDC= 2.77*std_wCV

cat('Mean wSD:',mean_wSD,'+-',1.96*std_wSD,
	'\nMean wCV:',mean_wCV,'+-',1.96*std_wCV,
	'\nMean RDC:',mean_RDC,'+-',1.96*std_RDC,
	'\nMean %RDC:',mean_pRDC,'+-',1.96*std_pRDC,'\n')

############################################################################
############################## SCATTERPLOT #################################
############################################################################
upper_sd <- mean_wSD + 1.65*std_wSD
q2 = ggplot(res_id, aes(wXs_r, wXs)) +
      geom_point()+
	# geom_errorbar(aes(xmin=wXs_r-wSDs_r, xmax=wXs_r+wSDs_r,
	# 			ymin=wXs-wSDs, ymax=wXs+wSDs),
	# 			width=0.005) +
      geom_smooth(method="lm") +
	labs(x='Reference', y='Measurement') +
	theme(text = element_text(size = 12)) +
	xlim(0.0,0.3) +
	ylim(0.0,0.3) +
  stat_regline_equation(
    aes(label = paste(..eq.label.., ..rr.label.., sep="~~~~"))
    )
fn2 = paste(map,"Precision-corr-std.png",sep="-")
ggsave(plot=q2, width=3.2, height=2.8, dpi=500, filename=fn2)

############################################################################
########################## BLAND ALTMAN PLOT ###############################
############################################################################
mean_diff <- mean(res_id$bias)
lower <- mean_diff - 1.96*sd(res_id$bias)
upper <- mean_diff + 1.96*sd(res_id$bias)
if (map=="PDFF"){
  yl = 0.03
  xl = 0.33
} else {
  yl = 35.0
  xl = 115.0
}
q2= ggplot(res_id, aes(mean, bias)) +
  geom_point(size=1.1) +
  geom_hline(yintercept = mean_diff) +
  geom_hline(yintercept = lower, color = "red", linetype="dashed") +
  geom_hline(yintercept = upper, color = "red", linetype="dashed") +
  theme(text = element_text(size = 17)) +
  ylim(-yl,yl) +
  xlim(0.0,xl) +
  ylab("Difference") +
  xlab("Mean")
fn2 = paste(map,"Precision-BlandAltman.png",sep="-")
ggsave(plot=q2, width=4, height=3, dpi=400, filename=fn2)

############################################################################
########################## Within-subject SD ###############################
############################################################################

subjects = c(LETTERS[1:len_ij])
if (map=="PDFF"){
  zero_val = 0.15
  yl_mx = 0.35
} else {
  zero_val = 30.0
  yl_mx = 125.0
}
zero_data <- data.frame(
id = c("RDC"),
mean = c(zero_val),
wSD = c(mean_wSD)
)
# par(mfrow = c(1,len_ij))
zero_plot <- ggplot(zero_data, aes(id, mean)) + 
	geom_errorbar(aes(ymin=mean-2.77*wSD, ymax=mean+2.77*wSD),
			size=1.5, width=0.5, color="#CC9900") +
	geom_point(size=5) +
	labs(x="RDC", y=map) +
	ylim(0.0,yl_mx) + 
	theme(axis.title.y = element_blank(),
		axis.text.x = element_blank(),
		plot.margin = unit(c(2.0,.0,2.0,.0), "pt"),
		text = element_text(size=20,face="bold"))
for (each_subject in subjects) {
  sub_Data = subset(res_id, id==each_subject)
  plot_var_name = paste("ggplot_",each_subject,sep="")
  temp_plot <- ggplot(sub_Data, aes(roi, mean)) + 
	geom_errorbar(aes(ymin=mean-2.77*wSDs, ymax=mean+2.77*wSDs),
			size=1.5, width=0.5, color="#CC9900") +
	geom_point(size=5, aes(color = roi), show.legend = FALSE) +
	labs(x=each_subject) +
	ylim(0.0,yl_mx) + 
	theme(axis.title.y = element_blank(),
		axis.text.x = element_blank(),
		plot.margin = unit(c(2.0,.0,2.0,.0), "pt"),
		axis.ticks.y = element_blank(),
        	axis.text.y = element_blank(),
		text = element_text(size=20))
  
  # Sys.sleep(0.2)
  assign(plot_var_name, temp_plot)
}
fn3 = paste(map,"RDC-distrib.png",sep="-")
png(fn3, width = 1440, height = 400, units = "px")
gridExtra::grid.arrange(zero_plot,ggplot_A, ggplot_B, ggplot_C, ggplot_D,
				ggplot_E, ggplot_F, ggplot_G, ggplot_H,
				ggplot_I, ggplot_J, ggplot_K, ggplot_L,
				ggplot_M, ggplot_N, ggplot_O, ggplot_P,
				ggplot_Q, ggplot_R, ggplot_S, ggplot_T, ncol=len_ij+1)

g <- arrangeGrob(zero_plot, ggplot_A, ggplot_B, ggplot_C, ggplot_D,
			ggplot_E, ggplot_F, ggplot_G, ggplot_H,
			ggplot_I, ggplot_J, ggplot_K, ggplot_L,
			ggplot_M, ggplot_N, ggplot_O, ggplot_P,
			ggplot_Q, ggplot_R, ggplot_S, ggplot_T, ncol=len_ij+1)
#ggsave(file=fn3, g) #saves g
dev.off()