rm(list=ls(all=TRUE))

library(readxl)
library(pwr)
library(tidyverse)
library(ggpubr)
library(rstatix)
library(emmeans)

############################################################################
########################## DATA ARRANGEMENT ################################
############################################################################

model = "/TEaug-004/"
epoch = "200"

dir = paste("C:/Users/jpmen/Documents/OT-CycleGAN/output",model,"Ep-",epoch,sep="")
setwd(dir)

TEs_suffix = c('13_21','13_22','13_23','13_24','14_21','14_22')
for (i in c(1:length(TEs_suffix)))
{
  filename = paste(c('PDFF_ROIs_',TEs_suffix[i],'.xlsx'),collapse="")
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
id = wXs$sample_id,
wXs_r = wXs_r$refs,
wSDs_r = wSDs_r$refs,
wCVs_r = wSDs_r$refs/wXs_r$refs,
wXs = wXs$meas,
wSDs = wSDs$meas,
wCVs = wSDs$meas/wXs$meas
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
	geom_errorbar(aes(xmin=wXs_r-wSDs_r, xmax=wXs_r+wSDs_r,
				ymin=wXs-wSDs, ymax=wXs+wSDs),
				width=0.005) +
      geom_smooth(method="lm") +
	labs(x='Ref. ROI mean', y='Meas. ROI mean') +
	ylim(0.02,0.3) +
  stat_regline_equation(
    aes(label = paste(..eq.label.., ..rr.label.., sep="~~~~"))
    )
fn2 = "Precision-corr-std.png"
ggsave(plot=q2, width=3.2, height=2.8, dpi=500, filename=fn2)