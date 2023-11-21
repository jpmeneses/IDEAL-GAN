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

model = "/Sup-200/"
map = "PDFF"
epoch = "100"

dir = paste("C:/Users/jpmen/Documents/IDEAL-GAN/output",model,"Ep-",epoch,sep="")
setwd(dir)

filename = paste(c(map,'_test_ROIs.xlsx'),collapse="")
ROI_sheets = excel_sheets(filename)
for (j in c(1:length(ROI_sheets)))
{
  roi_data=read_excel(filename,sheet=ROI_sheets[j])
  len_ij = length(t(roi_data[,2]))
  if (j==1)
  {
	refs = c(t(roi_data[,1]))
	meas = c(t(roi_data[,2]))
  } else
  {
      refs = c(refs,t(roi_data[,1]))
      meas = c(meas,t(roi_data[,2]))
  }
}
n_data = length(meas)

# Create a data frame
pdff_Data <- data.frame(
refs = c(refs),
meas = c(meas),
bias = c(meas - refs),
mean = c((meas + refs)/2),
roi = factor(c(rep(c("RHL"),len_ij),rep(c("LHL"),len_ij)))
)

# Remove zero-values
pdff_Data = subset(pdff_Data, meas > 0.0)

# Dataset summary stats
pdff_Data %>%
  group_by(roi) %>%
  get_summary_stats(meas, type="common")

############################################################################
############################## SCATTERPLOT #################################
############################################################################
q2 = ggplot(pdff_Data, aes(refs, meas)) +
      geom_point()+
      geom_smooth(method="lm",fullrange=TRUE) +
	labs(x='Reference', y='Measurement') +
	theme(text = element_text(size = 12)) +
	xlim(0.0,0.4) +
	ylim(0.0,0.4) +
  stat_regline_equation(
    aes(label = paste(..eq.label.., ..rr.label.., sep="~~~~"))
    )
fn2 = paste(map,"Test-corr.png",sep="-")
ggsave(plot=q2, width=3.2, height=2.8, dpi=500, filename=fn2)

############################################################################
########################## BLAND ALTMAN PLOT ###############################
############################################################################
mean_diff <- mean(pdff_Data$bias)
lower <- mean_diff - 1.96*sd(pdff_Data$bias)
upper <- mean_diff + 1.96*sd(pdff_Data$bias)
if (map=="PDFF"){
  yl = 0.06
  xl = 0.35
} else {
  yl = 35.0
  xl = 115.0
}
q2= ggplot(pdff_Data, aes(mean, bias)) +
  geom_point(size=1.1) +
  geom_hline(yintercept = mean_diff) +
  geom_hline(yintercept = lower, color = "red", linetype="dashed") +
  geom_hline(yintercept = upper, color = "red", linetype="dashed") +
  theme(text = element_text(size = 17)) +
  ylim(-yl,yl) +
  xlim(0.0,xl) +
  ylab("Difference") +
  xlab("Mean")
fn2 = paste(map,"Test-BlandAltman.png",sep="-")
ggsave(plot=q2, width=4, height=3, dpi=400, filename=fn2)
