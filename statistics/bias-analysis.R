rm(list=ls(all=TRUE))

library(readxl)
library(pwr)
library(tidyverse)
library(ggpubr)
library(rstatix)

############################################################################
########################## DATA ARRANGEMENT ################################
############################################################################

model = "/Sup-009/"
epoch = "200"

dir = paste("C:/Users/jpmen/Documents/IDEAL-GAN/output",model,"Ep-",epoch,sep="")
setwd(dir)

ls_sheets = excel_sheets('PDFF_phantom_ROIs.xlsx')
for (i in c(1:length(ls_sheets)))
{
	roi_data=read_excel('PDFF_phantom_ROIs.xlsx',sheet=ls_sheets[i])
	if (i==1)
	{
		refs = c(t(roi_data[,1]))
		meas = c(t(roi_data[,3]))
		im_id = rep(c(i),length(t(roi_data[,1])))
	} else
	{
		# meas = meas + c(t(roi_data[,3]))
		refs = c(refs,t(roi_data[,1]))
		meas = c(meas,t(roi_data[,3]))
		im_id = append(im_id,rep(c(i),length(t(roi_data[,1]))))
	}
}
# meas = meas/length(ls_sheets)
n_data = length(refs)

# Create a data frame
pdff_Data <- data.frame(
mean = c(meas),
refs = c(refs),
bias = c(meas-refs),
Site_Protocol = factor(c(im_id),labels=c("S1-P1","S1-P2",
		   "S2-P1","S2-P2","S3-P1","S3-P2","S6-P1","S6-P2"))
)
# factor(c(im_id),labels=c("S1-P1(V1)","S1-P2(V1)","S1-P1(V2)","S1-P2(V2)",
# 		   "S2-P1","S2-P2","S3-P1","S3-P2","S4-P2","S6-P1","S6-P2"))

# Dataset summary stats
pdff_Data %>%
  group_by(Site_Protocol) %>%
  get_summary_stats(mean, type="common")
 
############################################################################
########################### REGRESSION LINES ###############################
############################################################################
q = ggplot(pdff_Data, aes(refs, mean)) +
       geom_point(aes(color = Site_Protocol))+
       geom_smooth(method="lm") +
  stat_regline_equation(
    aes(label = paste(..eq.label.., ..rr.label.., sep="~~~~"))
    )
fn1 = "LS-corr.png"
ggsave(plot=q, width=6, height=4, dpi=1200, filename=fn1)


############################################################################
########################## BLAND ALTMAN PLOT ###############################
############################################################################
mean_diff <- mean(pdff_Data$bias)
lower <- mean_diff - 1.96*sd(pdff_Data$bias)
upper <- mean_diff + 1.96*sd(pdff_Data$bias)
q2= ggplot(pdff_Data, aes(refs, bias)) +
  geom_point(aes(color = Site_Protocol))+
  geom_hline(yintercept = mean_diff) +
  geom_hline(yintercept = lower, color = "red", linetype="dashed") +
  geom_hline(yintercept = upper, color = "red", linetype="dashed") +
  ylab("Difference Between Measurements") +
  xlab("Ground-Truth") + 
  ylim(-0.3,0.3)
fn2 = "Bias-BlandAltman.png"
ggsave(plot=q2, width=6, height=4, dpi=1200, filename=fn2)