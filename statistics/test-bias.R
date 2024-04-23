rm(list=ls(all=TRUE))

library(readxl)
library(pwr)
library(tidyverse)
library(ggpubr)
library(rstatix)

############################################################################
########################## DATA ARRANGEMENT ################################
############################################################################

model = "/Sup-318/"
map = "PDFF"
epoch = "90"

dir = paste("C:/Users/jpmen/Documents/IDEAL-GAN/output",model,"Ep-",epoch,sep="")
setwd(dir)

ls_sheets = excel_sheets(paste(map,'_ROIs_13_21.xlsx',sep=""))
for (i in c(1:length(ls_sheets)))
{
	roi_data=read_excel(paste(map,'_ROIs_13_21.xlsx',sep=""),sheet=ls_sheets[i])
	len_ij = length(t(roi_data[,2]))
	if (i==1)
	{
		refs = c(t(roi_data[,1]))
		meas = c(t(roi_data[,2]))
	} else
	{
		refs = c(refs,t(roi_data[,1]))
		meas = c(meas,t(roi_data[,2]))
	}
}
# meas = meas/length(ls_sheets)
n_data = length(refs)

# Create a data frame
pdff_Data <- data.frame(
mean = c(meas),
refs = c(refs),
bias = c(meas-refs),
roi = factor(c(rep(c("RHL"),len_ij),rep(c("LHL"),len_ij)))
)

############################################################################
########################## BLAND ALTMAN PLOT ###############################
############################################################################
mean_diff <- mean(pdff_Data$bias)
lower <- mean_diff - 1.96*sd(pdff_Data$bias)
upper <- mean_diff + 1.96*sd(pdff_Data$bias)
if (map=="PDFF"){
  yl = 0.1
  xl = 0.33
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