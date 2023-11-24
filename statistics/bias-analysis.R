rm(list=ls(all=TRUE))

library(readxl)
library(pwr)
library(tidyverse)
library(ggpubr)
library(rstatix)

############################################################################
########################## DATA ARRANGEMENT ################################
############################################################################

model = "/Sup-204/"
map = "PDFF"
epoch = "100"

dir = paste("C:/Users/jpmen/Documents/IDEAL-GAN/output",model,"Ep-",epoch,sep="")
setwd(dir)

ls_sheets = excel_sheets(paste(map,'_phantom_ROIs.xlsx',sep=""))
for (i in c(1:length(ls_sheets)))
{
	roi_data=read_excel(paste(map,'_phantom_ROIs.xlsx',sep=""),sheet=ls_sheets[i])
	if (i==1)
	{
		if (map=="PDFF") {refs = c(t(roi_data[,1]))}
		else {refs = c(t(roi_data[,2]))}
		meas = c(t(roi_data[,3]))
		im_id = rep(c(i),length(t(roi_data[,1])))
		vial_id = c(t(1:length(t(roi_data[,1]))))
	} else
	{
		# meas = meas + c(t(roi_data[,3]))
		if (map=="PDFF") {refs = c(refs,t(roi_data[,1]))}
		else {refs = c(refs,t(roi_data[,2]))}
		meas = c(meas,t(roi_data[,3]))
		im_id = append(im_id,rep(c(i),length(t(roi_data[,1]))))
		vial_id = c(vial_id,t(1:length(t(roi_data[,1]))))
	}
}
# meas = meas/length(ls_sheets)
n_data = length(refs)

# Create a data frame
pdff_Data <- data.frame(
mean = c(meas),
refs = c(refs),
bias = c(meas-refs),
vial = factor(c(vial_id)),
Site_Prot = factor(c(im_id),labels=c("S1-P1","S1-P2",
		   "S2-P1","S2-P2","S3-P1","S3-P2","S4-P2","S6-P1","S6-P2"))
)
# factor(c(im_id),labels=c("S1-P1(V1)","S1-P2(V1)","S1-P1(V2)","S1-P2(V2)",
# 		   "S2-P1","S2-P2","S3-P1","S3-P2","S4-P2","S6-P1","S6-P2"))

# Dataset summary stats
pdff_Vials <- pdff_Data %>%
  group_by(vial) %>%
  summarise(meanY = mean(mean),
		refsX = mean(refs),
		bias = mean(mean)-mean(refs))
overall_bias = mean(pdff_Vials$bias)
std_bias = sd(pdff_Vials$bias)
cat('Overall bias:',overall_bias,'+-',std_bias,'\n')


############################################################################
########################### REGRESSION LINES ###############################
############################################################################
if (map=="PDFF") {yl=1.0} else {yl=40.0}
q = ggplot(pdff_Data, aes(refs, mean)) +
       geom_point(aes(color = Site_Prot))+
       geom_smooth(method="lm", se=FALSE, fullrange=TRUE) +
	 theme(text = element_text(size = 14)) +
	 ylim(0.0,yl) +
	 ylab("Measurements") +
	 xlab("References") +
  stat_regline_equation(
    aes(label = paste(..eq.label.., ..rr.label.., sep="~~~~"))
    )
fn1 = paste(map,"LS-corr.png",sep="-")
ggsave(plot=q, width=5, height=3, dpi=400, filename=fn1)


############################################################################
########################## BLAND ALTMAN PLOT ###############################
############################################################################
mean_diff <- mean(pdff_Data$bias)
lower <- mean_diff - 1.96*sd(pdff_Data$bias)
upper <- mean_diff + 1.96*sd(pdff_Data$bias)
if (map=="PDFF") {yl_ba=1.0} else {yl_ba=40.0}
q2= ggplot(pdff_Data, aes(refs, bias)) +
  geom_point(aes(color = Site_Prot))+
  geom_hline(yintercept = mean_diff) +
  geom_hline(yintercept = lower, color = "red", linetype="dashed") +
  geom_hline(yintercept = upper, color = "red", linetype="dashed") +
  theme(text = element_text(size = 14)) +
  ylim(-yl_ba,yl_ba) +
  ylab("Difference") +
  xlab("Ground-Truth")
fn2 = paste(map,"Bias-BlandAltman.png",sep="-")
ggsave(plot=q2, width=5, height=3, dpi=400, filename=fn2)