rm(list=ls(all=TRUE))

library(tidyverse)
library(readxl)
library(lme4)

############################################################################
########################## DATA ARRANGEMENT ################################
############################################################################

models = c("/Sup-200/","/Sup-202/","/Sup-204/","/TEaug-300/")
map = "PDFF"
epoch = "100"

for (k in c(1:length(models)))
{
dir = paste("C:/Users/jpmen/Documents/IDEAL-GAN/output",models[k],"Ep-",epoch,sep="")
setwd(dir)

ls_sheets = excel_sheets(paste(map,'_phantom_ROIs.xlsx',sep=""))
for (i in c(1:length(ls_sheets)))
{
	roi_data=read_excel(paste(map,'_phantom_ROIs.xlsx',sep=""),sheet=ls_sheets[i])
	if (i==1 && k==1)
	{
		if (map=="PDFF") {refs = c(t(roi_data[,1]))}
		else {refs = c(t(roi_data[,2]))}
		meas = c(t(roi_data[,3]))
		im_id = rep(c(i),length(t(roi_data[,1])))
		vial_id = c(t(1:length(t(roi_data[,1]))))
		meth_id = rep(c(k),length(t(roi_data[,1])))
		if (k==1)
		{
			if (map=="PDFF") {refs = c(refs,t(roi_data[,1]))}
			meas = c(meas,t(roi_data[,2]))
			im_id = append(im_id,rep(c(i),length(t(roi_data[,1]))))
			vial_id = c(vial_id,t(1:length(t(roi_data[,1]))))
			meth_id = append(meth_id,rep(c(length(models)+1),length(t(roi_data[,1]))))
		}
	} else
	{
		# meas = meas + c(t(roi_data[,3]))
		if (map=="PDFF") {refs = c(refs,t(roi_data[,1]))}
		else {refs = c(refs,t(roi_data[,2]))}
		meas = c(meas,t(roi_data[,3]))
		im_id = append(im_id,rep(c(i),length(t(roi_data[,1]))))
		vial_id = c(vial_id,t(1:length(t(roi_data[,1]))))
		meth_id = append(meth_id,rep(c(k),length(t(roi_data[,1]))))
		if (k==1)
		{
			if (map=="PDFF") {refs = c(refs,t(roi_data[,1]))}
			meas = c(meas,t(roi_data[,2]))
			im_id = append(im_id,rep(c(i),length(t(roi_data[,1]))))
			vial_id = c(vial_id,t(1:length(t(roi_data[,1]))))
			meth_id = append(meth_id,rep(c(length(models)+1),length(t(roi_data[,1]))))
		}
	}
}
}
n_data = length(refs)

# Create a data frame
pdff_Data <- data.frame(
refs = c(refs)*100,
bias = c(meas-refs)*100,
method = factor(c(meth_id), labels=c("2D-Net","U-Net","MDWF-Net",
						"VET-Net","GraphCuts")),
Site_Prot = factor(c(im_id),labels=c("S1-P1","S1-P2",
		   "S2-P1","S2-P2","S3-P1","S3-P2","S4-P2","S6-P1","S6-P2"))
)
# factor(c(im_id),labels=c("S1-P1(V1)","S1-P2(V1)","S1-P1(V2)","S1-P2(V2)",
# 		   "S2-P1","S2-P2","S3-P1","S3-P2","S4-P2","S6-P1","S6-P2"))


# DEPICT BIAS DIFFERENCES DEPENDING ON THE USED METHOD
boxplot(bias ~ method, data = pdff_Data)
# pdff_Data <- subset(pdff_Data, method!="MDWF-Net")

## LINEAR MIXED MODEL
mixed.lmer <- lmer(bias ~ refs + (1|Site_Prot) + (1|method), data=pdff_Data)
summary(mixed.lmer)
# plot(mixed.lmer)

full.lmer <- lmer(bias ~ refs + (1|Site_Prot) + (1|method), data=pdff_Data, REML=FALSE)
reduced.lmer <- lmer(bias ~ refs + (1|Site_Prot), data=pdff_Data, REML=FALSE)
anova(reduced.lmer, full.lmer)

## GROUP BY METHOD AND FAT LEVEL
pdff_group <- pdff_Data %>%
  group_by(method) %>%
  summarise(mBias = mean(bias),
		LoA = 1.96*sd(bias),
		n = n())
print(pdff_group, n=55)

## BLAND-ALTMAN BY METHOD
sel_meth = "MDWF-Net"
pdff_meth = subset(pdff_Data, method==sel_meth)
pdff_meth <- within(pdff_meth, meas <- bias+refs)
meth.lmer <- lmer(meas ~ refs + (1|Site_Prot), data=pdff_meth)
summary(meth.lmer)

mean_diff <- mean(pdff_meth$bias)
lower <- mean_diff - 1.96*sd(pdff_meth$bias)
upper <- mean_diff + 1.96*sd(pdff_meth$bias)
if (map=="PDFF") {yl_ba=100.0} else {yl_ba=40.0}
q2= ggplot(pdff_meth, aes(refs, bias)) +
  geom_point(aes(color = Site_Prot))+
  geom_hline(yintercept = mean_diff) +
  geom_hline(yintercept = lower, color = "red", linetype="dashed") +
  geom_hline(yintercept = upper, color = "red", linetype="dashed") +
  theme(text = element_text(size = 14)) +
  ylim(-yl_ba,yl_ba) +
  ylab("Difference") +
  xlab("Ground-Truth")
fn2 = paste(map,sel_meth,"Bias-BlandAltman.png",sep="-")
ggsave(plot=q2, width=5, height=3, dpi=400, filename=fn2)