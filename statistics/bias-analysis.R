rm(list=ls(all=TRUE))

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
# meas = meas/length(ls_sheets)
n_data = length(refs)

# Create a data frame
pdff_Data <- data.frame(
refs = c(refs)*100,
bias = c(meas-refs)*100,
method = factor(c(meth_id), labels=c("2D-Net","U-Net","MDWF-Net",
						"VET-Net","GraphCuts")),
Site_Prot = factor(c(im_id),labels=c("S1-P1","S1-P2",
		   "S2-P1","S2-P2","S3-P1","S3-P2","S6-P1","S6-P2"))
)
# factor(c(im_id),labels=c("S1-P1(V1)","S1-P2(V1)","S1-P1(V2)","S1-P2(V2)",
# 		   "S2-P1","S2-P2","S3-P1","S3-P2","S4-P2","S6-P1","S6-P2"))


## LINEAR MIXED MODEL
mixed.lmer <- lmer(bias ~ refs + (1|Site_Prot) + (1|method), data=pdff_Data)
summary(mixed.lmer)

plot(mixed.lmer)

qqnorm(resid(mixed.lmer))
qqline(resid(mixed.lmer))

full.lmer <- lmer(bias ~ refs + (1|Site_Prot) + (1|method), data=pdff_Data, REML=FALSE)
reduced.lmer <- lmer(bias ~ (1|method) + (1|Site_Prot), data=pdff_Data, REML=FALSE)

anova(reduced.lmer, full.lmer)

## GROUP BY METHOD AND FAT LEVEL
pdff_group <- pdff_Data %>%
  group_by(method) %>%
  summarise(mBias = mean(bias),
		LoA = 1.96*sd(bias),
		n = n())
pdff_group