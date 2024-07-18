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

map = "R2s"
if (map == "PDFF")
{
  models = c("/Sup-200/","/Sup-202/","/Sup-204/","/TEaug-300/")
  k_resc = 100.0
} else
{
  models = c("/Sup-200/","/Sup-204/","/TEaug-300/")
  k_resc = 1.0
}
epoch = "100"

for (k in c(1:length(models)))
{
dir = paste("C:/Users/jpmen/Documents/IDEAL-GAN/output",models[k],"Ep-",epoch,sep="")
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
	if (k==1)
	{
		refs = c(t(roi_data[,1]))
		meas = c(t(roi_data[,2]))
		sample_id = c(1:len_ij)
		roi_id = rep(c(j),length(t(roi_data[,1])))
		meth_id = rep(c(k),length(t(roi_data[,1])))
		TEs = rep(c(TEs_suffix[i]),len_ij)
	} else
	{
		refs = c(t(roi_data[,1]))
		meas = c(meas,t(roi_data[,2]))
		sample_id = append(sample_id,c(1:len_ij))
		roi_id = append(roi_id,rep(c(j),length(t(roi_data[,1]))))
		meth_id = append(meth_id,rep(c(k),length(t(roi_data[,1]))))
		TEs = append(TEs,rep(c(TEs_suffix[i]),len_ij))
	}
    } else
    {
      refs = c(refs,t(roi_data[,1]))
      meas = c(meas,t(roi_data[,2]))
	sample_id = c(sample_id,c(1:len_ij)+(j-1)*len_ij)
	roi_id = append(roi_id,rep(c(j),length(t(roi_data[,1]))))
	meth_id = append(meth_id,rep(c(k),length(t(roi_data[,1]))))
      TEs = c(TEs,rep(c(TEs_suffix[i]),len_ij))
    }
  }
}
}
n_data = length(meas) 

# Create a data frame
pdff_Data <- data.frame(
refs = c(refs)*k_resc,
meas = c(meas)*k_resc,
sample_id = factor(sample_id,
			 labels=c(paste0(LETTERS[1:len_ij],sep='-',1),
				    paste0(LETTERS[1:len_ij],sep='-',2))
			 ),
id = factor(sample_id, labels=rep(LETTERS[1:len_ij],2)),
roi = factor(roi_id, labels=c("RHL","LHL")),
Method = factor(c(meth_id), labels=c("VET-Net (No TE)","MDWF-Net","VET-Net")),
TEs = factor(TEs)
)

# Remove zero-values and separate by ROIs
pdff_RHL = subset(pdff_Data, meas > 0.0 & roi == 'RHL')
pdff_LHL = subset(pdff_Data, meas > 0.0 & roi == 'LHL')

# Calculate mean of all TEs
pdff_RHL_bySubj <- pdff_RHL %>%
  group_by(Method, id, .add=TRUE) %>%
  summarise(mean = mean(meas),
		mean_ref = mean(refs))
# print(pdff_RHL_bySubj, n=60)

pdff_LHL_bySubj <- pdff_LHL %>%
  group_by(Method, id, .add=TRUE) %>%
  summarise(mean = mean(meas),
		mean_ref = mean(refs))
# print(pdff_LHL_bySubj, n=60)

############################################################################
########################### REGRESSION LINES ###############################
############################################################################
q = ggscatter(
  pdff_RHL_bySubj, x="mean_ref", y="mean",
  color="Method", xlab="Mean Ref. R2* [1/s]",
  ylab="Mean Measured R2* [1/s]", add="reg.line",
  )+
  xlim(0.0,150.0) +
  ylim(0.0,150.0) +
  stat_regline_equation(
    aes(label = paste(..eq.label.., ..rr.label.., sep="~~~~"), color=Method)
    )
fn = paste(map,"LR-RHL.png",sep="-")
ggsave(plot=q, width=5, height=5, dpi=400, filename=fn)

q2 = ggscatter(
  pdff_LHL_bySubj, x="mean_ref", y="mean",
  color="Method", xlab="Mean Ref. R2* [1/s]",
  ylab="Mean Measured R2* [1/s]", add="reg.line",
  )+
  xlim(0.0,150.0) +
  ylim(0.0,150.0) +
  stat_regline_equation(
    aes(label = paste(..eq.label.., ..rr.label.., sep="~~~~"), color=Method)
    )
fn2 = paste(map,"LR-LHL.png",sep="-")
ggsave(plot=q2, width=5, height=5, dpi=400, filename=fn2)