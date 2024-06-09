rm(list=ls(all=TRUE))

library(readxl)
library(tidyverse)
library(ggpubr)
library(lme4)
library(car)

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
		#refs = c(t(roi_data[,1]))
		meas = c(t(roi_data[,2]))
		sample_id = c(1:len_ij)
		roi_id = rep(c(j),length(t(roi_data[,1])))
		meth_id = rep(c(k),length(t(roi_data[,1])))
		TEs = rep(c(TEs_suffix[i]),len_ij)
		meas = c(meas,t(roi_data[,1]))
		sample_id = append(sample_id,c(1:len_ij))
		roi_id = append(roi_id,rep(c(j),length(t(roi_data[,1]))))
		meth_id = append(meth_id,rep(c(length(models)+1),length(t(roi_data[,1]))))
		TEs = append(TEs,rep(c(TEs_suffix[i]),len_ij))
	} else
	{
		meas = c(meas,t(roi_data[,2]))
		sample_id = append(sample_id,c(1:len_ij))
		roi_id = append(roi_id,rep(c(j),length(t(roi_data[,1]))))
		meth_id = append(meth_id,rep(c(k),length(t(roi_data[,1]))))
		TEs = append(TEs,rep(c(TEs_suffix[i]),len_ij))
	}
    } else
    {
      #refs = c(refs,t(roi_data[,1]))
      meas = c(meas,t(roi_data[,2]))
	sample_id = c(sample_id,c(1:len_ij)+(j-1)*len_ij)
	roi_id = append(roi_id,rep(c(j),length(t(roi_data[,1]))))
	meth_id = append(meth_id,rep(c(k),length(t(roi_data[,1]))))
      TEs = c(TEs,rep(c(TEs_suffix[i]),len_ij))
	if (k==1)
	{
		meas = c(meas,t(roi_data[,1]))
		sample_id = c(sample_id,c(1:len_ij)+(j-1)*len_ij)
		roi_id = append(roi_id,rep(c(j),length(t(roi_data[,1]))))
		meth_id = append(meth_id,rep(c(length(models)+1),length(t(roi_data[,1]))))
		TEs = append(TEs,rep(c(TEs_suffix[i]),len_ij))
	}
    }
  }
}
}
n_data = length(meas)

# Create a data frame
pdff_Data <- data.frame(
meas = c(meas)*100,
sample_id = factor(sample_id,
			 labels=c(paste0(LETTERS[1:len_ij],sep='-',1),
				    paste0(LETTERS[1:len_ij],sep='-',2))
			 ),
id = factor(sample_id, labels=rep(LETTERS[1:len_ij],2)),
roi = factor(roi_id, labels=c("RHL","LHL")),
method = factor(c(meth_id), labels=c("2D-Net","U-Net","MDWF-Net",
						"VET-Net","GraphCuts")),
TEs = factor(TEs)
)

# Remove zero-values and separate by ROIs
pdff_RHL = subset(pdff_Data, meas > 0.0 & roi == 'RHL')
pdff_LHL = subset(pdff_Data, meas > 0.0 & roi == 'LHL')

############################################################################
############################## RHL ANALYSIS ################################
############################################################################
pdff_RHL_RDCs <- pdff_RHL %>%
  group_by(method, id, .add=TRUE) %>%
  summarise(RDC = 1.96*sd(meas),
		mmeas = mean(meas),
		n = factor(n()))
# print(pdff_RHL_RDCs, n=80)

pdff_RHL_RDC_byMeth <- pdff_RHL_RDCs %>%
  group_by(method) %>%
  summarise(mRDC = mean(RDC),
		CI_low = mRDC-1.96*sd(RDC),
		CI_high = mRDC+1.96*sd(RDC))
print(pdff_RHL_RDC_byMeth)

## KRUSKAL-WALLIS TEST - DOES THE METHOD AFFECT THE MEAN RDC?
# Check homoscedastacity
leveneTest(RDC ~ method, data = pdff_RHL_RDCs, center = "median")
# Non-parametric test
kruskal.test(RDC ~ method, data = pdff_RHL_RDCs)

sel_meth = "GraphCuts"
pdff_RHL_meth = subset(pdff_RHL_RDCs, method == sel_meth)
if (map=="PDFF") {yl=8.5} else {yl=40.0}
q1 = ggplot(pdff_RHL_meth, aes(mmeas, RDC)) +
       geom_point(colour="red")+
       theme(text = element_text(size = 14)) +
	 ylim(0.0,yl) +
	 ylab("Mean RDC") +
	 xlab("Mean PDFF") 
fn1 = paste(map,sel_meth,"RDC-LR-RHL.png",sep="-")
ggsave(plot=q1, width=4, height=3, dpi=400, filename=fn1)

############################################################################
############################## LHL ANALYSIS ################################
############################################################################
pdff_LHL_RDCs <- pdff_LHL %>%
  group_by(method, id, .add=TRUE) %>%
  summarise(RDC = 1.96*sd(meas),
		mmeas = mean(meas),
		n = factor(n()))
# print(pdff_LHL_RDCs, n=80)

pdff_LHL_RDC_byMeth <- pdff_LHL_RDCs %>%
  group_by(method) %>%
  summarise(mRDC = mean(RDC),
		CI_low = mRDC-1.96*sd(RDC),
		CI_high = mRDC+1.96*sd(RDC))
print(pdff_LHL_RDC_byMeth)

## KRUSKAL-WALLIS TEST - DOES THE METHOD AFFECT THE MEAN RDC?
# Check homoscedastacity
leveneTest(RDC ~ method, data = pdff_LHL_RDCs, center = "median")
# Non-parametric test
kruskal.test(RDC ~ method, data = pdff_LHL_RDCs)
# Post-hoc comparison
pairwise.wilcox.test(x=pdff_LHL_RDCs$RDC, g=pdff_LHL_RDCs$method, p.adjust.method = "holm")

# Mean PDFF v/s Mean RDC plot
pdff_LHL_meth = subset(pdff_LHL_RDCs, method == sel_meth)
q2 = ggplot(pdff_LHL_meth, aes(mmeas, RDC)) +
       geom_point(colour="blue")+
       theme(text = element_text(size = 14)) +
	 ylim(0.0,yl) +
	 ylab("Mean RDC") +
	 xlab("Mean PDFF")
fn2 = paste(map,sel_meth,"RDC-LR-LHL.png",sep="-")
ggsave(plot=q2, width=4, height=3, dpi=400, filename=fn2)