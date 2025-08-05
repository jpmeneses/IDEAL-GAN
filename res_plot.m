clearvars('-except','location'), clc
% clearvars, clc
% location = uigetdir();

load([location,'/results_MP_GC_IM_0007/IM_0007_MP_GC.mat'])
% load([location,'/site6_1p5T_protocol2.mat'])
F_gt = F; P_gt = P; R_gt = R; R2_gt = R2;
% msk=abs(mean(imDataAll.images,5));
% F_gt=fwmc_ff.*(msk>10); R2_gt=fwmc_r2star.*(msk>10); 
% F_gt = F_gt(1:end-mod(size(F_gt,1),16),:,:);
% R2_gt = R2_gt(1:end-mod(size(R2_gt,1),16),:,:);
% P_gt=zeros(size(F_gt)); 

load([location,'/2D_NSA1_ORIG_IM_0007/res_MP_UNet_0007.mat'])
% load([location,'/results_MP_DL/res_MP_UNet_jpmen.mat'])
F_gt = F_gt.*(F(:,:,end:-1:1)>0);
P_gt = P_gt.*(P(:,:,end:-1:1)>0);
R_gt = R_gt.*(R(:,:,end:-1:1,:)>0);
R2_gt = R2_gt.*(R2(:,:,end:-1:1)>0);

figure(1), imshow3D(F)

%% Quantitative maps
n = 10;
figure(2)
t = tiledlayout(3,3,'TileSpacing','tight','Padding','tight');
ax1=nexttile; imagesc(F(:,:,n),[0,100]), colormap(ax1,'turbo'),
title('PDFF','FontSize',12), axis off
% ax2=nexttile; imagesc(R(:,:,n,1),[0,1]), colormap(ax2,'bone'),
% title('|\rho_W|','FontSize',10), axis off
% ax3=nexttile; imagesc(R(:,:,n,2),[0,1]), colormap(ax3,'pink'), 
% title('|\rho_F|','FontSize',10), axis off
ax4=nexttile; imagesc(R2(:,:,n),[0,200]), colormap(ax4,'copper'),
title('R2*','FontSize',12), axis off
ax5=nexttile; imagesc(P(:,:,n),[-150,150]), colormap(ax5,'nebula'),
title('\phi','FontSize',12), axis off

ax6=nexttile; imagesc(F_gt(:,:,end-n+1),[0,100]), colormap(ax6,'turbo'),
axis off, % c6 = colorbar('southoutside','FontSize',9); 
% c6.Label.String = '%'; c6.Label.FontSize = 10; 
% ax7=nexttile; imagesc(abs(R_gt(:,:,end-n,1))), colormap(ax7,'bone'), 
% c7 = colorbar('southoutside','FontSize',9); axis off
% ax8=nexttile; imagesc(abs(R_gt(:,:,end-n,2))), colormap(ax8,'pink'), 
% c8 = colorbar('southoutside','FontSize',9); axis off
ax9=nexttile; imagesc(R2_gt(:,:,end-n+1),[0,200]), colormap(ax9,'copper'),
axis off, % c9 = colorbar('southoutside','FontSize',9);
% c9.Label.String = 'Hz'; c9.Label.FontSize = 10; 
ax10=nexttile; imagesc(P_gt(:,:,end-n+1),[-150,150]), colormap(ax10,'nebula'),
axis off, % c10 = colorbar('southoutside','FontSize',9); 
% c10.Label.String = 'Hz'; c10.Label.FontSize = 10; 

ax11=nexttile; imagesc(abs(F(:,:,n)-F_gt(:,:,end-n+1)),[0,100]), 
colormap(ax11,'turbo'), c11 = colorbar('southoutside','FontSize',11);
c11.Label.String = '%'; c11.Label.FontSize = 12; axis off
% ax12=nexttile; imagesc(abs(R_gt(:,:,end-n,1))), colormap(ax12,'bone'), 
% c12 = colorbar('southoutside','FontSize',9); axis off
% ax13=nexttile; imagesc(abs(R_gt(:,:,end-n,2))), colormap(ax13,'pink'), 
% c13 = colorbar('southoutside','FontSize',9); axis off
ax14=nexttile; imagesc(abs(R2(:,:,n)-R2_gt(:,:,end-n+1)),[0,200]),
colormap(ax14,'copper'), c14 = colorbar('southoutside','FontSize',11);
c14.Label.String = 'Hz'; c14.Label.FontSize = 12; axis off
ax15=nexttile; imagesc((P(:,:,n)-P_gt(:,:,end-n+1)),[-150,150]),
colormap(ax15,'nebula'), c15 = colorbar('southoutside','FontSize',11); 
c15.Label.String = 'Hz'; c15.Label.FontSize = 12; axis off

%% Uncertainty maps
figure(3)
t2 = tiledlayout(1,3,'TileSpacing','tight','Padding','tight');
bx1=nexttile; imagesc(F_var(:,:,n)./1e4,[1e-2,2e1]), set(gca,'ColorScale','log') 
colormap(bx1,'parula'), cc1 = colorbar('southoutside','FontSize',11); 
title('PDFF Var','FontSize',12), axis off
bx4=nexttile; imagesc(R2_var(:,:,n)./(200^2),[1e-4,2e-1]), set(gca,'ColorScale','log') 
colormap(bx4,'autumn'), cc4 = colorbar('southoutside','FontSize',11);
title('R2* Var','FontSize',12), axis off
bx5=nexttile; imagesc(P_var(:,:,n),[1e-5,2e-2]), set(gca,'ColorScale','log') 
colormap(bx5,'spring'), cc5 = colorbar('southoutside','FontSize',11);
title('\phi Var','FontSize',12), axis off

%% Isolated q-map
figure(4)
% imagesc(F(:,:,n),[0,100]), colormap('turbo'), 
% imagesc(F_gt(:,:,end-n+1),[0,100]), colormap('turbo'), 
imagesc(abs(F(:,:,n)-F_gt(:,:,end-n+1)),[0,20]), colormap('turbo'),
% imagesc(F_var(:,:,n)./1e4,[1e-2,2e1]), set(gca,'ColorScale','log'), colormap('parula')
colorbar('FontSize',14); axis off

%% Multiple linear regression 
% Check wether there is correlation between error, ROI-sd, and variance
% https://au.mathworks.com/help/stats/regress.html
X1a = PDFF_var_JGalgani_ROIs.Q2;
X1b = PDFF_var_ROIs_13_21.Q2;
X2a = PDFF_var_JGalgani_ROIs.Q3 - PDFF_var_JGalgani_ROIs.Q1;
X2b = PDFF_var_ROIs_13_21.Q3 - PDFF_var_ROIs_13_21.Q1;
X1 = [X1a; X1b]; X2 = [X2a; X2b];

y_a = PDFF_var_JGalgani_ROIs.PDFFVar; y_b = PDFF_var_ROIs_13_21.PDFFVar;
y = [y_a; y_b];

% Perform multiple linear regression analysis
[b,bint,r,rint,stats] = regress(y, [ones(size(X1, 1), 1), X1, X2]);

figure(5)
scatter3(X1,X2,y,'filled')
hold on
x1fit = min(X1):1e-3:max(X1);
x2fit = min(X2):1e-3:max(X2);
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT;
mesh(X1FIT,X2FIT,YFIT)
xlabel('PDFF error','FontSize',16)
ylabel('PDFF IQR','FontSize',16)
zlabel('PDFF Var','FontSize',16)
ax = gca; ax.FontSize = 14;
view(50,10)
hold off

contain0 = (rint(:,1)<0 & rint(:,2)>0);
idx = find(contain0==false)

figure(6)
hold on
scatter(y,r)
scatter(y(idx),r(idx),'b','filled')
xlabel("PDFF error")
ylabel("PDFF IQR")
hold off