clearvars, clc
location = uigetdir();

load([location,'/results_MP_GC_IM_0014/IM_0014_MP_GC.mat'])
F_gt = F; P_gt = P; R_gt = R; R2_gt = R2;

load([location,'/2D_NSA1_ORIG_IM_0014/res_MP_AIDEAL_0014.mat'])
F_gt = F_gt.*(F(:,:,end:-1:1)>0);
P_gt = P_gt.*(P(:,:,end:-1:1)>0);
R_gt = R_gt.*(R(:,:,end:-1:1,:)>0);
R2_gt = R2_gt.*(R2(:,:,end:-1:1)>0);

figure(1), imshow3D(F)

%% Quantitative maps
n = 12;
figure(2)
t = tiledlayout(3,3,'TileSpacing','tight','Padding','compact');
ax1=nexttile; imagesc(F(:,:,n),[0,100]), colormap(ax1,'turbo'),
title('PDFF','FontSize',10), axis off
% ax2=nexttile; imagesc(R(:,:,n,1),[0,1]), colormap(ax2,'bone'),
% title('|\rho_W|','FontSize',10), axis off
% ax3=nexttile; imagesc(R(:,:,n,2),[0,1]), colormap(ax3,'pink'), 
% title('|\rho_F|','FontSize',10), axis off
ax4=nexttile; imagesc(R2(:,:,n),[0,200]), colormap(ax4,'copper'),
title('R2*','FontSize',10), axis off
ax5=nexttile; imagesc(P(:,:,n),[-150,150]), colormap(ax5,'nebula'),
title('\phi','FontSize',10), axis off

ax6=nexttile; imagesc(F_gt(:,:,end-n),[0,100]), colormap(ax6,'turbo'),
axis off, % c6 = colorbar('southoutside','FontSize',9); 
% c6.Label.String = '%'; c6.Label.FontSize = 10; 
% ax7=nexttile; imagesc(abs(R_gt(:,:,end-n,1))), colormap(ax7,'bone'), 
% c7 = colorbar('southoutside','FontSize',9); axis off
% ax8=nexttile; imagesc(abs(R_gt(:,:,end-n,2))), colormap(ax8,'pink'), 
% c8 = colorbar('southoutside','FontSize',9); axis off
ax9=nexttile; imagesc(R2_gt(:,:,end-n),[0,200]), colormap(ax9,'copper'),
axis off, % c9 = colorbar('southoutside','FontSize',9);
% c9.Label.String = 'Hz'; c9.Label.FontSize = 10; 
ax10=nexttile; imagesc(P_gt(:,:,end-n),[-150,150]), colormap(ax10,'nebula'),
axis off, % c10 = colorbar('southoutside','FontSize',9); 
% c10.Label.String = 'Hz'; c10.Label.FontSize = 10; 

ax11=nexttile; imagesc(abs(F(:,:,n)-F_gt(:,:,end-n)),[0,100]), 
colormap(ax11,'turbo'), c11 = colorbar('southoutside','FontSize',9);
c11.Label.String = '%'; c11.Label.FontSize = 10; axis off
% ax12=nexttile; imagesc(abs(R_gt(:,:,end-n,1))), colormap(ax12,'bone'), 
% c12 = colorbar('southoutside','FontSize',9); axis off
% ax13=nexttile; imagesc(abs(R_gt(:,:,end-n,2))), colormap(ax13,'pink'), 
% c13 = colorbar('southoutside','FontSize',9); axis off
ax14=nexttile; imagesc(abs(R2(:,:,n)-R2_gt(:,:,end-n)),[0,200]),
colormap(ax14,'copper'), c14 = colorbar('southoutside','FontSize',9);
c14.Label.String = 'Hz'; c14.Label.FontSize = 10; axis off
ax15=nexttile; imagesc((P(:,:,n)-P_gt(:,:,end-n)),[-150,150]),
colormap(ax15,'nebula'), c15 = colorbar('southoutside','FontSize',9); 
c15.Label.String = 'Hz'; c15.Label.FontSize = 10; axis off

%% Uncertainty maps
figure(3)
t2 = tiledlayout(1,3,'TileSpacing','tight','Padding','compact');
bx1=nexttile; imagesc(F_var(:,:,n),[1e2,1e5]), set(gca,'ColorScale','log') 
colormap(bx1,'parula'), cc1 = colorbar('southoutside','FontSize',9); 
title('PDFF Var','FontSize',10), axis off
bx4=nexttile; imagesc(R2_var(:,:,n),[1e1,1e4]), set(gca,'ColorScale','log') 
colormap(bx4,'autumn'), cc4 = colorbar('southoutside','FontSize',9);
title('R2* Var','FontSize',10), axis off
bx5=nexttile; imagesc(P_var(:,:,n),[1e-5,1e-1]), set(gca,'ColorScale','log') 
colormap(bx5,'cool'), cc5 = colorbar('southoutside','FontSize',9);
title('\phi Var','FontSize',10), axis off