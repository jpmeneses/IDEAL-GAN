clearvars, clc

cnn_range = 0:1e-4:1;
r2s_range = 0:1e-2:200;
r2s_gt = [5,20,50,80,120,200];
te_n = 5.4e-3; % 3rd echo

for i = 1:length(r2s_gt)
    L = (exp(-r2s_gt(i)*te_n)-exp(-r2s_range*te_n)).^2;
    d_L = 2*(exp(-r2s_gt(i)*te_n)-exp(-r2s_range*te_n)).*(te_n*r2s_range);
    subplot(2,3,i), plot(r2s_range,L,'LineWidth',2), hold on, grid on
    plot(r2s_range,d_L,'LineWidth',2), hold off, legend('L','d_L')
    title(['R2* = ',num2str(r2s_gt(i))])
end