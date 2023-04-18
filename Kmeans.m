%% load dataB of 18 subjuects
clc;clear
filename_18='.\datasets\Diabetes_v6\T1DM327_T2DM110_align_1+576.csv'
xlRange_18='A1:N437';
[data_origin,~,raw] = xlsread(filename_18,'T1DM327_T2DM110_align_1+576',xlRange_18);
% filename_18='.\datasets\Diabetes_v6\T1DM110_T2DM110_align_1+576.csv'
% xlRange_18='D1:N220';
% [data_origin,~,raw] = xlsread(filename_18,'T1DM110_T2DM110_align_1+576',xlRange_18);

%% select data,remain data expect LLS and last 3 items
data=data_origin(:,4:end)
labels = data_origin(:,1)
% data=data_origin(:,4:end-3);
% data=data_origin(:,3:end-3);
% X=data(:,[5,8,11]);
% data = data_origin;
%%
% 标准化数据
mu1 = mean(data);
sig = std(data);

dataStandardized = (data - mu1) ./ sig;

[coeff,score,latent,tsquared,explained,mu] = pca(dataStandardized);
%%
% X=score(:,1:2);
X=score(:,1:7);

%% 画图
yyaxis left
bar(latent)
xlabel('The principle component')
ylabel('The principal component variances')
yyaxis right
plot(1:10,explained,1:10,explained,'-*')
ylabel('The percentage of the total variance explained by each PC(%)')
legend("PCs variances","PCs variance fraction")
hold on
stem(5.5,30,'HandleVisibility','off','LineWidth',2)
annotation('textarrow',[0.65 0.55],[0.6 0.5],'String','CPV=90.18%','LineWidth',2,'color','red')

%% K-均值聚类
% 使用 kmeans 将训练数据分成三个簇。
for i = 1:10
    
    %
        rng(i);
        [idx,C] = kmeans(X,2);
        idx';
        result1 = calMI(idx,labels)
        sum(idx)
    % end
%     rng(i);
%     [idx,C] = kmeans(X,2);
%     sum(idx)
%     idx'
%     result1 = get_mutual_infomation(idx,18);
%     result1(1)
%     % 绘制簇和簇质心。
	figure
    gscatter(X(:,1),X(:,2),idx,'bgmycrk')
    hold on
    plot(C(:,1),C(:,2),'kx')
    legend('Cluster 1','Cluster 2','Cluster Centroid')


%     figure
%     gscatter(X(:,1),X(:,2),idx,'bgmycrk')
%     hold on
%     plot(C(:,1),C(:,2),'kx')
%     legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5','Cluster 6','Cluster 7','Cluster Centroid')

    
%     % legend('Cluster 1','Cluster 2','Cluster 3','Cluster Centroid')
%  
% 绘制簇和簇质心。
%     figure
%     scatter3(X(idx==1,1),X(idx==1,2),X(idx==1,3),'b','filled')
%     hold on;
%     scatter3(X(idx==2,1),X(idx==2,2),X(idx==2,3),'r','filled')
%     plot3(C(:,1),C(:,2),C(:,3),'kx')
%     legend('Cluster 1','Cluster 2','Cluster Centroid')

end

%%
eva = evalclusters(X,'kmeans','gap','KList',[1:10])
% eva = evalclusters(X,'kmeans','CalinskiHarabasz','KList',[1:4])
figure;
plot(eva);
hold on 
bar(eva.CriterionValues)
%%
%     rng(i);
    [idx,C] = kmeans(X,2);
    idx'
    result1 = get_mutual_infomation(idx,18);
    result1(1)
    % 绘制簇和簇质心。
    figure
    gscatter(X(:,1),X(:,2),idx,'bgm')
    hold on
    plot(C(:,1),C(:,2),'kx')
    legend('Cluster 1','Cluster 2','Cluster Centroid')
    % legend('Cluster 1','Cluster 2','Cluster 3','Cluster Centroid')
%  
%     eva = evalclusters(X,'kmeans','gap','KList',[1:4])
%     figure;
%     plot(eva);

%%
% 绘制簇和簇质心。
    figure
    scatter3(X(idx==1,1),X(idx==1,2),X(idx==1,3),'b','filled')
    hold on;
    scatter3(X(idx==2,1),X(idx==2,2),X(idx==2,3),'r','filled')
    plot3(C(:,1),C(:,2),C(:,3),'kx')
    legend('Cluster 1','Cluster 2','Cluster Centroid')

%%
LLS=data(:,3)
lei1 = [0 0 0 0 1 0 0 0 1 0 0 1 1 1 1 1 1 1]'
result1 = get_mutual_infomation(lei1,18)
res1 = result1(1)

lei2 = [0 0 0 0 1 0 0 1 1 0 0 1 1 1 1 1 1 1]'
result2 = get_mutual_infomation(lei2,18)
res2 = result2(1)

lei3 = [0 0 0 0 0 0 0 1 0 0 0 1 1 1 1 1 1 1]'
result3 = get_mutual_infomation(lei3,18)
res3 = result3(1)

lei4 = [1 1 1 1 2 1 1 3 2 1 1 3 3 2 3 3 3 3]'
result4 = get_mutual_infomation(lei4,18)
res4 = result4(1)
