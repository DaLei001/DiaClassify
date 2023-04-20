%计算两列向量之间的互信息
%u1：输入计算的向量1
%u2：输入计算的向量2
%wind_size：向量的长度
function result = get_mutual_infomation(t,nums_of_subjects)

sheetname='Sheet1';
if nums_of_subjects==18
    filename_18='.\18例dataB.xlsx';
    xlRange_18='A1:P19';
    [data,~,raw] = xlsread(filename_18,sheetname,xlRange_18);
elseif nums_of_subjects==12
    filename_12='C:\Dalei_lab\数据\低氧训练后睡眠数据与耐氧能力数据汇总12人.xlsx';
    xlRange_12='A1:P13';
    [data,~,raw] = xlsread(filename_12,sheetname,xlRange_12);
end

% id = data(:,1); % 序号
% name = raw(2:end,2); % 姓名
% lake_louise_score = data(:,3); % 低氧睡眠后路易斯湖评分
% sleep_time = data(:,4); % 睡眠时间1
% deep_sleep_time = data(:,5); % 深睡时间1
% deep_sleep_time_ratio = data(:,6); % 深睡比例1
% wake_time = data(:,7); % 觉醒时间1
% nums_of_wake = data(:,8); % 觉醒次数1
% average_HR = data(:,9); % 平均心率1
% average_SpO2 = data(:,10); % 平均血氧1
% wake_up_time_SpO2 = data(:,11); % 晨醒血氧1
% body_motion_time = data(:,12); % 体动时间指数1
% nums_of_body_motion = data(:,13); % 体动次数指数1
% hypoxia_test1 = data(:,14); % 低氧测试1（间歇性缺氧训练前的测试）
% hypoxia_test2 = data(:,15); % 低氧测试2（间歇性缺氧训练后的测试）
% changes_in_hypoxia_test = data(:,16); % 低氧测试变化

result=zeros(1,14);
for a=3:16
    result(1,a-2)=calmi(t,data(:,a));
end
end

function mi = calmi(u1,u2)
x = [u1, u2];
wind_size=length(u1);
n = wind_size;
[xrow, xcol] = size(x);
bin = zeros(xrow,xcol);
pmf = zeros(n, 2);
for i = 1:2
    minx = min(x(:,i));
    maxx = max(x(:,i));
    binwidth = (maxx - minx) / n;
    edges = minx + binwidth*(0:n);
    histcEdges = [-Inf edges(2:end-1) Inf];
    [occur,bin(:,i)] = histc(x(:,i),histcEdges,1); %通过直方图方式计算单个向量的直方图分布
    pmf(:,i) = occur(1:n)./xrow;
end
%计算u1和u2的联合概率密度
jointOccur = accumarray(bin,1,[n,n]);  %（xi，yi）两个数据同时落入n*n等分方格中的数量即为联合概率密度
jointPmf = jointOccur./xrow;
Hx = -(pmf(:,1))'*log2(pmf(:,1)+eps);
Hy = -(pmf(:,2))'*log2(pmf(:,2)+eps);
Hxy = -(jointPmf(:))'*log2(jointPmf(:)+eps);
MI = Hx+Hy-Hxy;
mi = MI/sqrt(Hx*Hy);
end