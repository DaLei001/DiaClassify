%������������֮��Ļ���Ϣ
%u1��������������1
%u2��������������2
%wind_size�������ĳ���
function result = get_mutual_infomation(t,nums_of_subjects)

sheetname='Sheet1';
if nums_of_subjects==18
    filename_18='.\18��dataB.xlsx';
    xlRange_18='A1:P19';
    [data,~,raw] = xlsread(filename_18,sheetname,xlRange_18);
elseif nums_of_subjects==12
    filename_12='C:\Dalei_lab\����\����ѵ����˯�������������������ݻ���12��.xlsx';
    xlRange_12='A1:P13';
    [data,~,raw] = xlsread(filename_12,sheetname,xlRange_12);
end

% id = data(:,1); % ���
% name = raw(2:end,2); % ����
% lake_louise_score = data(:,3); % ����˯�ߺ�·��˹������
% sleep_time = data(:,4); % ˯��ʱ��1
% deep_sleep_time = data(:,5); % ��˯ʱ��1
% deep_sleep_time_ratio = data(:,6); % ��˯����1
% wake_time = data(:,7); % ����ʱ��1
% nums_of_wake = data(:,8); % ���Ѵ���1
% average_HR = data(:,9); % ƽ������1
% average_SpO2 = data(:,10); % ƽ��Ѫ��1
% wake_up_time_SpO2 = data(:,11); % ����Ѫ��1
% body_motion_time = data(:,12); % �嶯ʱ��ָ��1
% nums_of_body_motion = data(:,13); % �嶯����ָ��1
% hypoxia_test1 = data(:,14); % ��������1����Ъ��ȱ��ѵ��ǰ�Ĳ��ԣ�
% hypoxia_test2 = data(:,15); % ��������2����Ъ��ȱ��ѵ����Ĳ��ԣ�
% changes_in_hypoxia_test = data(:,16); % �������Ա仯

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
    [occur,bin(:,i)] = histc(x(:,i),histcEdges,1); %ͨ��ֱ��ͼ��ʽ���㵥��������ֱ��ͼ�ֲ�
    pmf(:,i) = occur(1:n)./xrow;
end
%����u1��u2�����ϸ����ܶ�
jointOccur = accumarray(bin,1,[n,n]);  %��xi��yi����������ͬʱ����n*n�ȷַ����е�������Ϊ���ϸ����ܶ�
jointPmf = jointOccur./xrow;
Hx = -(pmf(:,1))'*log2(pmf(:,1)+eps);
Hy = -(pmf(:,2))'*log2(pmf(:,2)+eps);
Hxy = -(jointPmf(:))'*log2(jointPmf(:)+eps);
MI = Hx+Hy-Hxy;
mi = MI/sqrt(Hx*Hy);
end