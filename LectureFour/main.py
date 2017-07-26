from __future__ import division, print_function
# -*- coding: utf-8 -*-
import os
import pandas as pd
import zipfile
from sklearn import preprocessing
from ml_tools import perform_machine_learning
"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/05/01
    项目名称：Lending Club借贷数据处理及初步分析
             Lending Club借贷数据探索性分析及可视化
             Lending Club借贷违约预测
"""


import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')     # 设置图片显示的主题样式

# 解决matplotlib显示中文问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

dataset_path = 'D:/testData/dataset'
zip_file_name = 'loan.csv.zip'
csv_file_name = './loan.csv'


def analyze_lending_club_data(lc_data):
    """
        对lending club数据进行分析

        参数
        ======
        lc_data:    lending club数据集

        返回
        ======
        None
    """
    # 选择列
    used_cols = ['loan_amnt', 'term', 'int_rate', 'grade', 'issue_d', 'addr_state', 'loan_status']
    used_data = lc_data[used_cols]
    # 查看数据集信息
    print('\n分析数据预览：')
    print(used_data.head())

    # 1. 查看不同借贷状态的数据量
    print('\n各借贷状的态数据量')
    print(used_data['loan_status'].value_counts())

    # 2. 按月份统计借贷金额总量
    # 转换数据类型
    print('时间数据类型转换...')
    used_data['issue_d2'] = pd.to_datetime(used_data['issue_d'])
    print('\n分析数据预览：')
    print(used_data.head())
    print('\n分析数据集基本信息：')
    print(used_data.info())

    data_group_by_date = used_data.groupby(['issue_d2']).sum()
    data_group_by_date.reset_index(inplace=True)
    data_group_by_date['issue_month'] = data_group_by_date['issue_d2'].apply(lambda x: x.to_period('M'))

    load_amount_group_by_month = data_group_by_date.groupby('issue_month')['loan_amnt'].sum()
    # 结果转换为DataFrame
    load_amount_group_by_month_df = pd.DataFrame(load_amount_group_by_month).reset_index()

    # 可视化
    load_amount_group_by_month_df.plot()
    plt.xlabel('日期')
    plt.ylabel('借贷总量')
    plt.title('日期 vs 借贷总量')
    plt.tight_layout()
    plt.savefig('./pics/loan_amount_vs_month.png')
    plt.show()

    print('\n按月统计借贷总额预览：')
    print(load_amount_group_by_month_df.head())
    # 保存结果
    load_amount_group_by_month_df.to_csv('/output/load_amount_by_month.csv', index=False)

    # 3. 按地区（州）统计借贷金额总量
    data_group_by_state = used_data.groupby(['addr_state'])['loan_amnt'].sum()

    # 可视化
    data_group_by_state.plot(kind='bar')
    plt.xlabel('州')
    plt.ylabel('借贷总量')
    plt.title('州 vs 借贷总量')
    plt.tight_layout()
    plt.savefig('/pics/loan_amount_vs_state.png')
    plt.show()

    # 结果转换为DataFrame
    data_group_by_state_df = pd.DataFrame(data_group_by_state).reset_index()
    print('\n按州统计借贷总额预览：')
    print(data_group_by_state_df.head())
    # 保存结果
    data_group_by_state_df.to_csv('/output/load_amount_by_state.csv', index=False)

    # 4. 借贷评级、期限和利率的关系
    data_group_by_grade_term = used_data.groupby(['grade', 'term'])['int_rate'].mean()
    data_group_by_grade_term_df = pd.DataFrame(data_group_by_grade_term).reset_index()

    print('\n借贷评级、期限和利率关系预览：')
    print(data_group_by_grade_term_df.head())
    # 保存结果
    data_group_by_grade_term_df.to_csv('D:/testResult/output/intrate_by_grade_term.csv', index=False)

    # 转换为透视表
    data_group_by_grade_term_pivot = data_group_by_grade_term_df.pivot(index='grade', columns='term', values='int_rate')
    # 保存结果
    data_group_by_grade_term_pivot.to_csv('D:/testResult/output/intrate_by_grade_term2.csv')


def create_label(status_val):
    """
        根据status创建0, 1标签

        参数
        ======
        status_val: loan_status值

        返回
        =======
        label:  如果loan_status是'Fully Paid'，返回0，否则返回1
    """
    label = 1
    if status_val == 'Fully Paid':
        label = 0
    return label


def proc_emp_length(emp_length_val):
    """
        根据emp_length的值返回相应的特征，对应规则如下:
            '< 1 year'  -> 0.5
            'n/a'       -> 0.5
            '10+ years' -> 10
            其他         -> 对应年份值 (比如 '2 years' -> 2)

        参数
        ======
        emp_length_val: emp_length值

        返回
        =======
        emp_length_feat:  转换后的emp_length特征值
    """
    # 补全该函数
    if emp_length_val == '< 1 year' or emp_length_val == 'n/a':
        emp_length_feat = 0.5
    elif emp_length_val == '10+ years':
        emp_length_feat = 10
    else:
        emp_length_feat = float(emp_length_val.rstrip(' years'))
    return emp_length_feat


def run_main():
    """
        主函数
    """
    zip_file_path = os.path.join(dataset_path, zip_file_name)
    csv_file_path = os.path.join(dataset_path, csv_file_name)

    if not os.path.exists(csv_file_path):
        # 如果不存在csv文件，解压zip文件
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall(dataset_path)

    # == 1. 读取数据集 ==
    raw_data = pd.read_csv(csv_file_path)
    # 审查数据集
    # insepct_data(raw_data)

    # 对lending club数据进行分析
    # analyze_lending_club_data(raw_data)

    # == 2. 数据处理 ==
    # 数据处理及转换，用于后续模型的输入
    # 2.1 “借贷状态” (loan_status) 数据处理
    # 根据借贷状态筛选数据，只保留借贷状态为'Fully Paid', 'Charged Off'和'Default'的数据
    # 'Charged Off'和'Default'的解释请参考：https://help.lendingclub.com/hc/en-us/articles/216127747
    filter_mask = raw_data['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default'])#判断loan_status列是否存在列表中的值
    filter_data = raw_data[filter_mask]
    print(filter_data['loan_status'].value_counts())
    # 为数据添加 0, 1 标签，'Fully Paid' -> 0, Otherwise -> 1
    proc_filter_data = filter_data.copy()
    proc_filter_data['label'] = filter_data['loan_status'].apply(create_label)#apply有循环效果

    # 2.2 “工作年份” (emp_length) 数据处理
    # 使用apply函数处理emp_length特征
    proc_filter_data['emp_length_feat'] = filter_data['emp_length'].apply(proc_emp_length)

    # 2.3 “开始借贷每月付款金额” (installment) 数据处理
    proc_filter_data['installment_feat'] = proc_filter_data['installment'] / (proc_filter_data['annual_inc'] / 12)

    # 2.4 “借贷评级” (grade) 数据处理
    label_enc = preprocessing.LabelEncoder()#标签编码
    # proc_filter_data['grade_feat'] = label_enc.fit_transform(proc_filter_data['grade'].values)
    proc_filter_data['grade_feat'] = label_enc.fit_transform(proc_filter_data['grade'].values)

    # 2.5 “借贷期限” (term) 数据处理
    proc_filter_data['term_feat'] = proc_filter_data['term'].apply(lambda x: int(x[1:3]))

    # 选择使用的列
    numeric_cols = ['int_rate', 'grade_feat', 'loan_amnt', 'installment', 'annual_inc', 'dti',
                    'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
                    'total_acc', 'collections_12_mths_ex_med', 'acc_now_delinq', 'term_feat',
                    'installment_feat', 'emp_length_feat']

    # 课后作业：
    # 尝试加入"purpose"特征，可参照“home_ownership”数据处理的方式
    category_cols = ['home_ownership']

    label_col = ['label']

    user_cols = numeric_cols + category_cols + label_col

    final_samples = proc_filter_data[user_cols]

    # 去掉空值
    final_samples.dropna(inplace=True)

    # 保存处理后的数据集
    proc_data_filepath = 'D:/testResult/output/proc_data.csv'
    final_samples.to_csv(os.path.join(proc_data_filepath), index=False)

    if os.path.exists(csv_file_path):
        # 如果存在csv文件，删除csv文件，释放空间
        os.remove(csv_file_path)

    # 数据集处理及模型学习
    perform_machine_learning(proc_data_filepath, numeric_cols, category_cols, label_col)

if __name__ == '__main__':
    run_main()
