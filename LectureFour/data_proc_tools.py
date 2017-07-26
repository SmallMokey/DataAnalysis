# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/05/01
    项目名称：Lending Club借贷数据处理及初步分析
"""

from __future__ import division, print_function


def insepct_data(df_data):
    """
        审查数据集

        参数
        ======
        df_data:    dataframe类型的数据

        返回值
        ======
        None
    """
    # 查看数据集信息
    print('\n数据预览：')
    print(df_data.head())

    print('\n数据统计信息：')
    print(df_data.describe())

    print('\n数据集基本信息：')
    print(df_data.info())
