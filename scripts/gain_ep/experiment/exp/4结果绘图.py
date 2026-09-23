# pyright: basic
"""
# 实验结果绘图脚本

该脚本用于绘制和分析实验结果。
"""

from pysci.research.gain_ep.experiment.analyze import (
    Point2D,
    plot_comprehensive_experiment,
    prepare_comprehensive_experiment_data,
)

# 总结果文件夹根路径
# root_folder: str = "D:\\EveryoneDownloaded\\exp0723\\"
root_folder: str = "D:\\科研实践\\汇报ppt\\20260725\\新有源扫场结果\\"

# %% 数据准备
_ = prepare_comprehensive_experiment_data(
    # left_r_0_passive_folder = root_folder + "\\12_M-_R_1st_sweep",
    left_r_0_active_folder=root_folder + "\\7_M-_R_sweep",
    # left_r_minus1_passive_folder = root_folder + "\\5_M+_L_1st_sweep",
    left_r_minus1_active_folder=root_folder + "\\5_M+_L_sweep",
    # right_r_plus1_passive_folder = root_folder + "\\14_M-_l_1st_sweep",
    right_r_plus1_active_folder=root_folder + "\\8_M-_l_sweep",
    # right_r_0_passive_folder = root_folder + "\\10_M+_r_1st_sweep",
    right_r_0_active_folder=root_folder + "\\6_M+_r_sweep",
    left_background_folder=root_folder + "\\9_X_L_bg_sweep",
    right_background_folder=root_folder + "\\10_X_R_bg_sweep",
    save_path=root_folder + "\\plot_data.pkl",
)

# %% 绘图
# mode = "abs"
mode = "fourier"
_ = plot_comprehensive_experiment(
    data_pkl_path=root_folder + "\\plot_data.pkl",
    # --- 区域选取参数 ---
    picked_center=Point2D(x=155.0, y=155.5),
    picked_area_radius=100,
    area_shape="square",
    # --- 积分参数 ---
    integral_mode=mode,
    save_path=root_folder + f"\\{mode}",
)
