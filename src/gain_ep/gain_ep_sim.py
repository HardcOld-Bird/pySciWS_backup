"""gainEP项目的COMSOL仿真接口

此模块提供用于gainEP项目的COMSOL仿真自动化接口，包括数据类定义、
仿真运行和反馈环路模拟。
"""

import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mph
import numpy as np

# 导入并应用项目的中文字体配置
from config.matplotlib_config import setup_chinese_fonts

setup_chinese_fonts()


@dataclass
class SimulationInput:
    """仿真输入参数数据类

    封装10个物理输入参数：2个背景压力场幅值和8个法向位移。

    Attributes:
        pamp_L: 背景压力场 L 的压力幅值（实数）
        pamp_R: 背景压力场 R 的压力幅值（实数）
        vn_1: 法向位移 1 的向内位移（复数）
        vn_2: 法向位移 2 的向内位移（复数）
        vn_3: 法向位移 3 的向内位移（复数）
        vn_4: 法向位移 4 的向内位移（复数）
        vn_5: 法向位移 5 的向内位移（复数）
        vn_6: 法向位移 6 的向内位移（复数）
        vn_7: 法向位移 7 的向内位移（复数）
        vn_8: 法向位移 8 的向内位移（复数）
    """

    pamp_L: float
    pamp_R: float
    vn_1: complex
    vn_2: complex
    vn_3: complex
    vn_4: complex
    vn_5: complex
    vn_6: complex
    vn_7: complex
    vn_8: complex


@dataclass
class SimulationOutput:
    """仿真输出参数数据类

    封装16个域探针的复数输出结果。

    Attributes:
        dom1 到 dom16: 16个域探针的复数结果
    """

    dom1: complex
    dom2: complex
    dom3: complex
    dom4: complex
    dom5: complex
    dom6: complex
    dom7: complex
    dom8: complex
    dom9: complex
    dom10: complex
    dom11: complex
    dom12: complex
    dom13: complex
    dom14: complex
    dom15: complex
    dom16: complex

    def to_array(self) -> np.ndarray:
        """将输出数据转换为NumPy数组

        Returns:
            包含16个复数值的NumPy数组
        """
        return np.array(
            [
                self.dom1,
                self.dom2,
                self.dom3,
                self.dom4,
                self.dom5,
                self.dom6,
                self.dom7,
                self.dom8,
                self.dom9,
                self.dom10,
                self.dom11,
                self.dom12,
                self.dom13,
                self.dom14,
                self.dom15,
                self.dom16,
            ]
        )


class GainEPSimulator:
    """gainEP COMSOL 仿真器类

    此类封装了 gainEP 项目的 COMSOL 仿真功能，管理与 COMSOL Server 的连接，
    提供单次仿真和反馈环路仿真的接口。

    Attributes:
        mph_file: COMSOL 模型文件路径
        client: MPh Client 对象，用于与 COMSOL Server 通信

    Examples:
        使用上下文管理器（推荐）:
        >>> from pathlib import Path
        >>> mph_file = Path("mphs/gainEP/gainEP_10in16out.mph")
        >>> with GainEPSimulator(mph_file) as simulator:
        ...     sim_input = SimulationInput(1.0, 1.0, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j)
        ...     output = simulator.run_simulation(sim_input)
        ...     # 连接会自动断开

        手动管理连接:
        >>> simulator = GainEPSimulator(mph_file)
        >>> simulator.connect()
        >>> output = simulator.run_simulation(sim_input)
        >>> simulator.disconnect()
    """

    def __init__(self, mph_file: Path, feedback_constant: complex | float = 0.0):
        """初始化仿真器

        Args:
            mph_file: COMSOL 模型文件路径 (.mph文件)
            feedback_constant: 反馈常数（复数或实数），用于 constant 模式，默认为 0.0

        Raises:
            FileNotFoundError: 如果 mph_file 不存在
        """
        if not mph_file.exists():
            raise FileNotFoundError(f"COMSOL模型文件不存在: {mph_file}")

        self.mph_file = mph_file
        self.client: mph.Client | None = None
        self.feedback_constant = complex(feedback_constant)  # 统一转换为复数

        # 传递函数矩阵：shape (10, 8)
        # 行索引0-1: 左右入射
        # 行索引2-9: 法向位移1-8
        # 列索引0-7: 探针9-16的输出
        self.transfer_functions: np.ndarray | None = None

        # 增益系数：shape (8,)，从only文件测量，用作all8模式的反馈系数
        self.gain_coefficients: np.ndarray | None = None

        # 设置存储路径
        # 获取项目根目录（mph_file 在 mphs/gainEP/ 下）
        self.workspace_root = mph_file.parent.parent.parent
        self.calib_dir = self.workspace_root / "storage" / "gainEP" / "calib"
        self.plots_dir = self.workspace_root / "storage" / "gainEP" / "plots"

        # 确保目录存在
        self.calib_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def connect(self) -> None:
        """连接到 COMSOL Server

        如果已经连接，则不执行任何操作。

        Raises:
            RuntimeError: 如果无法连接到 COMSOL Server
        """
        if self.client is not None:
            return  # 已经连接

        try:
            self.client = mph.start()
            print("✓ 已连接到 COMSOL Server")
        except Exception as e:
            raise RuntimeError(f"无法连接到COMSOL Server: {e}")

    def disconnect(self) -> None:
        """断开与 COMSOL Server 的连接

        如果未连接，则不执行任何操作。
        """
        if self.client is not None:
            try:
                self.client.disconnect()
                self.client = None
                print("✓ 已断开 COMSOL 连接")
            except Exception as e:
                print(f"⚠ 断开连接时出现警告: {e}")
                self.client = None

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()
        return False  # 不抑制异常

    def set_feedback_constant(self, constant: complex | float) -> None:
        """设置反馈常数

        Args:
            constant: 新的反馈常数值（复数或实数）
        """
        self.feedback_constant = complex(constant)
        print(f"✓ 反馈常数已设置为: {self.feedback_constant}")

    def calib(self, force_recompute: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """计算传递函数矩阵和增益系数

        此方法计算两组数据：

        1. 传递函数矩阵（transfer_functions）：从 gainEP_10in16out.mph 测量
           - 运行10次仿真，每次仅将一个输入设为1，其余为0
           - 10个输入：左入射、右入射、法向位移1-8
           - 每次仿真记录探针9-16的所有8个输出值
           - 结果为 10×8 矩阵：
             * 行索引0-1: 左右入射对探针9-16的影响
             * 行索引2-9: 法向位移1-8对探针9-16的影响
             * 列索引0-7: 探针9-16的输出

        2. 增益系数（gain_coefficients）：从 gainEP_10in16out_only{1~8}.mph 测量
           - 左入射=1，右入射=0，所有法向位移=0
           - 计算探针i / 探针(i+8)，用作 all8 模式的反馈系数
           - 结果为长度8的数组

        计算完成后，两组数据会自动保存到 storage/gainEP/calib/ 目录。

        Args:
            force_recompute: 是否强制重新计算（忽略缓存），默认为 False

        Returns:
            tuple: (传递函数矩阵, 增益系数数组)
                - 传递函数矩阵: shape (10, 8) 的复数数组
                - 增益系数数组: shape (8,) 的复数数组

        Raises:
            RuntimeError: 如果未连接到 COMSOL Server
            FileNotFoundError: 如果找不到对应的仿真文件

        Examples:
            >>> with GainEPSimulator(mph_file) as simulator:
            ...     # 首次调用会计算并缓存
            ...     transfer_funcs, gain_coeffs = simulator.calib()
            ...     # transfer_funcs.shape == (10, 8)
            ...     # gain_coeffs.shape == (8,)
            ...     # 后续调用会从缓存读取
            ...     transfer_funcs, gain_coeffs = simulator.calib()
            ...     # 强制重新计算
            ...     transfer_funcs, gain_coeffs = simulator.calib(force_recompute=True)
        """
        # 确保已连接
        if self.client is None:
            raise RuntimeError("未连接到 COMSOL Server，请先调用 connect() 方法")

        # 生成缓存文件名
        cache_filename = f"{self.mph_file.stem}_calibration.pkl"
        cache_path = self.calib_dir / cache_filename

        # 尝试从缓存读取
        if not force_recompute and cache_path.exists():
            print("\n从缓存读取传递函数和增益系数...")
            print(f"缓存文件: {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                    self.transfer_functions = cached_data["transfer_functions"]
                    self.gain_coefficients = cached_data["gain_coefficients"]
                    cache_time = cached_data.get("timestamp", "未知")
                    print("✓ 成功从缓存读取")
                    print(f"  缓存时间: {cache_time}")
                    print(f"  传递函数矩阵形状: {self.transfer_functions.shape}")
                    print(f"  增益系数数量: {len(self.gain_coefficients)}")
                    return self.transfer_functions, self.gain_coefficients
            except Exception as e:
                print(f"⚠ 读取缓存失败: {e}")
                print("将重新计算...")

        # 缓存不存在或读取失败，重新计算
        print("\n开始计算传递函数和增益系数...")

        # ========================================================================
        # 第一部分：计算传递函数（使用当前mph_file，测量所有输入对所有输出的影响）
        # ========================================================================
        print("\n" + "=" * 70)
        print("第一部分：计算传递函数矩阵（10个输入 × 8个输出探针）")
        print("=" * 70)
        print(f"使用仿真文件: {self.mph_file.name}")
        print("将运行10次仿真：2个入射 + 8个法向位移")
        print("每次仿真记录探针9-16的所有8个输出值")

        # 初始化传递函数矩阵：10个输入 × 8个输出探针
        transfer_functions_matrix = []

        # 定义10个输入的名称（用于日志输出）
        input_names = ["左入射", "右入射"] + [f"法向位移{i}" for i in range(1, 9)]

        # 运行10次仿真
        for input_idx in range(10):
            print(f"\n--- 测量输入 {input_idx + 1}/10: {input_names[input_idx]} ---")

            # 创建输入：仅第input_idx个输入为1，其余为0
            if input_idx == 0:
                # 左入射为1
                sim_input = SimulationInput(
                    pamp_L=1.0,
                    pamp_R=0.0,
                    vn_1=0j,
                    vn_2=0j,
                    vn_3=0j,
                    vn_4=0j,
                    vn_5=0j,
                    vn_6=0j,
                    vn_7=0j,
                    vn_8=0j,
                )
            elif input_idx == 1:
                # 右入射为1
                sim_input = SimulationInput(
                    pamp_L=0.0,
                    pamp_R=1.0,
                    vn_1=0j,
                    vn_2=0j,
                    vn_3=0j,
                    vn_4=0j,
                    vn_5=0j,
                    vn_6=0j,
                    vn_7=0j,
                    vn_8=0j,
                )
            else:
                # 法向位移为1（input_idx 2-9 对应 vn_1-vn_8）
                vn_values = [0 + 0j] * 8
                vn_values[input_idx - 2] = 1 + 0j
                sim_input = SimulationInput(
                    pamp_L=0.0,
                    pamp_R=0.0,
                    vn_1=vn_values[0],
                    vn_2=vn_values[1],
                    vn_3=vn_values[2],
                    vn_4=vn_values[3],
                    vn_5=vn_values[4],
                    vn_6=vn_values[5],
                    vn_7=vn_values[6],
                    vn_8=vn_values[7],
                )

            # 运行仿真
            output = self.run_simulation(sim_input)

            # 提取探针9-16的所有8个输出值
            output_array = output.to_array()
            output_probes = output_array[8:16]  # 探针9-16（索引8-15）

            transfer_functions_matrix.append(output_probes)

            # 打印输出摘要
            print("✓ 探针9-16输出: ", end="")
            for j, probe_val in enumerate(output_probes, start=9):
                if j == 9:
                    print(f"{probe_val:.4e}", end="")
                else:
                    print(f", {probe_val:.4e}", end="")
            print()

        # 转换为NumPy数组：shape (10, 8)
        self.transfer_functions = np.array(transfer_functions_matrix)
        print("\n✓ 传递函数矩阵计算完成！")
        print(f"  矩阵形状: {self.transfer_functions.shape} (10个输入 × 8个输出探针)")

        # ========================================================================
        # 第二部分：计算增益系数（使用only文件，左入射激励）
        # ========================================================================
        print("\n" + "=" * 70)
        print("第二部分：计算增益系数（从only文件测量）")
        print("=" * 70)
        print("将使用8个独立的仿真文件（gainEP_10in16out_only1~8.mph）")

        gain_coefficients = []

        # 获取mphs目录路径
        mphs_dir = self.workspace_root / "mphs" / "gainEP"

        for i in range(8):
            print(f"\n--- 计算管槽 {i + 1} 的增益系数 ---")

            # 构建对应的仿真文件路径
            mph_file_path = mphs_dir / f"gainEP_10in16out_only{i + 1}.mph"

            # 检查文件是否存在
            if not mph_file_path.exists():
                raise FileNotFoundError(f"仿真文件不存在: {mph_file_path}")

            print(f"使用仿真文件: {mph_file_path.name}")

            # 创建输入：左入射=1，右入射=0，所有法向位移=0
            sim_input = SimulationInput(
                pamp_L=1.0,  # 左入射为1
                pamp_R=0.0,  # 右入射为0
                vn_1=0 + 0j,
                vn_2=0 + 0j,
                vn_3=0 + 0j,
                vn_4=0 + 0j,
                vn_5=0 + 0j,
                vn_6=0 + 0j,
                vn_7=0 + 0j,
                vn_8=0 + 0j,
            )

            # 临时保存当前mph_file，切换到only文件
            original_mph_file = self.mph_file
            self.mph_file = mph_file_path

            try:
                # 运行仿真
                output = self.run_simulation(sim_input)

                # 提取探针值
                output_array = output.to_array()
                probe_i = output_array[i]  # 探针1-8（索引0-7）
                probe_i_plus_8 = output_array[i + 8]  # 探针9-16（索引8-15）

                # 计算增益系数：探针i / 探针(i+8)
                gain_coefficient = probe_i / probe_i_plus_8

                gain_coefficients.append(gain_coefficient)
                print(f"✓ 探针{i + 1}: {probe_i:.6e}")
                print(f"✓ 探针{i + 9}: {probe_i_plus_8:.6e}")
                print(f"✓ 管槽 {i + 1} 增益系数: {gain_coefficient:.6e}")

            finally:
                # 恢复原始mph_file
                self.mph_file = original_mph_file

        self.gain_coefficients = np.array(gain_coefficients)
        print("\n✓ 增益系数计算完成！")

        # ========================================================================
        # 保存到缓存文件
        # ========================================================================
        print("\n" + "=" * 70)
        print("保存到缓存")
        print("=" * 70)

        try:
            cache_data = {
                "transfer_functions": self.transfer_functions,
                "gain_coefficients": self.gain_coefficients,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mph_file": str(self.mph_file),
                "only_mph_files": [
                    f"gainEP_10in16out_only{i + 1}.mph" for i in range(8)
                ],
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            print("✓ 数据已保存到缓存文件")
            print(f"  缓存路径: {cache_path}")
        except Exception as e:
            print(f"⚠ 保存缓存失败: {e}")

        print("\n✓ 传递函数和增益系数已存储到实例变量")
        print(
            f"  self.transfer_functions: 矩阵形状 {self.transfer_functions.shape} (10个输入 × 8个输出探针)"
        )
        print(f"  self.gain_coefficients: {len(self.gain_coefficients)} 个复数")

        return self.transfer_functions, self.gain_coefficients

    def run_simulation(self, sim_input: SimulationInput) -> SimulationOutput:
        """运行单次仿真并返回16个域探针结果

        此方法加载COMSOL模型，设置10个输入参数（2个背景压力场幅值 + 8个法向位移），
        运行仿真，提取16个域探针的复数结果。

        Args:
            sim_input: SimulationInput对象，包含10个输入参数

        Returns:
            SimulationOutput对象，包含16个域探针的复数结果

        Raises:
            RuntimeError: 如果未连接到 COMSOL Server 或仿真失败

        Examples:
            >>> with GainEPSimulator(mph_file) as simulator:
            ...     sim_input = SimulationInput(1.0, 1.0, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j)
            ...     output = simulator.run_simulation(sim_input)
        """
        # 确保已连接
        if self.client is None:
            raise RuntimeError("未连接到 COMSOL Server，请先调用 connect() 方法")

        try:
            # 加载模型
            model = self.client.load(str(self.mph_file))

            # 获取Java模型对象
            java_model = model.java

            # 设置背景压力场幅值
            # bpf1: 背景压力场 L
            # bpf2: 背景压力场 R
            java_model.physics("acpr").feature("bpf1").set("pamp", sim_input.pamp_L)
            java_model.physics("acpr").feature("bpf2").set("pamp", sim_input.pamp_R)

            # 设置法向位移（复数值）
            # ndisp1 到 ndisp8 对应 8 个法向位移边界条件
            vn_list = [
                sim_input.vn_1,
                sim_input.vn_2,
                sim_input.vn_3,
                sim_input.vn_4,
                sim_input.vn_5,
                sim_input.vn_6,
                sim_input.vn_7,
                sim_input.vn_8,
            ]
            for i, vn in enumerate(vn_list, start=1):
                # COMSOL中复数格式为 "real+imag*i"
                vn_str = f"{vn.real}+{vn.imag}*i"
                java_model.physics("acpr").feature(f"ndisp{i}").set("ndisp", vn_str)

            # 运行仿真
            model.solve()

            # 从探针结果表中提取数据
            # 所有探针的结果存储在同一个表中（tbl5）
            # 获取第一个探针的表名称
            probe = java_model.probe("point1")
            table_name = probe.getString("table")

            # 获取表数据
            result_table = java_model.result().table(table_name)
            table_data = result_table.getTableData(True)  # True表示包含标题

            # 解析复数字符串
            def parse_complex(s):
                """解析COMSOL的复数字符串格式: 'real+imagi'"""
                s = str(s).strip()
                s = s.replace("i", "j")
                return complex(s)

            # 提取16个域探针的数据
            # 表结构: [freq, point1, point2, ..., point16]
            # 注意：表数据可能没有标题行，直接就是数据
            if len(table_data) < 1:
                raise RuntimeError("探针表数据为空，无法提取结果")

            # 取第一行数据（单次仿真）
            row = table_data[0]

            # 检查列数（应该是17列：1个频率 + 16个探针）
            if len(row) < 17:
                raise RuntimeError(f"表数据列数不足，期望至少17列，实际{len(row)}列")

            # 提取16个探针的值（索引1到16，索引0是频率）
            probe_results = []
            for i in range(1, 17):
                probe_results.append(parse_complex(row[i]))

            if len(probe_results) != 16:
                raise RuntimeError(
                    f"提取的探针数据数量不正确，期望16个，实际{len(probe_results)}个"
                )

            # 创建SimulationOutput对象
            output = SimulationOutput(*probe_results)

            return output

        except Exception:
            # 如果发生错误，重新抛出异常
            raise


    def solve_ground_truth(
        self,
        sim_input: SimulationInput,
        verify: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """求解使实际波场与理想波场一致的法向位移和等效增益系数

        此方法通过求解矩阵方程，找到使实际系统（探针9-16）与理想系统（探针1-8）
        完全一致的法向位移值，并计算每个管槽的等效增益系数。

        工作流程：
        1. 运行初始仿真，获取16个探针值
        2. 使用探针1-8作为"目标场"（理想系统）
        3. 从目标场中扣除左右入射的贡献，得到"理想系统的反馈贡献"
        4. 求解矩阵方程：找到8个法向位移，使其产生的声压等于"理想系统的反馈贡献"
        5. （可选）验证：用求解的法向位移重新运行仿真，检查探针1-8与探针9-16是否相等
        6. 计算等效增益系数：对每个管槽，用 总声压 / 入射声压

        Args:
            sim_input: 初始仿真输入（包含左右入射和理想系统的法向位移）
            verify: 是否进行验证步骤（默认True）

        Returns:
            tuple: (求解的法向位移, 等效增益系数, 验证误差)
                - solved_vn: shape (8,) 的复数数组，求解得到的法向位移
                - effective_gain_coeffs: shape (8,) 的复数数组，等效增益系数
                - verification_error: 验证误差（探针差值的方差），越小越好

        Raises:
            RuntimeError: 如果未连接到 COMSOL Server 或传递函数未计算
            np.linalg.LinAlgError: 如果矩阵方程无解

        Examples:
            >>> with GainEPSimulator(mph_file) as simulator:
            ...     # 先计算传递函数
            ...     simulator.calib()
            ...     # 求解ground truth
            ...     sim_input = SimulationInput(1.0, 0.0, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j)
            ...     vn, gain_coeffs, error = simulator.solve_ground_truth(sim_input)
            ...     print(f"验证误差: {error:.6e}")
        """
        # 确保已连接
        if self.client is None:
            raise RuntimeError("未连接到 COMSOL Server，请先调用 connect() 方法")

        # 确保传递函数已计算
        if self.transfer_functions is None:
            raise RuntimeError("传递函数未计算，请先调用 calib() 方法")

        print("\n" + "=" * 70)
        print("求解 Ground Truth：使实际系统与理想系统一致")
        print("=" * 70)

        # ========================================================================
        # 第一步：运行初始仿真，获取16个探针值
        # ========================================================================
        print("\n第一步：运行初始仿真...")
        output = self.run_simulation(sim_input)
        output_array = output.to_array()

        # 提取探针1-8作为"目标场"（理想系统）
        target_field = output_array[0:8]  # 探针1-8（索引0-7）
        print(f"✓ 获取目标场（探针1-8）")

        # ========================================================================
        # 第二步：计算理想系统的反馈贡献
        # ========================================================================
        print("\n第二步：计算理想系统的反馈贡献...")

        # 从传递函数矩阵中提取左右入射对探针1-8的影响
        # 行索引0: 左入射，行索引1: 右入射
        # 列索引0-7: 探针9-16，但我们需要探针1-8的影响
        # 注意：传递函数矩阵记录的是对探针9-16的影响，
        # 但由于对称性，我们假设左右入射对探针1-8的影响与对探针9-16的影响相同
        left_incident_contribution = self.transfer_functions[0, :] * sim_input.pamp_L
        right_incident_contribution = self.transfer_functions[1, :] * sim_input.pamp_R

        # 从目标场中扣除左右入射的贡献
        ideal_feedback_contribution = (
            target_field
            - left_incident_contribution
            - right_incident_contribution
        )

        print(f"✓ 计算得到理想系统的反馈贡献")
        print(f"  目标场范数: {np.linalg.norm(target_field):.6e}")
        print(f"  左入射贡献范数: {np.linalg.norm(left_incident_contribution):.6e}")
        print(f"  右入射贡献范数: {np.linalg.norm(right_incident_contribution):.6e}")
        print(f"  理想反馈贡献范数: {np.linalg.norm(ideal_feedback_contribution):.6e}")

        # ========================================================================
        # 第三步：求解法向位移矩阵方程
        # ========================================================================
        print("\n第三步：求解法向位移矩阵方程...")

        # 构建传递函数矩阵：8个法向位移对8个探针的影响
        # 从完整的传递函数矩阵中提取行2-9（法向位移1-8）
        transfer_matrix = self.transfer_functions[2:10, :]  # shape: (8, 8)

        # 求解线性方程组：transfer_matrix @ vn = ideal_feedback_contribution
        # 即：找到vn，使得8个法向位移产生的声压等于理想反馈贡献
        try:
            solved_vn = np.linalg.solve(transfer_matrix, ideal_feedback_contribution)
            print(f"✓ 成功求解法向位移")
            print(f"  求解的法向位移范数: {np.linalg.norm(solved_vn):.6e}")
        except np.linalg.LinAlgError as e:
            print(f"✗ 矩阵方程求解失败: {e}")
            raise

        # ========================================================================
        # 第四步：验证求解结果（可选）
        # ========================================================================
        verification_error = 0.0
        if verify:
            print("\n第四步：验证求解结果...")

            # 创建新的输入：使用求解的法向位移
            verification_input = SimulationInput(
                pamp_L=sim_input.pamp_L,
                pamp_R=sim_input.pamp_R,
                vn_1=solved_vn[0],
                vn_2=solved_vn[1],
                vn_3=solved_vn[2],
                vn_4=solved_vn[3],
                vn_5=solved_vn[4],
                vn_6=solved_vn[5],
                vn_7=solved_vn[6],
                vn_8=solved_vn[7],
            )

            # 重新运行仿真
            verification_output = self.run_simulation(verification_input)
            verification_array = verification_output.to_array()

            # 计算探针1-8与探针9-16的差值
            probes_1_8 = verification_array[0:8]
            probes_9_16 = verification_array[8:16]
            differences = probes_1_8 - probes_9_16

            # 计算方差（差值的平方和）
            verification_error = np.sum(np.abs(differences) ** 2)

            print(f"✓ 验证完成")
            print(f"  探针1-8与探针9-16的差值方差: {verification_error:.6e}")
            if verification_error < 1e-10:
                print(f"  ✓ 验证成功！差值方差接近0")
            else:
                print(f"  ⚠ 警告：差值方差较大，可能存在数值误差")

        # ========================================================================
        # 第五步：计算等效增益系数
        # ========================================================================
        print("\n第五步：计算等效增益系数...")

        effective_gain_coeffs = np.zeros(8, dtype=complex)

        # 使用验证后的输出（如果进行了验证）或重新计算
        if verify:
            final_output_array = verification_array
        else:
            # 如果没有验证，需要重新运行一次仿真
            final_input = SimulationInput(
                pamp_L=sim_input.pamp_L,
                pamp_R=sim_input.pamp_R,
                vn_1=solved_vn[0],
                vn_2=solved_vn[1],
                vn_3=solved_vn[2],
                vn_4=solved_vn[3],
                vn_5=solved_vn[4],
                vn_6=solved_vn[5],
                vn_7=solved_vn[6],
                vn_8=solved_vn[7],
            )
            final_output = self.run_simulation(final_input)
            final_output_array = final_output.to_array()

        # 对每个管槽计算等效增益系数
        for idx in range(8):
            # 总声压：可以从探针(9+idx)获取（索引8+idx）
            total_pressure = final_output_array[8 + idx]

            # 从传递函数矩阵中提取法向位移(idx+1)对探针(9+idx)的传递函数
            # 行索引：2 + idx（法向位移idx+1）
            # 列索引：idx（探针9+idx）
            transfer_func = self.transfer_functions[2 + idx, idx]

            # 入射声压 = 总声压 - 法向位移的贡献
            incident_pressure = total_pressure - solved_vn[idx] * transfer_func

            # 等效增益系数 = 总声压 / 入射声压
            if np.abs(incident_pressure) > 1e-15:
                effective_gain_coeff = total_pressure / incident_pressure
            else:
                print(f"  ⚠ 警告：管槽{idx + 1}的入射声压接近0，无法计算增益系数")
                effective_gain_coeff = 0 + 0j

            effective_gain_coeffs[idx] = effective_gain_coeff

            print(
                f"  管槽{idx + 1}: 总声压={total_pressure:.6e}, "
                f"入射声压={incident_pressure:.6e}, "
                f"等效增益系数={effective_gain_coeff:.6f}"
            )

        print("\n✓ Ground Truth 求解完成！")
        print(f"  求解的法向位移: {solved_vn}")
        print(f"  等效增益系数: {effective_gain_coeffs}")
        print(f"  验证误差: {verification_error:.6e}")

        # ========================================================================
        # 保存等效增益系数到文件（每次运行都覆盖）
        # ========================================================================
        print("\n保存等效增益系数到文件...")

        truth_filename = f"{self.mph_file.stem}_ground_truth.pkl"
        truth_path = self.calib_dir / truth_filename

        try:
            truth_data = {
                "effective_gain_coefficients": effective_gain_coeffs,
                "solved_vn": solved_vn,
                "verification_error": verification_error,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mph_file": str(self.mph_file),
                "input_pamp_L": sim_input.pamp_L,
                "input_pamp_R": sim_input.pamp_R,
            }
            with open(truth_path, "wb") as f:
                pickle.dump(truth_data, f)
            print(f"✓ 等效增益系数已保存到文件")
            print(f"  文件路径: {truth_path}")
        except Exception as e:
            print(f"⚠ 保存文件失败: {e}")

        return solved_vn, effective_gain_coeffs, verification_error

    def _load_ground_truth_coefficients(self) -> np.ndarray:
        """从文件加载 Ground Truth 等效增益系数

        此方法从 solve_ground_truth 保存的文件中读取等效增益系数。

        Returns:
            shape (8,) 的复数数组，等效增益系数

        Raises:
            FileNotFoundError: 如果 Ground Truth 文件不存在
            RuntimeError: 如果读取文件失败

        Examples:
            >>> coeffs = simulator._load_ground_truth_coefficients()
        """
        truth_filename = f"{self.mph_file.stem}_ground_truth.pkl"
        truth_path = self.calib_dir / truth_filename

        if not truth_path.exists():
            raise FileNotFoundError(
                f"Ground Truth 文件不存在: {truth_path}\n"
                f"请先运行 solve_ground_truth() 方法生成该文件"
            )

        try:
            with open(truth_path, "rb") as f:
                truth_data = pickle.load(f)
                effective_gain_coeffs = truth_data["effective_gain_coefficients"]
                return effective_gain_coeffs
        except Exception as e:
            raise RuntimeError(f"读取 Ground Truth 文件失败: {e}")


    def _generate_feedback(
        self,
        current_input: SimulationInput,
        current_output: SimulationOutput,
        mode: str,
    ) -> SimulationInput:
        """生成反馈输入

        根据当前输入和输出，以及指定的工作模式，计算下一次迭代的输入参数。

        工作模式：
        - from_calib: 使用校准结果中的8个增益系数作为反馈系数
        - from_truth: 使用 solve_ground_truth 计算的8个等效增益系数作为反馈系数
        - constant: 所有8个管槽使用相同的 self.feedback_constant 作为反馈系数

        反馈公式（对所有模式都相同）：
        1. 计算入射声压 = 总声压 - 法向位移自身的贡献（使用传递函数）
        2. 计算目标总声压 = 入射声压 × 反馈系数
        3. 计算新的法向位移（使用传递函数）

        Args:
            current_input: 当前迭代的输入参数
            current_output: 当前迭代的输出结果
            mode: 工作模式，"from_calib"、"from_truth" 或 "constant"

        Returns:
            下一次迭代的输入参数

        Raises:
            ValueError: 如果 mode 不合法
            RuntimeError: 如果传递函数未计算或所需的系数文件不存在
        """
        # 验证传递函数已计算
        if self.transfer_functions is None:
            raise RuntimeError("传递函数未计算，请先调用 calib()")

        # 验证 mode 合法性
        valid_modes = ["from_calib", "from_truth", "constant"]
        if mode not in valid_modes:
            raise ValueError(
                f"mode 必须是 {valid_modes} 之一，当前值: {mode}"
            )

        # 根据 mode 获取反馈系数
        if mode == "from_calib":
            if self.gain_coefficients is None:
                raise RuntimeError("增益系数未计算，请先调用 calib()")
            feedback_coeffs = self.gain_coefficients
        elif mode == "from_truth":
            feedback_coeffs = self._load_ground_truth_coefficients()
        elif mode == "constant":
            feedback_coeffs = np.full(8, self.feedback_constant, dtype=complex)

        # 获取当前法向位移和输出声压
        current_vn = np.array(
            [
                current_input.vn_1,
                current_input.vn_2,
                current_input.vn_3,
                current_input.vn_4,
                current_input.vn_5,
                current_input.vn_6,
                current_input.vn_7,
                current_input.vn_8,
            ]
        )

        output_array = current_output.to_array()
        # 探针9-16（索引8-15）对应管槽1-8的声压
        total_pressures = output_array[8:16]

        # 初始化新的法向位移
        new_vn = np.zeros(8, dtype=complex)

        # 对所有8个管槽进行反馈
        for idx in range(8):
            total_pressure = total_pressures[idx]

            # 从传递函数矩阵中提取法向位移(idx+1)对探针(9+idx)的传递函数
            # 法向位移(idx+1)在矩阵中的行索引：2 + idx
            # 探针(9+idx)在矩阵中的列索引：idx
            transfer_func = self.transfer_functions[2 + idx, idx]
            feedback_coeff = feedback_coeffs[idx]  # 该管槽的反馈系数
            current_vn_value = current_vn[idx]

            # 计算入射声压（扣除法向位移自身的贡献）
            incident_pressure = total_pressure - current_vn_value * transfer_func

            # 计算目标总声压（使用该管槽的反馈系数）
            target_total_pressure = incident_pressure * feedback_coeff

            # 计算差值
            pressure_diff = target_total_pressure - total_pressure

            # 计算法向位移增量
            vn_increment = pressure_diff / transfer_func

            # 计算新的法向位移（增量调整）
            new_vn[idx] = current_vn_value + vn_increment

        # 创建新的输入（左右入射幅值保持不变）
        return SimulationInput(
            pamp_L=current_input.pamp_L,
            pamp_R=current_input.pamp_R,
            vn_1=new_vn[0],
            vn_2=new_vn[1],
            vn_3=new_vn[2],
            vn_4=new_vn[3],
            vn_5=new_vn[4],
            vn_6=new_vn[5],
            vn_7=new_vn[6],
            vn_8=new_vn[7],
        )

    def run_feedback_loop(
        self,
        initial_input: SimulationInput,
        num_iterations: int,
        mode: str = "from_calib",
    ) -> tuple[list[SimulationInput], list[SimulationOutput]]:
        """运行反馈环路仿真

        此方法模拟一个反馈环路的物理过程：多次运行仿真，将上一次的输出结果
        经过内部反馈函数运算后作为下一次的输入参数。

        工作流程：
        1. 使用initial_input运行第一次仿真，得到第一个输出
        2. 将当前输入和输出传入内部反馈函数，得到新的输入
        3. 使用新输入运行下一次仿真
        4. 重复步骤2-3，直到完成指定次数的迭代

        Args:
            initial_input: SimulationInput对象，第一轮仿真的初始条件
            num_iterations: 反馈迭代次数（总仿真次数）
            mode: 反馈模式，可选值：
                - "from_calib": 使用校准结果中的增益系数（默认）
                - "from_truth": 使用 solve_ground_truth 计算的等效增益系数
                - "constant": 所有管槽使用相同的 feedback_constant

        Returns:
            tuple: 包含2个元素的元组：
                - list[SimulationInput]: 所有迭代的输入参数列表
                - list[SimulationOutput]: 所有迭代的输出结果列表

        Raises:
            ValueError: 如果num_iterations < 1 或 mode 不合法
            RuntimeError: 如果未连接到 COMSOL Server 或传递函数未计算

        Examples:
            >>> with GainEPSimulator(mph_file, feedback_constant=1.0) as simulator:
            ...     # 先计算传递函数
            ...     simulator.calib()
            ...     # 运行反馈环路（使用校准系数）
            ...     initial = SimulationInput(1.0, 0.0, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j)
            ...     inputs, outputs = simulator.run_feedback_loop(initial, 8, mode="from_calib")
            ...     # 或使用常数反馈
            ...     inputs, outputs = simulator.run_feedback_loop(initial, 8, mode="constant")
        """
        # 验证参数
        if num_iterations < 1:
            raise ValueError(f"num_iterations必须 >= 1，当前值: {num_iterations}")

        valid_modes = ["from_calib", "from_truth", "constant"]
        if mode not in valid_modes:
            raise ValueError(f"mode 必须是 {valid_modes} 之一，当前值: {mode}")

        # 确保已连接
        if self.client is None:
            raise RuntimeError("未连接到 COMSOL Server，请先调用 connect() 方法")

        # 确保传递函数已计算
        if self.transfer_functions is None:
            raise RuntimeError("传递函数未计算，请先调用 calib()")

        # 根据 mode 检查所需的系数是否已准备
        if mode == "from_calib" and self.gain_coefficients is None:
            raise RuntimeError("增益系数未计算，请先调用 calib()")
        elif mode == "from_truth":
            # 尝试加载 ground truth 系数，如果不存在会抛出异常
            try:
                self._load_ground_truth_coefficients()
            except FileNotFoundError as e:
                raise RuntimeError(str(e))

        # 存储所有迭代的输入和输出
        all_inputs: list[SimulationInput] = []
        all_outputs: list[SimulationOutput] = []

        # 当前输入从初始条件开始
        current_input = initial_input

        # 执行反馈环路迭代
        print(f"\n开始反馈环路仿真，共 {num_iterations} 次迭代...")
        print(f"反馈模式: {mode}")
        if mode == "constant":
            print(f"反馈常数: {self.feedback_constant}")
        else:
            print(f"反馈系数来源: {mode}")

        for iteration in range(num_iterations):
            print(f"\n--- 迭代 {iteration + 1}/{num_iterations} ---")

            # 运行仿真
            output = self.run_simulation(current_input)

            # 保存当前迭代的输入和输出
            all_inputs.append(current_input)
            all_outputs.append(output)

            print("✓ 仿真完成")

            # 如果不是最后一次迭代，计算下一次的输入
            if iteration < num_iterations - 1:
                current_input = self._generate_feedback(current_input, output, mode)
                print("✓ 已计算下一次迭代的输入参数")

        print("\n反馈环路仿真全部完成！")

        # 绘制探针差值折线图
        self._plot_probe_differences(all_outputs, num_iterations, mode)

        return (all_inputs, all_outputs)

    def _plot_probe_differences(
        self,
        all_outputs: list[SimulationOutput],
        num_iterations: int,
        mode: str = "all8",
    ) -> None:
        """在复平面上绘制探针差值演化轨迹并保存

        在复平面上绘制8条折线，第n条线代表每次迭代中，第n个探针减去第n+8个探针的结果
        在复平面上的演化轨迹。横轴是实部，纵轴是虚部。

        图像会自动保存到 storage/gainEP/plots/ 目录。

        Args:
            all_outputs: 所有迭代的输出结果列表
            num_iterations: 迭代次数
            mode: 反馈模式，用于生成文件名
        """
        # 计算每次迭代的8个探针差值
        # probe_diffs_history[i][j] 表示第i次迭代中，探针j与探针j+8的差值
        probe_diffs_history = []

        for output in all_outputs:
            output_array = output.to_array()
            iteration_diffs = []
            for i in range(8):
                diff = output_array[i] - output_array[i + 8]
                iteration_diffs.append(diff)
            probe_diffs_history.append(iteration_diffs)

        # 转换为NumPy数组以便处理
        probe_diffs_history = np.array(
            probe_diffs_history
        )  # shape: (num_iterations, 8)

        # 创建图形 - 使用复平面绘图
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # 定义颜色映射
        colors = plt.cm.tab10(np.linspace(0, 1, 8))

        # 绘制8条轨迹线
        for i in range(8):
            # 提取该探针差值在所有迭代中的复数值
            trajectory = probe_diffs_history[:, i]
            real_parts = trajectory.real
            imag_parts = trajectory.imag

            # 先绘制连接线（不带标记）
            ax.plot(
                real_parts,
                imag_parts,
                color=colors[i],
                linewidth=2,
                label=f"探针{i + 1} - 探针{i + 9}",
                alpha=0.6,
                zorder=1,
            )

            # 绘制所有中间迭代点（小圆点）
            ax.scatter(
                real_parts[1:-1],
                imag_parts[1:-1],
                color=colors[i],
                s=30,  # 点的大小
                alpha=0.5,
                edgecolors="white",
                linewidths=0.5,
                zorder=2,
            )

            # 标记起点（第一次迭代）- 方形
            ax.scatter(
                real_parts[0],
                imag_parts[0],
                marker="s",
                s=150,  # 更大的起点
                color=colors[i],
                edgecolors="black",
                linewidths=2,
                alpha=0.9,
                zorder=3,
            )

            # 标记终点（最后一次迭代）- 星形
            ax.scatter(
                real_parts[-1],
                imag_parts[-1],
                marker="*",
                s=300,  # 更大的终点
                color=colors[i],
                edgecolors="black",
                linewidths=2,
                alpha=0.9,
                zorder=3,
            )

            # 在起点和终点标注文字（仅对第四条线标注，避免重复）
            if i == 3:
                ax.annotate(
                    "起点",
                    xy=(real_parts[0], imag_parts[0]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                )
                ax.annotate(
                    "终点",
                    xy=(real_parts[-1], imag_parts[-1]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7
                    ),
                )

        # 绘制原点参考
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.plot(
            0,
            0,
            marker="x",
            markersize=10,
            color="red",
            markeredgewidth=2,
            label="原点",
        )

        # 设置坐标轴标签和标题
        ax.set_xlabel("实部", fontsize=12)
        ax.set_ylabel("虚部", fontsize=12)

        # 格式化标题
        if mode == "constant":
            if self.feedback_constant.imag == 0:
                mode_str = f"constant (值: {self.feedback_constant.real:.3f})"
            else:
                mode_str = f"constant (值: {self.feedback_constant.real:.3f}{self.feedback_constant.imag:+.3f}j)"
        else:
            mode_str = mode

        ax.set_title(
            f"反馈环路仿真 - 探针差值在复平面上的演化轨迹\n"
            f"(模式: {mode_str}, 迭代次数: {num_iterations})",
            fontsize=14,
            fontweight="bold",
        )

        # 设置图例
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

        # 设置网格
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

        # 设置相等的纵横比，使复平面不失真
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()

        # 生成文件名并保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.mph_file.stem}_feedback_{mode}_iter{num_iterations}_complex_plane_{timestamp}.png"
        save_path = self.plots_dir / filename

        try:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print("\n✓ 图像已保存")
            print(f"  保存路径: {save_path}")
        except Exception as e:
            print(f"\n⚠ 保存图像失败: {e}")

        plt.show()

        print("✓ 已绘制探针差值在复平面上的演化轨迹")
