class StdCSSIO(HiPerfCSSIO):  # 暂停维护
    """
    # 连续同步正弦波 AI/AO 类

    该类继承自 HiPerfCSSIO，提供动态波形生成功能的同步 AI/AO 任务实现。
    使用波形生成器实时生成正弦波信号，适用于（一般由于波形周期不标准）需要动态调整输出波形的场景。

    ## 主要特性：
    - 继承 HiPerfCSSIO 的所有硬件同步功能
    - 动态波形生成（非再生模式）
    - 实时填充 AO 缓冲区
    - 支持相位连续的波形生成
    - 数据导出时包含波形元数据（如 SineArgs）
    - 适用于需要动态调整输出的测量场景

    ## 与 HiPerfCSSIO 的主要区别：
    - 初始化时接收 waveform_generator 而不是固定的 output_waveform
    - 使用 AO 非再生模式而不是再生模式
    - 需要实时填充 AO 缓冲区
    - 数据导出时使用生成的波形（包含元数据）

    ## 使用示例：
    ```python
    from sweeper400.analyze import init_sampling_info, init_sine_args, SineGenerator
    from sweeper400.measure.cont_sync_io import ContSyncSineAIAO

    # 创建采样信息和波形生成器
    sampling_info = init_sampling_info(1000, 1000)
    sine_args = init_sine_args(100.0, 1.0, 0.0)
    generator = SineGenerator(sampling_info, sine_args)

    # 定义数据导出函数
    def export_data(ai_waveform, ao_waveform, chunks_num):
        print(f"导出第 {chunks_num} 段数据")

    # 创建同步 AI/AO 对象
    sync_io = ContSyncSineAIAO(
        ai_channel="PXI1Slot2/ai0",
        ao_channel="PXI1Slot2/ao0",
        waveform_generator=generator,
        export_function=export_data
    )

    # 启动任务
    sync_io.start()
    sync_io.enable_export = True  # 开始导出数据

    # 运行一段时间后停止
    time.sleep(10)
    sync_io.stop()
    ```
    """

    def __init__(
        self,
        ai_channel: str,
        ao_channel: str,
        waveform_generator: SineGenerator,
        export_function: Callable[[Waveform, Waveform, PositiveInt], None],
    ) -> None:
        """
        初始化连续同步正弦波 AI/AO 对象

        Args:
            ai_channel: AI 通道名称，例如 "PXI1Slot2/ai0"
            ao_channel: AO 通道名称，例如 "PXI1Slot2/ao0"
            waveform_generator: 波形生成器，用于生成连续的正弦波形
            export_function: 数据导出函数，接收 (ai_waveform, ao_waveform, chunks_num) 参数

        Raises:
            ValueError: 当参数无效时
        """
        logger.info(f"初始化 ContSyncSineAIAO - AI: {ai_channel}, AO: {ao_channel}")

        # 存储波形生成器
        self._waveform_generator = waveform_generator

        # 生成初始波形用于父类初始化
        initial_waveform = waveform_generator.generate()

        # 调用父类初始化
        super().__init__(ai_channel, ao_channel, initial_waveform, export_function)

        # AO 数据队列（用于协同处理 AI/AO 数据）（AI 队列已在父类中创建）
        self._ao_queue: deque[Waveform] = deque()

        logger.debug(
            f"ContSyncSineAIAO 初始化完成 - 采样率: {self._sampling_info['sampling_rate']} Hz, "
            f"每段样本数: {self._sampling_info['samples_num']}"
        )

    def _worker_thread_function(self):
        """
        工作线程函数（重写父类方法）

        使用事件驱动机制，只在有数据时才唤醒处理，节省 CPU 资源。
        负责两个主要任务：
        1. 波形生成和 AO 缓冲区填充
        2. AI/AO 数据的协同处理和导出
        """
        logger.debug("ContSyncSineAIAO 工作线程已启动（事件驱动模式）")

        while not self._stop_event.is_set():
            try:
                # 等待数据就绪事件或超时
                # 使用较短的超时确保即使没有数据也能定期检查 AO 缓冲区
                data_ready = self._data_ready_event.wait(
                    timeout=self._chunk_duration
                )  # 超时时间为一个采样周期

                if data_ready:
                    # 清除事件标志，准备下次等待
                    self._data_ready_event.clear()
                    logger.debug("检测到数据就绪事件，开始处理")

                # 无论是否有数据事件，都执行以下任务：

                # 任务1: 波形生成和 AO 缓冲区填充
                self._wtf_fill_ao_buffer()

                # 任务2: 数据导出处理
                self._wtf_process_data_export()

            except Exception as e:
                logger.error(f"ContSyncSineAIAO 工作线程处理失败: {e}")

        logger.debug("ContSyncSineAIAO 工作线程已退出")

    def _wtf_fill_ao_buffer(self):
        """
        填充 AO 缓冲区

        尽可能地生成新波形并填充硬件 AO 缓冲区，同时将波形信息保存到 _ao_queue 中。
        """
        if self._ao_task is None or not self._is_running:
            return

        try:
            # 检查缓冲区可用空间
            space_available = self._ao_task.out_stream.space_avail  # type: ignore
            samples_per_waveform = self._sampling_info["samples_num"]

            # 尽可能多地生成波形填充缓冲区
            while space_available >= samples_per_waveform:
                # 生成新波形
                new_waveform = self._waveform_generator.generate()

                # 将附带SineArgs元数据的新波形保存到 AO 队列
                self._ao_queue.append(new_waveform)

                # 写入硬件缓冲区
                self._ao_task.write(new_waveform, auto_start=False)  # type: ignore

                # 更新可用空间
                space_available -= samples_per_waveform  # type: ignore

        except Exception as e:
            logger.debug(f"AO 缓冲区填充失败: {e}")

    def _wtf_process_data_export(self):
        """
        处理数据导出（重写父类方法）

        协同处理 AI 和 AO 队列中的数据，进行数据导出。
        """
        # 检查是否有可配对的数据
        if not self._ai_queue or not self._ao_queue:
            return

        try:
            # 从队列首部取出数据包（FIFO）
            ai_package = self._ai_queue.popleft()
            ao_waveform = (
                self._ao_queue.popleft()
            )  # ao_waveform 为具有SineArgs元数据的 Waveform 对象

            # 检查是否需要导出
            if ai_package["enable_export"]:
                # 增加导出计数
                self._exported_chunks += 1

                # 创建 AI 波形对象，使用 AO 波形的 id
                ai_waveform = Waveform(
                    np.array(ai_package["ai_data"]),
                    sampling_rate=self._sampling_info["sampling_rate"],
                    id=ao_waveform.id,
                )

                # 调用导出函数
                self.export_function(
                    ai_waveform,
                    ao_waveform,
                    self._exported_chunks,
                )

                logger.debug(f"导出第 {self._exported_chunks} 段数据")

            else:
                # 重置导出计数
                self._exported_chunks = 0

        except Exception as e:
            logger.error(f"ContSyncSineAIAO 数据导出处理失败: {e}")

    def _setup_ao_task(self):
        """
        配置 AO 任务（重写父类方法）

        与父类的主要区别：
        1. 使用非再生模式而不是再生模式
        2. 需要持续填充缓冲区
        3. 需要更大的缓冲区以避免下溢
        """
        if self._ao_task is None:
            raise RuntimeError("AO 任务未创建")

        logger.debug("配置 ContSyncSineAIAO AO 任务")

        # 添加 AO 通道
        self._ao_task.ao_channels.add_ao_voltage_chan(  # type: ignore
            self._ao_channel, min_val=-10.0, max_val=10.0
        )

        # 配置时钟源和采样
        self._ao_task.timing.ref_clk_src = "PXIe_Clk100"
        self._ao_task.timing.ref_clk_rate = 100000000
        self._ao_task.timing.cfg_samp_clk_timing(  # type: ignore
            rate=self._sampling_info["sampling_rate"],
            sample_mode=AcquisitionType.CONTINUOUS,
        )

        # 设置非再生模式（与父类的主要区别）
        self._ao_task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION

        # 配置更大的输出缓冲区以避免下溢
        buffer_size = self._sampling_info["samples_num"] * 10  # 10倍缓冲区
        self._ao_task.out_stream.output_buf_size = buffer_size
        logger.debug(f"设置 AO 缓冲区大小: {buffer_size} 样本")

        # 尽可能填满初始缓冲区，使用连续的波形生成
        self._fill_initial_ao_buffer()

        logger.debug("ContSyncSineAIAO AO 任务配置完成")

    def _fill_initial_ao_buffer(self):
        """
        填充初始 AO 缓冲区

        使用 WaveformGenerator 生成连续的波形来填充缓冲区，确保相位连续性。
        """
        try:
            # 获取缓冲区总大小
            total_buffer_size = self._ao_task.out_stream.output_buf_size  # type: ignore
            samples_per_waveform = self._sampling_info["samples_num"]

            # 计算需要生成多少个波形来填满缓冲区
            waveforms_needed = min(
                total_buffer_size // samples_per_waveform, 10  # type: ignore
            )  # 最多10个

            logger.debug(f"初始填充缓冲区: 生成 {waveforms_needed} 个连续波形")

            # 生成连续波形并填充缓冲区和队列
            for _ in range(waveforms_needed):
                # 生成新波形
                new_waveform = self._waveform_generator.generate()

                # 将附带SineArgs元数据的新波形保存到 AO 队列
                self._ao_queue.append(new_waveform)

                # 写入硬件缓冲区
                self._ao_task.write(new_waveform, auto_start=False)  # type: ignore

            logger.debug(
                f"初始缓冲区填充完成: 写入 {waveforms_needed * samples_per_waveform} 样本"
            )

        except Exception as e:
            logger.error(f"初始缓冲区填充失败: {e}")
            raise

    def _cleanup_tasks(self):
        """清理 nidaqmx 任务资源（重写父类方法）"""
        logger.debug("清理 nidaqmx 任务资源")

        # 停止 AI 任务
        if self._ai_task is not None:
            try:
                if self._ai_task._handle is not None:  # type: ignore
                    self._ai_task.stop()
                    logger.debug("AI 任务已停止")
            except Exception as e:
                logger.warning(f"停止 AI 任务时出错: {e}")

            try:
                self._ai_task.close()
                logger.debug("AI 任务已关闭")
            except Exception as e:
                logger.warning(f"关闭 AI 任务时出错: {e}")

            self._ai_task = None

        # 停止 AO 任务
        if self._ao_task is not None:
            try:
                if self._ao_task._handle is not None:  # type: ignore
                    self._ao_task.stop()
                    logger.debug("AO 任务已停止")
            except Exception as e:
                logger.warning(f"停止 AO 任务时出错: {e}")

            try:
                self._ao_task.close()
                logger.debug("AO 任务已关闭")
            except Exception as e:
                logger.warning(f"关闭 AO 任务时出错: {e}")

            self._ao_task = None

        # 清理数据队列和事件状态
        with self._callback_lock:
            self._ai_queue.clear()
            self._ao_queue.clear()

        # 重置事件状态
        self._stop_event.clear()
        self._data_ready_event.clear()

    def start(self):
"""
启动同步的连续 AI/AO 任务（重写父类方法）

配置并启动硬件同步的 AI 和 AO 任务，使用 PXIe_CLK100 时钟源和触发器同步。

Raises:
    RuntimeError: 当任务启动失败时
"""
if self._is_running:
    logger.warning("任务已在运行中")
    return

logger.info("启动 ContSyncSineAIAO 同步 AI/AO 任务")

try:
    # 创建 AI 和 AO 任务
    self._ai_task = nidaqmx.Task("ContSyncAI")
    self._ao_task = nidaqmx.Task("ContSyncAO")

    # 配置 AI 任务
    self._setup_ai_task()

    # 配置 AO 任务
    self._setup_ao_task()

    # 配置硬件同步触发
    self._setup_hardware_sync()

    # 启动工作线程
    self._stop_event.clear()
    self._worker_thread = threading.Thread(
        target=self._worker_thread_function, name="StdCSSIO_Worker"
    )
    self._worker_thread.start()

    # 启动任务（AO 先启动，等待触发）
    self._ao_task.start()
    self._ai_task.start()  # AI 启动时会触发 AO

    self._is_running = True
    logger.info("ContSyncSineAIAO 同步 AI/AO 任务启动成功")

except Exception as e:
    logger.error(f"ContSyncSineAIAO 任务启动失败: {e}")
    self._cleanup_tasks()
    raise RuntimeError(f"ContSyncSineAIAO 同步 AI/AO 任务启动失败: {e}")
