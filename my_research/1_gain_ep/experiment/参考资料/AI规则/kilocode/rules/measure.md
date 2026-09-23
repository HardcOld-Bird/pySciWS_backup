<Measuring_Rules>

# Measuring Rules

以下Rules适用于NI数据采集相关功能（"src/sweeper400/measure"目录下）的开发。

## 概况：
- 我们正在python中使用nidaqmx包，控制NI数据采集卡进行数据采集工作。
- 如需查看nidaqmx包具体细节/官方例程，我已经将该包的完整代码仓库clone在本工作区的"参考资料/nidaqmx-python-master"处。（但你无需使用该目录中的文件，因为nidaqmx包已经安装在我们的Python环境中，你可以直接import。）

## 系统组成和结构：
- 目前，有一台型号为PXIe-1083的NI机箱通过Thunderbolt3线缆连接至本PC的雷电接口。
- 这台机箱共搭载5张PXIe-4468 DSA Analog I/O板卡，名称分别为"PXI1Slot2"、"PXI1Slot3"、"PXI1Slot4"、"PXI1Slot5"、"PXI1Slot6"。官方手册指出，这些板卡均可以使用各自机箱中的PXIe_CLK100、PXI_CLK10或PXIe_SYNC100时钟作为硬件采样时钟源。
- 对于我们的每一张板卡，其上都具有"ai0"和"ai1"两个模拟输入通道，"ao0"和"ao1"两个模拟输出通道，以及一个PFI数字I/O接口。也即，本系统共具有10个AI通道，10个AO通道，以及5个PFI接口。（NI MAX软件可正常找到这些通道，并可正常开始数据采集任务）

## 同步触发和同步时钟：
- 由于我们的数据采集情景对时间精度（相位）非常敏感，我们希望所有任务（AI和AO）都使用触发器进行严格的同步触发，并使用同一个时钟作为采样时钟源。
- 板卡具体的时序引擎、触发器、PFI接口名称等信息，详见“参考资料/板卡接口列表与连线.md”。

## 其他：
- 所有ai和ao通道的电压范围均为±10.0 V (true peak)，或RMS 7.07 V (Sine Wave)。为了设备的安全，任何情况下都不要令电压范围超过该值。（你可以默认使用该值作为AI/AO任务的相应参数）
- 我们目前的具体工作方式是，在"src/sweeper400/measure/"目录中实现所有数据采集相关的函数/类/方法/属性，并在"tests"文件夹中编写测试，调用相关功能。

</Measuring_Rules>
