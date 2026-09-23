MT系列运动控制产品软件二次开发手册
（MT_API.dll版本：v3.11 ）

目录

---

# 1 SDK概述

## 1.1 SDK约定

## 1.2 运动控制相关的概念
1.2.1 运动的类型
1.2.2 运动控制系统组成
1.2.3 限位开关、零位开关和软件限位
1.2.4 加速和减速
1.2.5 电机控制内部逻辑组成
1.2.6 步进细分
1.2.7 丝杠螺距
1.2.8 传动比/减速比
1.2.9 编码器
1.2.10 光栅尺
1.2.11 开环控制
1.2.12 闭环控制
1.2.13 相对和绝对

## 1.3 运动模式
1.3.1 开环零位模式
1.3.2 闭环零位模式
1.3.3 速度模式
1.3.4 开环位置模式
1.3.5 闭环位置模式
1.3.6 联动插补模式
1.3.7 圆弧插补模式
1.3.8 流指令模式
1.3.9 PLC模式
1.3.10 软件限位模式

## 1.4 利用SDK编程的一般流程图
1.4.1 速度模式和位置模式
1.4.2 多轴联动插补模式(直线插补)
1.4.3 圆弧插补模式
1.4.4 流指令模式
1.4.5 PLC模式

# 2 SDK的注意事项

## 2.1 线程安全性

## 2.2 SET类型函数执行有效性

## 2.3 GET类型函数的返回值

## 2.4 函数的适应性

## 2.5 光栅尺和编码器的计数

# 3 SDK函数说明

## 3.1 初始化
3.1.1 MT_Init
3.1.2 MT_DeInit
3.1.3 MT_Get_Dll_Version

## 3.2 通信端口
3.2.1 MT_Close_UART
3.2.2 MT_Open_UART
3.2.3 MT_Close_USB
3.2.4 MT_Open_USB
3.2.5 MT_Close_Net
3.2.6 MT_Open_Net

## 3.3 通信握手
3.3.1 MT_Check

## 3.4 硬件信息
3.4.1 MT_Get_Product_Resource
3.4.2 MT_Get_Product_ID
3.4.3 MT_Get_Product_SN
3.4.4 MT_Get_Product_ Version[兼容]
3.4.5 MT_Get_Product_ Version2

## 3.5 零位模式(开环和闭环)
3.5.1 MT_Set_Axis_Mode_Home[兼容]
3.5.2 MT_Set_Axis_Mode_Home_Home_Switch
3.5.3 MT_Set_Axis_Mode_Home_ Encoder_Index
3.5.4 MT_Set_Axis_Mode_Home_ Encoder_Home_Switch
3.5.5 MT_Set_Axis_Home_V
3.5.6 MT_Set_Axis_Home_Stop
3.5.7 MT_Set_Axis_Home_V_Start
3.5.8 MT_Set_Axis_Home_Acc
3.5.9 MT_Set_Axis_Home_Dec
3.5.10 MT_Get_Axis_Home_Acc
3.5.11 MT_Get_Axis_Home_Dec

## 3.6 速度模式
3.6.1 MT_Set_Axis_Mode_Velocity
3.6.2 MT_Get_Axis_Velocity_V_Target
3.6.3 MT_Set_Axis_Velocity_V_Target_Abs
3.6.4 MT_Set_Axis_Velocity_V_Target_Rel
3.6.5 MT_Set_Axis_Velocity_Stop
3.6.6 MT_Set_Axis_Velocity_V_Start
3.6.7 MT_Set_Axis_Velocity_Acc
3.6.8 MT_Set_Axis_Velocity_Dec
3.6.9 MT_Get_Axis_Velocity_Acc
3.6.10 MT_Get_Axis_Velocity_Dec

## 3.7 位置模式(开环和闭环)
3.7.1 MT_Set_Axis_Mode_Position[兼容]
3.7.2 MT_Set_Axis_Mode_Position_Open
3.7.3 MT_Set_Axis_Mode_Position_Close
3.7.4 MT_Get_Axis_Position_V_Max
3.7.5 MT_Set_Axis_Position_V_Max
3.7.6 MT_Get_Axis_Position_P_Target
3.7.7 MT_Set_Axis_Position_P_Target_Abs
3.7.8 MT_Set_Axis_Position_P_Target_Rel
3.7.9 MT_Set_Axis_Position_Stop
3.7.10 MT_Set_Axis_Position_V_Start
3.7.11 MT_Set_Axis_Position_Acc
3.7.12 MT_Set_Axis_Position_Dec
3.7.13 MT_Get_Axis_Position_Acc
3.7.14 MT_Get_Axis_Position_Dec
3.7.15 MT_Set_Axis_Position_Close_Dec_Factor
3.7.16 MT_Set_Encoder_Over_Enable
3.7.17 MT_Set_Encoder_Over_Max
3.7.18 MT_Set_Encoder_Over_Stable

## 3.8 编码器/光栅尺接口配置
3.8.1 MT_Set_Encoder_Z_Polarity
3.8.2 MT_Set_Encoder_Dir_Polarity

## 3.9 软件限位
3.9.1 MT_Set_Axis_Software_Limit_Neg_Value
3.9.2 MT_Set_Axis_Software_Limit_Pos_Value
3.9.3 MT_Set_Axis_Software_Limit_Enable
3.9.4 MT_Set_Axis_Software_Limit_Disable

## 3.10 通用电机相关函数
3.10.1 MT_Get_Axis_Num
3.10.2 MT_Get_Encoder_Num
3.10.3 MT_Get_Axis_Mode
3.10.4 MT_Get_Axis_Acc[兼容]
3.10.5 MT_Set_Axis_Acc[兼容]
3.10.6 MT_Get_Axis_Dec[兼容]
3.10.7 MT_Set_Axis_Dec[兼容]
3.10.8 MT_Get_Axis_V_Now
3.10.9 MT_Get_Axis_Software_P
3.10.10 MT_Set_Axis_Software_P
3.10.11 MT_Get_Axis_Status[兼容]
3.10.12 MT_Get_Axis_Status2
3.10.13 MT_Get_Axis_Status_Run
3.10.14 MT_Get_Axis_Status_Dir
3.10.15 MT_Get_Axis_Status_Neg
3.10.16 MT_Get_Axis_Status_Pos
3.10.17 MT_Get_Axis_Status_Zero
3.10.18 MT_Get_Axis_Status_Mode
3.10.19 MT_Get_Axis_Encoder_Pos
3.10.20 MT_Set_Axis_Encoder_Pos
3.10.21 MT_Set_Axis_Halt
3.10.22 MT_Set_Axis_Halt_All

## 3.11 存储器操作
3.11.1 MT_Get_Param_Mem_Len
3.11.2 MT_Get_Param_Mem_Data
3.11.3 MT_Set_Param_Mem_Data

## 3.12 光电隔离输入
3.12.1 MT_Get_Optic_In_Num
3.12.2 MT_Get_Optic_In_Single
3.12.3 MT_Get_Optic_In_All

## 3.13 光电隔离输出
3.13.1 MT_Get_Optic_Out_Num
3.13.2 MT_Set_Optic_Out_Single
3.13.3 MT_Set_Optic_Out_All

## 3.14 OC输出
3.14.1 MT_Get_OC_Out_Num
3.14.2 MT_Set_OC_Out_Single
3.14.3 MT_Set_OC_Out_All

## 3.15 多轴联动插补(包括直线插补)
3.15.1 MT_Set_Axis_Line_Axis
3.15.2 MT_Set_Axis_Line_Acc
3.15.3 MT_Set_Axis_Line_Dec
3.15.4 MT_Set_Axis_Line_V
3.15.5 MT_Set_Axis_Line_Run
3.15.6 MT_Set_Axis_Line_Stop
3.15.7 MT_Set_Axis_Line_Halt
3.15.8 MT_Set_Axis_Line_Rel
3.15.9 MT_Set_Axis_Line_Abs
3.15.10 MT_Set_Axis_Line_Run_Rel
3.15.11 MT_Set_Axis_Line_Run_Abs
3.15.12 MT_Get_Axis_Line_Num
3.15.13 MT_Get_Axis_Line_Status
3.15.14 MT_Get_Axis_Line_Acc
3.15.15 MT_Get_Axis_Line_Dec
3.15.16 MT_Get_Axis_Line_V
3.15.17 MT_Get_Axis_Line_Axis
3.15.18 MT_Set_Axis_Line_V_Start
3.15.19 MT_Set_Axis_Line_X_Count
3.15.20 MT_Set_Axis_Line_X_Axis
3.15.21 MT_Set_Axis_Line_X_Target_Rel
3.15.22 MT_Set_Axis_Line_X_Target_Abs
3.15.23 MT_Set_Axis_Line_X_Run_Rel
3.15.24 MT_Set_Axis_Line_X_Run_Abs

## 3.16 圆弧插补
3.16.1 MT_Set_Axis_Circle_Axis
3.16.2 MT_Set_Axis_Circle_Acc
3.16.3 MT_Set_Axis_Circle_Dec
3.16.4 MT_Set_Axis_Circle_V
3.16.5 MT_Set_Axis_Circle_Stop
3.16.6 MT_Set_Axis_Circle_Halt
3.16.7 MT_Set_Axis_Circle_R_CW_Run_Rel
3.16.8 MT_Set_Axis_Circle_R_CW_Run_Abs
3.16.9 MT_Set_Axis_Circle_R_CCW_Run_Rel
3.16.10 MT_Set_Axis_Circle_R_CCW_Run_Abs
3.16.11 MT_Get_Axis_Circle_Num
3.16.12 MT_Get_Axis_Circle_Status
3.16.13 MT_Get_Axis_Circle_Acc
3.16.14 MT_Get_Axis_Circle_Dec
3.16.15 MT_Get_Axis_Circle_V
3.16.16 MT_Get_Axis_Circle_Axis
3.16.17 MT_Set_Axis_Circle_V_Start

## 3.17 PLC模式
3.17.1 MT_Get_PLC_Var_Num
3.17.2 MT_Get_PLC_Var_Data
3.17.3 MT_Set_PLC_Var_Data
3.17.4 MT_Set_PLC_Pause
3.17.5 MT_Set_PLC_Stop
3.17.6 MT_Set_PLC_Run

## 3.18 Stream流指令模式
3.18.1 MT_Get_Stream_Space
3.18.2 MT_Set_Stream_Pause
3.18.3 MT_Set_Stream_Stop
3.18.4 MT_Set_Stream_Run
3.18.5 MT_Set_Stream_Clear
3.18.6 MT_Set_Stream_Delay
3.18.7 MT_Set_Stream_Line_Acc
3.18.8 MT_Set_Stream_Line_Dec
3.18.9 MT_Set_Stream_Line_V_Max
3.18.10 MT_Set_Stream_Circle_V_Start
3.18.11 MT_Set_Stream_Line_Stop
3.18.12 MT_Set_Stream_Line_Halt
3.18.13 MT_Set_Stream_Line_X_Run_Rel
3.18.14 MT_Set_Stream_Line_X_Run_Abs
3.18.15 MT_Set_Stream_Wait_Line
3.18.16 MT_Set_Stream_Circle_Axis
3.18.17 MT_Set_Stream_Circle_Acc
3.18.18 MT_Set_Stream_Circle_Dec
3.18.19 MT_Set_Stream_Circle_V_Max
3.18.20 MT_Set_Stream_Circle_V_Start
3.18.21 MT_Set_Stream_Circle_Stop
3.18.22 MT_Set_Stream_Circle_Halt
3.18.23 MT_Set_Stream_Circle_R_CW_Run_Rel
3.18.24 MT_Set_Stream_Circle_R_CW_Run_Abs
3.18.25 MT_Set_Stream_Circle_R_CCW_Run_Rel
3.18.26 MT_Set_Stream_Circle_R_CCW_Run_Abs
3.18.27 MT_Set_Stream_Wait_Circle
3.18.28 MT_Set_Stream_Optic_Out_Single
3.18.29 MT_Set_Stream_Optic_Out_All
3.18.30 MT_Set_Stream_OC_Out_Single
3.18.31 MT_Set_Stream_OC_Out_All
3.18.32 MT_Set_Stream_Dec_Enable
3.18.33 MT_Set_Stream_Dec_Disable

## 3.19 辅助计算
3.19.1 MT_ Help_Step_Line_Real_To_Steps
3.19.2 MT_ Help_Step_Circle_Real_To_Steps
3.19.3 MT_ Help_Step_Line_Steps_To_Real
3.19.4 MT_ Help_Step_Circle_Steps_To_Real
3.19.5 MT_ Help_Encoder_Line_Real_To_Steps
3.19.6 MT_ Help_Encoder_Circle_Real_To_Steps
3.19.7 MT_ Help_Encoder_Line_Steps_To_Real
3.19.8 MT_ Help_Encoder_Circle_Steps_To_Real
3.19.9 MT_ Help_Grating_Line_Real_To_Steps
3.19.10 MT_ Help_Grating_Circle_Real_To_Steps
3.19.11 MT_ Help_Grating_Line_Steps_To_Real
3.19.12 MT_ Help_Grating_Circle_Steps_To_Real
3.19.13 MT_ Help_Encoder_Factor
3.19.14 MT_ Help_Grating_Line_Factor
3.19.15 MT_Help_Grating_Circle_Factor
