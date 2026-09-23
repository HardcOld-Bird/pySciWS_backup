1.4 利用SDK编程的一般流程

<Flowchart>

初始化
-> 初始化（MT_Init）
-> 打开通信口（MT_Open_UART、MT_Open_USB）
-> 设置加速度、减速度，位置模式下最大速度（MT_Set_Axis_Acc、MT_Set_Axis_Dec、MT_Set_Axis_Position_V_Max）

-> 进入分支（在两个分支中二选一）：

速度模式分支（大括号`{}`内的部分）：{
-> 设置速度模式（MT_Set_Axis_Mode_Velocity）
-> 相对 / 绝对方式调整速度（MT_Set_Axis_Velocity_V_Target_Abs、MT_Set_Axis_Velocity_V_Target_Rel）
-> 读取状态（MT_Get_Axis_Status、MT_Get_Axis_V_Now）
-> 判断 “继续控制？”
-> 是：循环回 “相对 / 绝对方式调整速度” 步骤
-> 否：停止（MT_Set_Axis_Velocity_Stop）
}

位置模式分支（大括号`{}`内的部分）：{
-> 设置位置模式（MT_Set_Axis_Mode_Position）
-> 相对 / 绝对方式调整位置（MT_Set_Axis_Position_P_Target_Abs、MT_Set_Axis_Position_P_Target_Rel）
-> 读取状态（MT_Get_Axis_Status、MT_Get_Axis_Software_P_Now）
-> 判断 “继续控制？”
-> 是：循环回 “相对 / 绝对方式调整位置” 步骤
-> 否：停止（MT_Set_Axis_Position_Stop）
}

-> 走出分支
-> 关闭通信（MT_Close_UART、MT_Close_USB）
-> 释放资源（MT_DeInit）
-> 结束

</Flowchart>
