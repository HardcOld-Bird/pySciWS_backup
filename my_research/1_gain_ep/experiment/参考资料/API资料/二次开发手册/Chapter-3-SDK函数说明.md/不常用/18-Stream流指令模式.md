3.18 Stream流指令模式

3.18.1 MT_Get_Stream_Space
功能描述：读取板卡上Stream模式的指令空间空余量，有空余才能写入指令，否则会被丢弃。

VC
INT32
MT_Get_Stream_Space
(INT32* pValue)
VB
Function
MT_Get_Stream_Space
(ByRef Value As Long) As Long
Delphi
function
MT_Get_Stream_Space
(pValue:PInteger):Integer;
C#
public static extern int
MT_Get_Stream_Space
(ref Int32 Value);
输入参数
无

输出参数
Value
Stream的指令空间余量
函数返回
0
函数执行成功，读取到通道数有效
非0
函数执行失败，读取到通道数无效
备注


3.18.2 MT_Set_Stream_Pause
功能描述：暂停Stream模式的指令执行,使用MT_Set_Stream_Run后从停止的地方继续执行。

VC
INT32
MT_Set_Stream_Pause(void)
VB
Function
MT_Set_Stream_Pause () As Long
Delphi
function
MT_Set_Stream_Pause
():Integer;
C#
public static extern int
MT_Set_Stream_Pause();
输入参数





函数返回
0
函数执行成功
非0
函数执行失败
备注


3.18.3 MT_Set_Stream_Stop
功能描述：停止Stream模式的指令执行,使用MT_Set_Stream_Run后从头开始执行。

VC
INT32
MT_Set_Stream_Stop(void)
VB
Function
MT_Set_Stream_Stop
() As Long
Delphi
function
MT_Set_Stream_Stop
():Integer;
C#
public static extern int
MT_Set_Stream_Stop();
输入参数





函数返回
0
函数执行成功
非0
函数执行失败
备注


3.18.4 MT_Set_Stream_Run
功能描述：启动Stream模式。

VC
INT32
MT_Set_Stream_Run(void)
VB
Function
MT_Set_Stream_Run
() As Long
Delphi
function
MT_Set_Stream_Run
():Integer;
C#
public static extern int
MT_Set_Stream_Run();
输入参数





函数返回
0
函数执行成功
非0
函数执行失败
备注


3.18.5 MT_Set_Stream_Clear
功能描述：清除指令空间中还未执行的Stream指令。

VC
INT32
MT_Set_Stream_Clear(void)
VB
Function
MT_Set_Stream_Clear
() As Long
Delphi
function
MT_Set_Stream_Clear
():Integer;
C#
public static extern int
MT_Set_Stream_Clear();
输入参数





函数返回
0
函数执行成功
非0
函数执行失败
备注


3.18.6 MT_Set_Stream_Delay
功能描述：在Stream指令序列中插入延时等待，等待精度为1ms。延时后才继续执行后续的指令,不阻塞后续指令。
VC
INT32
MT_Set_Stream_Delay
(INT32
Value);
VB
Function
MT_Set_Stream_Delay
(ByVal
Value As
Long) As Long
Delphi
function
MT_Set_Stream_Delay
(Value:Integer):Integer;
C#
public static extern int
MT_Set_Stream_Delay
(Int32
Value);
输入参数
无

输出参数
Value
等待的ms数
函数返回
0
函数执行成功
非0
函数执行失败
备注


3.18.7 MT_Set_Stream_Line_Acc
功能描述：Stream模式设置指定插补模块的加速度,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Line_Acc
(WORD AObj,INT32 Value)
VB
Function
MT_Set_Stream_Line_Acc
(ByVal AObj As Integer, ByVal
Value As Long) As
Long
Delphi
function
MT_Set_Stream_Line_Acc
(AObj:Word;Value:Integer):Integer;
C#
public static extern int
MT_Set_Stream_Line_Acc
(UInt16 AObj,
Int32
Value);
输入参数
AObj
插补模块序号，不是插补轴的序号

Value
本插补模块的加速度
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1
3.18.8 MT_Set_Stream_Line_Dec
功能描述：Stream模式设置指定插补模块的加速度,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Line_Dec
(WORD AObj,INT32 Value)
VB
Function
MT_Set_Stream_Line_Dec
(ByVal
AObj As Integer, ByVal
Value As Long) As
Long
Delphi
function
MT_Set_Stream_Line_Dec
(AObj:Word;Value:Integer):Integer;
C#
public static extern int
MT_Set_Stream_Line_Dec
(UInt16 AObj,
Int32
Value);
输入参数
AObj
插补模块序号，不是插补轴的序号

Value
本插补模块的减速度
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1
3.18.9 MT_Set_Stream_Line_V_Max
功能描述：Stream模式设置指定插补模块的速度,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Line_V_Max
(WORD AObj,INT32 Value)
VB
Function
MT_Set_Stream_Line_V
_Max(ByVal AObj As Integer, ByVal
Value As
Long)
As Long
Delphi
function
MT_Set_Stream_Line_V
_Max(AObj:Word;Value:Integer):Integer;
C#
public static extern int
MT_Set_Stream_Line_V_Max
(UInt16 AObj,
Int32
Value);
输入参数
AObj
插补模块序号，不是插补轴的序号

Value
本插补模块的速度
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1

3.18.10 MT_Set_Stream_Circle_V_Start
功能描述：Stream模式设置指定插补模块的起始速度,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Circle_V_Start
(WORD AObj,INT32 Value)
VB
Function
MT_Set_Stream_Circle_V
_Start(ByVal AObj As Integer, ByVal
Value As Long)
As Long
Delphi
function
MT_Set_Stream_Circle_V
_Start(AObj:Word;Value:Integer):Integer;
C#
public static extern int
MT_Set_Stream_Circle_V_Start
(UInt16 AObj,
Int32
Value);
输入参数
AObj
插补模块序号，不是插补轴的序号

Value
本插补模块的速度
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1

3.18.11 MT_Set_Stream_Line_Stop
功能描述：Stream模式减速停止插补轴,不阻塞后续指令

VC
INT32
MT_Set_Stream_Line_Stop
(WORD AObj)
VB
Function
MT_Set_Stream_Line_Stop
(ByVal AObj As Integer) As Long
Delphi
function
MT_Set_Stream_Line_Stop
(AObj:Word):Integer;
C#
public static extern int
MT_Set_Stream_Line_Stop
(UInt16 AObj);
输入参数
AObj
插补模块序号，不是插补轴的序号
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1
3.18.12 MT_Set_Stream_Line_Halt
功能描述：Stream模式立即停止插补轴,不阻塞后续指令

VC
INT32
MT_Set_Stream_Line_Halt
(WORD AObj)
VB
Function
MT_Set_Stream_Line_Halt
(ByVal AObj As Integer) As Long
Delphi
function
MT_Set_Stream_Line_Halt
(AObj:Word):Integer;
C#
public static extern int
MT_Set_Stream_Line_Halt
(UInt16 AObj);
输入参数
AObj
插补模块序号，不是插补轴的序号
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1

3.18.13 MT_Set_Stream_Line_X_Run_Rel
功能描述：Stream模式相对方式启动联动插补,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Line_X_Run_Rel(WORD AObj,INT32 AStream_Num,INT32*
pStream,INT32* pTarget)
VB
Function
MT_Set_Stream_Line_X_Run_Rel (ByVal AObj As
Integer,ByVal
AStream_Num As Long,ByRef pStream
As Long,ByRef pTarget As Long) As Long
Delphi
function
MT_Set_Stream_Line_X_Run_Rel
(AObj:Word;
AStream_Num:Integer;pStream:PInteger;pTarget:PInteger):Integer;
C#
public static extern int
MT_Set_Stream_Line_X_Run_Rel(UInt16 AObj,Int32
AStream_Num,ref Int32 pStream,ref Int32 pTarget)
输入参数
AObj
插补模块序号，不是插补轴的序号

AStream_Num
参与插补的轴的数量2-X

pStream
参与插补轴的序号的数组，需要传入一个数组，这个数组的长度要大于等于参与插补轴的数量

ATarget
参与插补轴的相对目标的数组，需要传入一个数组，这个数组的长度要大于等于参与插补轴的数量
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1

3.18.14 MT_Set_Stream_Line_X_Run_Abs
功能描述：Stream模式绝对方式启动联动插补,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Line_X_Run_Abs(WORD AObj,INT32 AStream_Num,INT32*
pStream,INT32* pTarget)
VB
Function
MT_Set_Stream_Line_X_Run_Abs
(ByVal AObj As Integer,ByVal
AStream_Num As Long,ByRef pStream
As Long,ByRef pTarget As Long) As Long
Delphi
function
MT_Set_Stream_Line_X_Run_Abs
(AObj:Word;
AStream_Num:Integer;pStream:PInteger;pTarget:PInteger):Integer;
C#
public static extern int
MT_Set_Stream_Line_X_Run_Abs(UInt16 AObj,Int32
AStream_Num,ref Int32 pStream,ref Int32 pTarget)
输入参数
AObj
插补模块序号，不是插补轴的序号

AStream_Num
参与插补的轴的数量2-X

pStream
参与插补轴的序号的数组，需要传入一个数组，这个数组的长度要大于等于参与插补轴的数量

ATarget
参与插补轴的绝对目标的数组，需要传入一个数组，这个数组的长度要大于等于参与插补轴的数量
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1

3.18.15 MT_Set_Stream_Wait_Line
功能描述：等待Stream模式联动插补停止，会阻塞后续指令的执行。

VC
INT32
MT_Set_Stream_Wait_Line
(WORD AObj)
VB
Function
MT_Set_Stream_Wait_Line
(ByVal AObj As
Integer) As Long
Delphi
function
MT_Set_Stream_Wait_Line
(AObj:Word):Integer;
C#
public static extern int
MT_Set_Stream_Wait_Line
(UInt16 AObj);
输入参数
AObj
插补模块序号，不是插补轴的序号
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1

3.18.16 MT_Set_Stream_Circle_Axis
功能描述：Stream模式下设置指定插补模块中参与插补的运动轴。不同的板卡有不同数量的插补模块，多个模块可以同时进行多个圆弧插补，每个插补模块可以随意指定任意2轴参与插补，同时进行插补的模块不可以有相同的插补轴。例如，模块0用轴2，轴3插补，模块1用轴1轴2插补，这两个模块独立动作时没有问题，不可以同时动作,不阻塞后续指令。。

VC
INT32
MT_Set_Stream_Circle_Axis
(WORD AObj,INT32
Axis_ID0,INT32 Axis_ID1)
VB
Function
MT_Set_Stream_Circle_Axis
(ByVal AObj As Integer, ByVal
Axis_ID0
As
Long,
ByVal
Axis_ID1
As
Long) As Long
Delphi
function
MT_Set_Stream_Circle_Axis
(AObj:Word;Axis_ID0,Axis_ID1:Integer):Integer;
C#
public static extern int
MT_Set_Stream_Circle_Axis
(UInt16 AObj,
Int32
Axis_ID0,
Int32
Axis_ID1);
输入参数
AObj
插补模块序号，不是插补轴的序号

Axis_ID0
本插补模块的第一个插补轴序号

Axis_ID1
本插补模块的第二个插补轴序号
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1
插补轴的序号从0开始，N的轴，序号为0,1,2,3…N-1
3.18.17 MT_Set_Stream_Circle_Acc
功能描述：Stream模式下设置指定插补模块的加速度,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Circle_Acc
(WORD AObj,INT32 Value)
VB
Function
MT_Set_Stream_Circle_Acc
(ByVal AObj As Integer, ByVal
Value As Long) As
Long
Delphi
function
MT_Set_Stream_Circle_Acc
(AObj:Word;Value:Integer):Integer;
C#
public static extern int
MT_Set_Stream_Circle_Acc
(UInt16 AObj,
Int32
Value);
输入参数
AObj
插补模块序号，不是插补轴的序号

Value
本插补模块的加速度
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1
3.18.18 MT_Set_Stream_Circle_Dec
功能描述：Stream模式下设置指定插补模块的加速度,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Circle_Dec
(WORD AObj,INT32 Value)
VB
Function
MT_Set_Stream_Circle_Dec
(ByVal AObj As Integer, ByVal
Value As Long) As
Long
Delphi
function
MT_Set_Stream_Circle_Dec
(AObj:Word;Value:Integer):Integer;
C#
public static extern int
MT_Set_Stream_Circle_Dec
(UInt16 AObj,
Int32
Value);
输入参数
AObj
插补模块序号，不是插补轴的序号

Value
本插补模块的减速度
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1
3.18.19 MT_Set_Stream_Circle_V_Max
功能描述：Stream模式下设置指定插补模块的速度,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Circle_V_Max
(WORD AObj,INT32 Value)
VB
Function
MT_Set_Stream_Circle_V
_Max(ByVal AObj As Integer, ByVal
Value As Long)
As Long
Delphi
function
MT_Set_Stream_Circle_V
_Max(AObj:Word;Value:Integer):Integer;
C#
public static extern int
MT_Set_Stream_Circle_V_Max
(UInt16 AObj,
Int32
Value);
输入参数
AObj
插补模块序号，不是插补轴的序号

Value
本插补模块的速度
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1

3.18.20 MT_Set_Stream_Circle_V_Start
功能描述：Stream模式下设置指定插补模块的起始速度,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Circle_V_Start
(WORD AObj,INT32 Value)
VB
Function
MT_Set_Stream_Circle_V
_Start(ByVal AObj As Integer, ByVal
Value As Long)
As Long
Delphi
function
MT_Set_Stream_Circle_V
_Start(AObj:Word;Value:Integer):Integer;
C#
public static extern int
MT_Set_Stream_Circle_V_Start
(UInt16 AObj,
Int32
Value);
输入参数
AObj
插补模块序号，不是插补轴的序号

Value
本插补模块的起始速度
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1

3.18.21 MT_Set_Stream_Circle_Stop
功能描述：Stream模式下减速停止插补轴,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Circle_Stop
(WORD AObj)
VB
Function
MT_Set_Stream_Circle_Stop
(ByVal AObj As Integer) As Long
Delphi
function
MT_Set_Stream_Circle_Stop
(AObj:Word):Integer;
C#
public static extern int
MT_Set_Stream_Circle_Stop
(UInt16 AObj);
输入参数
AObj
插补模块序号，不是插补轴的序号
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1
3.18.22 MT_Set_Stream_Circle_Halt
功能描述：Stream模式下立即停止插补轴,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Circle_Halt
(WORD AObj)
VB
Function
MT_Set_Stream_Circle_Halt
(ByVal AObj As Integer)
As Long
Delphi
function
MT_Set_Stream_Circle_Halt
(AObj:Word):Integer;
C#
public static extern int
MT_Set_Stream_Circle_Halt
(UInt16 AObj);
输入参数
AObj
插补模块序号，不是插补轴的序号
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1
3.18.23 MT_Set_Stream_Circle_R_CW_Run_Rel
功能描述：Stream模式下以相对当前的位置的参数顺时针圆弧插补，立即启动,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Circle_R_CW_Run_Rel
(WORD AObj,INT32 AR,INT32
Axis_Target0,INT32 Axis_Target1)
VB
Function
MT_Set_Stream_Circle_R_CW_Run_Rel
(ByVal AObj As Integer,ByVal AR As
Long, ByVal
Axis_Target0
As
Long, ByVal
Axis_Target1
As
Long) As Long
Delphi
Function
MT_Set_Stream_Circle_R_CW_Run_Rel
(AObj:Word;AR,Axis_Target0,Axis_Target1:Integer):Integer
C#
public static extern int
MT_Set_Stream_Circle_R_CW_Run_Rel
(UInt16 AObj,Int32
AR,
Int32
Axis_Target0,
Int32
Axis_Target1);
输入参数
AObj
插补模块序号，不是插补轴的序号

AR
圆弧插补的半径，正数为劣弧，负数为优弧，半径为圆弧的半径，无相对绝对之分

Axis_Target0
本插补模块的第一个插补轴相对移动的距离

Axis_Target1
本插补模块的第二个插补轴相对移动的距离
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1
3.18.24 MT_Set_Stream_Circle_R_CW_Run_Abs
功能描述：Stream模式下以绝对的位置的参数顺时针圆弧插补，立即启动,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Circle_R_CW_Run_Abs
(WORD AObj,INT32 AR,INT32
Axis_Target0,INT32 Axis_Target1)
VB
Function
MT_Set_Stream_Circle_R_CW_Run_Abs
(ByVal AObj As
Integer,ByVal AR As
Long, ByVal
Axis_Target0
As
Long, ByVal
Axis_Target1
As
Long) As Long
Delphi
Function
MT_Set_Stream_Circle_R_CW_Run_Abs
(AObj:Word;AR,Axis_Target0,Axis_Target1:Integer):Integer
C#
public static extern int
MT_Set_Stream_Circle_R_CW_Run_Abs
(UInt16 AObj,Int32
AR,
Int32
Axis_Target0,
Int32
Axis_Target1);
输入参数
AObj
插补模块序号，不是插补轴的序号

AR
圆弧插补的半径，正数为劣弧，负数为优弧，半径为圆弧的半径，无相对绝对之分

Axis_Target0
本插补模块的第一个插补轴需要移动到的绝对位置

Axis_Target1
本插补模块的第二个插补轴需要移动到的绝对位置
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1
3.18.25 MT_Set_Stream_Circle_R_CCW_Run_Rel
功能描述：Stream模式下以相对当前的位置的参数逆时针圆弧插补，立即启动,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Circle_R_CCW_Run_Rel
(WORD AObj,INT32 AR,INT32
Axis_Target0,INT32 Axis_Target1)
VB
Function
MT_Set_Stream_Circle_R_CCW_Run_Rel
(ByVal AObj As Integer,ByVal AR As
Long, ByVal
Axis_Target0
As
Long, ByVal
Axis_Target1
As
Long) As Long
Delphi
Function
MT_Set_Stream_Circle_R_CCW_Run_Rel
(AObj:Word;AR,Axis_Target0,Axis_Target1:Integer):Integer
C#
public static extern int
MT_Set_Stream_Circle_R_CCW_Run_Rel
(UInt16 AObj,Int32
AR,
Int32
Axis_Target0,
Int32
Axis_Target1);
输入参数
AObj
插补模块序号，不是插补轴的序号

AR
圆弧插补的半径，正数为劣弧，负数为优弧，半径为圆弧的半径，无相对绝对之分

Axis_Target0
本插补模块的第一个插补轴相对移动的距离

Axis_Target1
本插补模块的第二个插补轴相对移动的距离
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1
3.18.26 MT_Set_Stream_Circle_R_CCW_Run_Abs
功能描述：Stream模式下以绝对的位置的参数逆时针圆弧插补，立即启动,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Circle_R_CCW_Run_Abs
(WORD AObj,INT32 AR,INT32
Axis_Target0,INT32 Axis_Target1)
VB
Function
MT_Set_Stream_Circle_R_CCW_Run_Abs
(ByVal AObj As Integer,ByVal AR
As Long, ByVal
Axis_Target0
As
Long, ByVal
Axis_Target1
As
Long) As Long
Delphi
Function
MT_Set_Stream_Circle_R_CCW_Run_Abs
(AObj:Word;AR,Axis_Target0,Axis_Target1:Integer):Integer
C#
public static extern int
MT_Set_Stream_Circle_R_CCW_Run_Abs
(UInt16 AObj,Int32
AR,
Int32
Axis_Target0,
Int32
Axis_Target1);
输入参数
AObj
插补模块序号，不是插补轴的序号

AR
圆弧插补的半径，正数为劣弧，负数为优弧，半径为圆弧的半径，无相对绝对之分

Axis_Target0
本插补模块的第一个插补轴需要移动到的绝对位置

Axis_Target1
本插补模块的第二个插补轴需要移动到的绝对位置
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1

3.18.27 MT_Set_Stream_Wait_Circle
功能描述：等待Stream模式圆弧插补停止，会阻塞后续指令的执行。

VC
INT32
MT_Set_Stream_Wait_Circle
(WORD AObj)
VB
Function
MT_Set_Stream_Wait_Circle
(ByVal AObj As Integer) As
Long
Delphi
function
MT_Set_Stream_Wait_Circle(AObj:Word):Integer;
C#
public static extern int
MT_Set_Stream_Wait_Circle
(UInt16 AObj);
输入参数
AObj
插补模块序号，不是插补轴的序号
函数返回
0
函数执行成功
非0
函数执行失败
备注
插补模块的序号从0开始，N的插补模块，序号为0,1,2,3…N-1

3.18.28 MT_Set_Stream_Optic_Out_Single
功能描述：Stream模式下设置指定光电隔离输出通道上的电平状态,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Optic_Out_Single
(WORD AObj,INT32
pValue);
VB
Function
MT_Set_Stream_Optic_Out_Single
(ByVal AObj As Integer, ByVal
Value As
Long) As Long
Delphi
function
MT_Set_Stream_Optic_Out_Single
(AObj:Word;Value:Integer):Integer;
C#
public static extern int
MT_Set_Stream_Optic_Out_Single
(UInt16 AObj,
Int32
Value);
输入参数
AObj
光电隔离输出通道号
输出参数
Value
指定通道上的电平状态，1为高电平，0为低电平
函数返回
0
函数执行成功
非0
函数执行失败
备注
通道的序号从0开始，N的通道，序号为0,1,2,3…N-1

3.18.29 MT_Set_Stream_Optic_Out_All
功能描述：Stream模式下设置光电隔离输出所有通道上的电平状态,不阻塞后续指令。

VC
INT32
MT_Set_Stream_Optic_Out_All
(INT32
Value);
VB
Function
MT_Set_Stream_Optic_Out_All
(ByVal
Value As
Long) As Long
Delphi
function
MT_Set_Stream_Optic_Out_All
(Value:Integer):Integer;
C#
public static extern int
MT_Set_Stream_Optic_Out_All
(Int32
Value);
输入参数
无

输出参数
Value
所有通道上的电平状态，1为高电平，0为低电平
通道0为LSB(最低位)
通道N对应第N-1位
函数返回
0
函数执行成功
非0
函数执行失败
备注
通道的序号从0开始，N的通道，序号为0,1,2,3…N-1

3.18.30 MT_Set_Stream_OC_Out_Single
功能描述：Stream模式下设置指定OC输出通道上的电平状态,不阻塞后续指令

VC
INT32
MT_Set_Stream_OC_Out_Single
(WORD AObj,INT32
pValue);
VB
Function
MT_Set_Stream_OC_Out_Single
(ByVal AObj As Integer, ByVal
Value As
Long) As Long
Delphi
function
MT_Set_Stream_OC_Out_Single
(AObj:Word;Value:Integer):Integer;
C#
public static extern int
MT_Set_Stream_OC_Out_Single
(UInt16 AObj,
Int32
Value);
输入参数
AObj
光电隔离输出通道号
输出参数
Value
指定通道上的电平状态，1为高电平，0为低电平
函数返回
0
函数执行成功
非0
函数执行失败
备注
通道的序号从0开始，N的通道，序号为0,1,2,3…N-1

3.18.31 MT_Set_Stream_OC_Out_All
功能描述：Stream模式下设置OC输出所有通道上的电平状态,不阻塞后续指令

VC
INT32
MT_Set_Stream_OC_Out_All
(INT32
Value);
VB
Function
MT_Set_Stream_OC_Out_All
(ByVal
Value As
Long) As Long
Delphi
function
MT_Set_Stream_OC_Out_All
(Value:Integer):Integer;
C#
public static extern int
MT_Set_Stream_OC_Out_All
(Int32
Value);
输入参数
无

输出参数
Value
所有通道上的电平状态，1为高电平，0为低电平
通道0为LSB(最低位)
通道N对应第N-1位
函数返回
0
函数执行成功
非0
函数执行失败
备注
通道的序号从0开始，N的通道，序号为0,1,2,3…N-1

3.18.32 MT_Set_Stream_Dec_Enable
功能描述：Stream模式中运动（直线插补和圆弧插补）中连续两段中间允许减速后再加速。

VC
INT32
MT_Set_Stream_Dec_Enable
(void)
VB
Function
MT_Set_Stream_Dec_Enable
() As Long
Delphi
function
MT_Set_Stream_Dec_Enable
():Integer;
C#
public static extern int
MT_Set_Stream_Dec_Enable
();
输入参数





函数返回
0
函数执行成功
非0
函数执行失败
备注


3.18.33 MT_Set_Stream_Dec_Disable
功能描述：Stream模式中运动（直线插补和圆弧插补）中连续两段中间不允许有减速后再加速。但是可以通过速度设置指令来修改速度，建议在第一段后最后一段路径允许加减速

VC
INT32
MT_Set_Stream_Dec_Disable
(void)
VB
Function
MT_Set_Stream_Dec_Disable
() As Long
Delphi
function
MT_Set_Stream_Dec_Disable
():Integer;
C#
public static extern int
MT_Set_Stream_Dec_Disable
();
输入参数





函数返回
0
函数执行成功
非0
函数执行失败
备注
