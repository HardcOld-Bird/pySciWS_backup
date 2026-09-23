3.17 PLC模式

3.17.1 MT_Get_PLC_Var_Num
功能描述：读取板卡上PLC模式的全局变量的数量

VC
INT32
MT_Get_PLC_Var_Num
(INT32* pValue)
VB
Function
MT_Get_PLC_Var_Num
(ByRef Value As Long) As Long
Delphi
function
MT_Get_PLC_Var_Num
(pValue:PInteger):Integer;
C#
public static extern int
MT_Get_PLC_Var_Num
(ref Int32 Value);
输入参数
无

输出参数
Value
PLC的全局变量数量
函数返回
0
函数执行成功，读取到通道数有效
非0
函数执行失败，读取到通道数无效
备注
模块的序号从0开始，N的通道，序号为0,1,2,3…N-1

3.17.2 MT_Get_PLC_Var_Data
功能描述：读取PLC模式指定序号的全局变量的值

VC
INT32
MT_Get_PLC_Var_Data
(WORD AObj,INT32* pValue);
VB
Function
MT_Get_PLC_Var_Data
(ByVal AObj As Integer, ByRef Value As
Long) As Long
Delphi
function
MT_Get_PLC_Var_Data
(AObj:Word;pValue:PInteger):Integer;
C#
public static extern int
MT_Get_PLC_Var_Data
(UInt16 AObj, ref
Int32
Value);
输入参数
AObj
PLC全局变量的序号，0..N-1
输出参数
Value
指定序号全局变量的值
函数返回
0
函数执行成功，读取到的数据有效
非0
函数执行失败，读取到的数据无效
备注
PLC全局变量的序号从0开始，N的全局变量，序号为0,1,2,3…N-1

3.17.3 MT_Set_PLC_Var_Data
功能描述：设置PLC模式指定序号的全局变量的值。

VC
INT32
MT_Set_PLC_Var_Data
(WORD AObj,INT32 Value)
VB
Function
MT_Set_PLC_Var_Data
(ByVal AObj As Integer, ByVal
Value As Long) As Long
Delphi
function
MT_Set_PLC_Var_Data
(AObj:Word;Value:Integer):Integer;
C#
public static extern int
MT_Set_PLC_Var_Data
(UInt16 AObj,
Int32
Value);
输入参数
AObj
PLC全局变量的序号，0..N-1

Value
指定序号全局变量的值
函数返回
0
函数执行成功
非0
函数执行失败
备注
PLC全局变量的序号从0开始，N的全局变量，序号为0,1,2,3…N-1

3.17.4 MT_Set_PLC_Pause
功能描述：暂停PLC模式的指令执行,使用MT_Set_PLC_Run后从停止的地方继续执行。

VC
INT32
MT_Set_PLC_Pause(void)
VB
Function
MT_Set_PLC_Pause () As Long
Delphi
function
MT_Set_PLC_Pause
():Integer;
C#
public static extern int
MT_Set_PLC_Pause();
输入参数





函数返回
0
函数执行成功
非0
函数执行失败
备注


3.17.5 MT_Set_PLC_Stop
功能描述：停止PLC模式的指令执行,使用MT_Set_PLC_Run后从头开始执行。

VC
INT32
MT_Set_PLC_Stop(void)
VB
Function
MT_Set_PLC_Stop
() As Long
Delphi
function
MT_Set_PLC_Stop
():Integer;
C#
public static extern int
MT_Set_PLC_Stop();
输入参数





函数返回
0
函数执行成功
非0
函数执行失败
备注


3.17.6 MT_Set_PLC_Run
功能描述：启动PLC模式。

VC
INT32
MT_Set_PLC_Run(void)
VB
Function
MT_Set_PLC_Run
() As Long
Delphi
function
MT_Set_PLC_Run
():Integer;
C#
public static extern int
MT_Set_PLC_Run();
输入参数





函数返回
0
函数执行成功
非0
函数执行失败
备注
