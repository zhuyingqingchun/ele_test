# 舵机模型数学公式与拓扑说明

## 1. 模型概述
当前模型是一个高保真的航天电机械作动器诊断模型，其级联结构如下：

`指令轨迹 -> 参考整形 -> 位置PD -> 速度PI -> dq电流PI -> SVPWM逆变器 -> PMSM -> 齿轮/转轴/背隙 -> 负载与气动 -> 传感器与观测器 -> 反馈`

其包含以下部分：
- PMSM `dq` 电气动态
- 逆变器一阶动态与电压矢量饱和
- 三相电流真值与测量值
- 负载侧编码器与负载速度估计器
- 电机侧编码器与电角度观测器
- 弹性转轴、背隙、摩擦、气动负载
- 绕组/机壳热网络
- 电气、机械与传感器故障的渐进式注入

## 2. 符号说明
- `theta_ref`：位置指令
- `theta_l, omega_l`：负载角度与负载速度
- `theta_m, omega_m`：电机转子角度与电机速度
- `i_d, i_q`：`dq` 电流
- `v_d, v_q`：`dq` 电压
- `p`：极对数
- `R_s`：相电阻
- `L_d, L_q`：`dq` 轴电感
- `psi_f`：永磁体磁链
- `J_m, J_l`：电机转动惯量与负载转动惯量
- `B_m, B_l`：电机与负载粘性阻尼
- `N`：齿轮传动比
- `eta_g`：齿轮效率
- `K_s, D_s`：转轴刚度与阻尼
- `T_em`：电磁转矩
- `T_shaft`：转轴转矩
- `T_load`：总负载转矩
- `T_fric`：摩擦转矩
- `U_bus`：可用母线电压
- `I_lim`：可用电流上限
- `dt`：仿真步长

## 3. 指令与参考整形
### 3.1 指令轨迹
不同工况使用不同的 `theta_ref_raw(t)` 轨迹形式，例如：多级阶跃、激烈阶跃、反向切换、正弦扫描、任务混合等。

### 3.2 一阶参考滤波
```math
\theta_{ref,f}(k)=\theta_{ref}(k-1)+\alpha_r\left(\theta_{ref,raw}(k)-\theta_{ref}(k-1)\right)
```

```math
\alpha_r = \min\left(\frac{dt}{\tau_{ref}}, 1\right)
```

### 3.3 参考速率限制
```math
\Delta \theta_{ref} = \operatorname{clip}\left(\theta_{ref,f}(k)-\theta_{ref}(k-1), -\dot\theta_{ref,max}dt, \dot\theta_{ref,max}dt\right)
```

```math
\theta_{ref}(k)=\theta_{ref}(k-1)+\Delta\theta_{ref}
```

## 4. 位置环
```math
e_\theta = \theta_{ref}-\theta_{meas}
```

```math
\omega_{l,ref}=\operatorname{clip}(K_{p,\theta}e_\theta-K_{d,\theta}\hat\omega_l, -\omega_{l,max}, \omega_{l,max})
```

```math
\omega_{m,ref}=N\,\omega_{l,ref}
```

其中，`hat{omega}_l` 由负载编码器微分后再经过一阶滤波得到。

## 5. 负载转矩观测器
### 5.1 电机加速度估计
```math
\dot\omega_{m,raw}(k)=\frac{\omega_{m,meas}(k)-\omega_{m,meas}(k-1)}{dt}
```

```math
\hat{\dot\omega}_m(k)=\hat{\dot\omega}_m(k-1)+\alpha_o\left(\dot\omega_{m,raw}(k)-\hat{\dot\omega}_m(k-1)\right)
```

```math
\alpha_o=\min\left(\frac{dt}{\tau_o},1\right)
```

### 5.2 等效参数
```math
K_t = 1.5p\psi_f
```

```math
J_{eq}=J_m+\frac{J_l}{N^2}
```

```math
B_{eq}=B_m+\frac{B_l+B_{aero}}{N^2}
```

### 5.3 负载转矩估计
```math
\hat T_{em}=K_t i_{q,meas}
```

```math
\hat T_{load,m}=\hat T_{em}-J_{eq}\hat{\dot\omega}_m-B_{eq}\omega_{m,meas}
```

### 5.4 扰动补偿电流
```math
i_{q,dist}^* = \operatorname{clip}\left( K_o\frac{\hat T_{load,m}}{K_t}, -i_{dist,max}, i_{dist,max} \right)
```

## 6. 速度环
```math
e_\omega = \omega_{m,ref}-\omega_{m,meas}
```

```math
x_\omega(k)=\operatorname{clip}(x_\omega(k-1)+e_\omega dt,-10,10)
```

```math
i_{q,raw}^*=K_{p,\omega}e_\omega+K_{i,\omega}x_\omega+i_{q,dist}^*
```

```math
i_q^*=\operatorname{clip}(i_{q,raw}^*,-I_{lim},I_{lim})
```

若发生电流饱和，则对积分器执行 anti-windup 回退。

### 6.1 d轴参考
```math
i_d^*=0
```

## 7. 电流环与补偿
电角速度：

```math
\omega_e = p\,\omega_{m,meas}
```

误差定义：

```math
e_d = i_d^*-i_{d,meas},\qquad e_q = i_q^*-i_{q,meas}
```

### 7.1 d轴控制
```math
v_{d,raw}=K_{p,d}e_d+K_{i,d}x_d-\omega_eL_q i_{q,meas}
```

### 7.2 q轴控制
```math
v_{q,raw}=K_{p,q}e_q+K_{i,q}x_q+\omega_e(L_d i_{d,meas}+\psi_f)
```

### 7.3 电压裕量前馈
对于 `voltage_margin_track`：

```math
v_{q,raw}\leftarrow v_{q,raw}+k_{ff}\min\left(\frac{U_{bus}}{U_{cond}},1\right)\psi_f\omega_e
```

当前实现中使用 `k_ff = 0.08`。

### 7.4 低温反向补偿
对于 `cold_takeoff_reversal`：

```math
w_{rev}=\begin{cases}
1, & \omega_{m,ref}\omega_{m,meas}<0 \\
0, & \text{otherwise}
\end{cases}
```

```math
i_q^* \leftarrow \operatorname{clip}\left(i_q^* + 0.22\,w_{rev}\operatorname{sgn}(\omega_{m,ref}) + 0.10\tanh\left(\frac{\omega_{m,ref}}{N}\right), -I_{lim}, I_{lim}\right)
```

随后重新计算 `e_q` 与 `v_{q,raw}`。

## 8. 电压饱和与逆变器
可用电压矢量上限：

```math
V_{max}=\frac{m_{lim}}{\sqrt{3}}U_{bus}
```

如果 `sqrt(v_d^2+v_q^2) > V_max`，则对电压矢量进行缩放：

```math
s_v = \frac{V_{max}}{\sqrt{v_{d,raw}^2+v_{q,raw}^2}}
```

```math
v_d^*=s_v v_{d,raw},\qquad v_q^*=s_v v_{q,raw}
```

否则 `s_v=1`。

逆变器一阶动态：

```math
v_{d,app}(k)=v_{d,app}(k-1)+\alpha_{inv}(v_d^*-v_{d,app}(k-1))
```

```math
v_{q,app}(k)=v_{q,app}(k-1)+\alpha_{inv}(v_q^*-v_{q,app}(k-1))
```

```math
\alpha_{inv}=\min\left(\frac{dt}{\tau_{inv}},1\right)
```

PWM 占空比近似：

```math
D = \frac{\sqrt{v_{d,app}^2+v_{q,app}^2}}{V_{max}}
```

## 9. PMSM 电气模型
### 9.1 电角度
```math
\theta_e = \operatorname{wrap}(p\theta_m)
```

### 9.2 温度相关参数
```math
R_s^{eff}=R_s\left(1+\alpha_R(T_w-25)\right)s_R
```

```math
\psi_f^{base}=\psi_f\max\left(0.72,1+\alpha_\psi(T_w-25)\right)
```

对于热降额故障：

```math
R_s^{eff}\leftarrow R_s^{eff}(1+k_{R,th}\Delta T_{over})
```

```math
\psi_f^{eff}\leftarrow \psi_f^{base}\max(0.5,1-k_{\psi,th}\Delta T_{over})
```

### 9.3 dq电流动态
```math
\dot i_d = \frac{v_{d,app}-R_s^{eff}i_d+\omega_eL_q i_q}{L_d}
```

```math
\dot i_q = \frac{v_{q,app}-R_s^{eff}i_q-\omega_e(L_d i_d+\psi_f^{eff})}{L_q}
```

### 9.4 电磁转矩
```math
T_{em}=1.5p\left(\psi_f^{eff}i_q+(L_d-L_q)i_di_q\right)
```

## 10. 机械传动与负载模型
### 10.1 转轴扭转
```math
\Delta\theta_s = \theta_m/N-\theta_l
```

### 10.2 转轴转矩
```math
T_{shaft}=K_s\Delta\theta_s + D_s\left(\omega_m/N-\omega_l\right)
```

### 10.3 摩擦转矩
```math
T_{fric}=T_c\tanh\left(\frac{\omega_m}{\omega_{eps}}\right)+B_m\omega_m
```

### 10.4 负载总转矩
```math
T_{load}=T_{aero}+T_{ext}+T_{jam}
```

### 10.5 电机侧机械动态
```math
J_m\dot\omega_m=T_{em}-T_{fric}-\frac{T_{shaft}}{\eta_g N}
```

### 10.6 负载侧机械动态
```math
J_l\dot\omega_l=T_{shaft}-B_l\omega_l-T_{load}
```

## 11. 热模型
### 11.1 绕组温度
```math
C_w\dot T_w=P_{loss}-\frac{T_w-T_h}{R_{wh}}
```

### 11.2 机壳温度
```math
C_h\dot T_h=\frac{T_w-T_h}{R_{wh}}-\frac{T_h-T_{amb}}{R_{ha}}
```

其中 `P_loss` 通常由铜耗与铁耗组成。

## 12. 观测量与信号拓扑
当前系统中的主要观测量包括：
- 负载侧位置/速度观测链
- 电机侧位置/电角度观测链
- 电流、电压、母线电流等电气量
- 转矩、转轴扭转、速度残差等机械量
- 绕组温度、机壳温度、环境温度等热量
- 振动与健康指示量

这些信号共同构成后续诊断模型的多模态输入基础。
