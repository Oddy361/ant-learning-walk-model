import numpy as np
import math
from image_processing import get_img_view
from image_processing import img_wrapper
import image_processing
import scipy.io as sio
import numpy as np
import cv2
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from image_processing import visual_sense
from insect_brain_model import MushroomBodyModel, CentralComplexModel
from scipy.stats import vonmises
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
# Training MB Network
from visual_homing import VisualHomingAgent
from concurrent.futures import ThreadPoolExecutor


# 此文件包含一些建模与其他计算的一些常用函数文件
""" 计算Agent相对巢穴的方向"""
def Get_HeadingFun(X_Coodinate,Y_Coodinate):
    # Calculate the angle theta with respect to the x-axis using atan2 for the vector (-X, -Y)
    theta = math.atan2(-Y_Coodinate, -X_Coodinate)
    
    # Convert angle to the range [0, 2*pi)
    if theta < 0:
        theta += 2 * math.pi
    
    return theta

def Get_AllHeading(VH_Instance,X_Coodinate,Y_Coodinate,world,N=4):
    # 本函数需要完成在一个确定的位置，利用已经训练好的蘑菇体，计算所有方向的熟悉度
    # 参数解释
    # VH_Instance：VisualHomingAgent类，用于给出已经训练好的蘑菇体
    # X_Coodinate：当前蚂蚁所在的横坐标
    # Y_Coodinate：当前蚂蚁所在的纵坐标
    # N：间隔N°，此处必须注意N的输入必须是360的约数，否则报错

    if 360 % N != 0:
        raise ValueError("N必须是360的约数")

    # 不同角度所得到的熟悉度矩阵初始化

    Familiarity_Angle=np.zeros(int(360/N));z=0.01

    # 测试角度为每隔360/N°,在此构造一个每隔4°的方向
    Angle=(np.linspace(N,360,int(360/N))*np.pi*2)/360

    for j in range(int(360/N)):
        # 确定角度Theta 其单位也为弧度制
        Theta=Angle[j]
        # 利用θ与给定坐标轴在世界中返回当前所在视图
        # 对视图的振幅与相位求解
        # 获取Agent当前方向的视图，并直接压缩、归一化、调整为向量
        Img_Collect = (1 - np.double(cv2.resize(
            image_processing.get_img_view(
                world, X_Coodinate, Y_Coodinate, z , Theta,
                res=1, hfov_d=360, v_max=np.pi / 2, v_min=-np.pi / 12,
                wrap=False, blur=False, blur_kernel_size=3
            ), (36, 10)
        )) / 255).flatten()
        
        Familiarity_Angle[j] = VH_Instance.mb.run(Img_Collect)
        
    return Familiarity_Angle

def Get_BestHeading(Familiarity_Angle):
    """
    找到熟悉度最小的方向并返回当前朝向的角度。
    
    :param F: N 维向量，其中第 k 项是指 (k-1) * 360 / N 度方向对应的熟悉度
    :return: En值最小的方向的角度，也就是最熟悉的角度
    """
    N = len(Familiarity_Angle)
    # 找到熟悉度最小值的索引
    min_index = np.argmin(Familiarity_Angle)
    # 计算对应的角度
    min_angle = (min_index) * 360 / N
    # 尝试使用平均值对图像进行构建
    # Model1 最小值
    min_Familiarity=Familiarity_Angle[min_index]
    # Model2 熟悉度平均值
    # ！！！min_Familiarity=np.mean(Familiarity_Angle)
    return min_Familiarity,min_angle

"""Helper Function"""
# 数组与数值的混合数据处理
# 定义一个函数来处理数组中的嵌套数组
def flatten_coords(coords):
    flattened = []
    for coord in coords:
        if isinstance(coord, np.ndarray):
            flattened.append(coord.item())  # 提取单个数值
        else:
            flattened.append(coord)
    return flattened

# 注意该函数需要输入弧度制
# 将弧度转移到[-pi, pi]范围内的函数
def wrap_to_pi(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


"""构建Image Memory"""
def Construction_Memory(CoodinateX,CoodinateY,world):
    
    # 本函数基于给定的坐标位置与世界，返回构造Route_Memory的所有必要信息

    Heading=Get_HeadingFun(CoodinateX,CoodinateY);z=0.01

    # 获取Agent当前方向的视图，并直接压缩、归一化、调整为向量
    Img_Collect = (1 - np.double(cv2.resize(
        image_processing.get_img_view(
            world, CoodinateX, CoodinateY, z, Heading,
            res=1, hfov_d=360, v_max=np.pi / 2, v_min=-np.pi / 12,
            wrap=False, blur=False, blur_kernel_size=3
        ), (36, 10)
    )) / 255).flatten()

    # 返回坐标位置，朝向，与当前视角对应的视图像素强度

    return CoodinateX,CoodinateY,Heading,Img_Collect

""" ZM Memory """
def Construction_ZM_Memory(CoodinateX,CoodinateY,world):
    # 本函数基于给定的坐标位置与世界，构建Homing视图的ZM记忆，返回振幅信息与卷起来的视图
    Heading=Get_HeadingFun(CoodinateX,CoodinateY);z=0.01
    # 获取蚂蚁在所设置坐标位置获得的图像
    TestView_Img=image_processing.get_img_view(world, CoodinateX, CoodinateY, z, Heading, res=1, hfov_d=360, v_max=np.pi / 2, v_min=-np.pi / 12,
                 wrap=True, blur=False, blur_kernel_size=3)
    # 对视图的振幅与相位求解
    TestView_A,TestView_P=image_processing.visual_sense(world,CoodinateX,CoodinateY, Heading, z=0.01, hfov=360, nmax=16, blur=False, kernel_size=3)
    return CoodinateX,CoodinateY,Heading,TestView_Img,TestView_A,TestView_P

"""同步记录收集视图信息"""
# 增加字典
def Update_RouteMemory(RouteMemory,CoodinateX,CoodinateY,world):
    # 使用时注意一次加一个记忆，如果需要大批量引入记得使用for循环
    # 本函数利用Construction_Memory构建RouteMemory所需元素后，再将这些元素更新值输入到字典中
    # 调用Construction_Memory函数 获取巢穴方向的视图

    CoodinateX,CoodinateY,Heading,Img_Collect=Construction_Memory(CoodinateX,CoodinateY,world)

    # 对字典进行更新

    RouteMemory['imgs'].append(Img_Collect)
    RouteMemory['pos'].append([CoodinateX,CoodinateY])
    RouteMemory['h'].append(Heading)

    # 返回更新后的字典

    return RouteMemory

# 视觉记忆处理，更新TestMemory并且将新增的记忆图像一个一个输入到蘑菇体中，返回更新好的所需记忆与对蘑菇体的改变
def VH_Visual_MemoryUpdate(VH_Instance,selected_points,TestMemory,world):

    # 使用for循环对记忆元素进行更新，此时要注意我们的TestMemory与师兄的route格式已经不一致了，后续进行实验要注意更改
    for point in selected_points:
        CoodinateX,CoodinateY=point
        Update_RouteMemory(TestMemory,CoodinateX,CoodinateY,world)

    # 将新图片逐个放入蘑菇体进行训练
    for point in selected_points:
        # 调用记忆构造函数，针对每个点构造记忆，必须注意这些记忆都是朝向Nest的
        CoodinateX,CoodinateY,Heading,Img_Collect=Construction_Memory(CoodinateX,CoodinateY,world)
        # 在此处要求蘑菇体进行学习！把值改变为True
        VH_Instance.mb.reward=True
        en = VH_Instance.mb.run(Img_Collect)
        print(en)
    # 训练结束后记得改变蘑菇体为读取记忆状态
    VH_Instance.mb.reward=False

    print(len(TestMemory['imgs']))

    return TestMemory


""" 完成Learning Walk的数据分析函数 Begin """

# 根据LW所得到的数据LWX与LWY计算轨迹围成面积与最远探索距离
def OneWalk_MaxDistance(LWX,LWY):
    Distance=np.sqrt(np.array(LWX)**2 + np.array(LWY)**2)
    MaxDistance=np.max(Distance)
    return MaxDistance

def OneWalk_TotleArea(LWX,LWY):
    # 将轨迹点转换为坐标对的列表
    points = list(zip(LWX, LWY))

    # 创建多边形对象
    polygon = Polygon(points)

    # 检查多边形是否有效，如果无效（即存在交叉），合并处理
    if not polygon.is_valid:
        polygon = unary_union(polygon)

    # 计算多边形的面积
    area = polygon.area
    return area

""" 完成Learning Walk的数据分析函数 End """

# 根据V_En得到视觉权重
def map_visual_en_to_gamma(Visual_En):
    # 检查输入是否在合理范围内
    if Visual_En < 0 or Visual_En > 500:
        raise ValueError("Visual_En should be in the range 0-500")
    
    # 线性映射：将 Visual_En 从 [0, 400] 映射到 [1, 0]
    gamma = 1 - (Visual_En / 400)
    gamma = max(0,gamma)
    
    return gamma
# 根据O_En得到嗅觉权重
def map_o_en_to_alpha(O_En):
    # 检查输入是否在合理范围内
    if O_En < 0 or O_En >= 1:
        raise ValueError("Input O_En should be in the range [0, 1) (inclusive of 0, exclusive of 1)")
    
    # 计算 alpha 值
    # alpha = 1 / (1 - O_En) - 1
    alpha=O_En
    
    return alpha
# 根据r计算得到Lpi
def calculate_Lpi(r, Proper_PI_Distance):
    # 计算斜率 k，已知过点 (0, pi) 和 (Proper_PI_Distance, pi/2)
    k = (np.pi / 2 - np.pi) / Proper_PI_Distance
    # 计算 Lpi
    Lpi = k * r + np.pi
    # 如果 Lpi 小于 0，则将其置为 0
    Lpi = max(Lpi, 0)
    return Lpi
# 视觉感受的函数
def calculate_Lv(V_En, Visual_Feeling_Mean, Visual_Feeling_Delta):
    Visual_Feeling_Delta = 2 * Visual_Feeling_Delta
    if V_En >= Visual_Feeling_Mean +  Visual_Feeling_Delta:
        return 0
    elif V_En <= Visual_Feeling_Mean - Visual_Feeling_Delta:
        return np.pi
    else:
        # 线性映射 V_En 在 (Visual_Feeling_Mean - Visual_Feeling_Delta, Visual_Feeling_Mean + Visual_Feeling_Delta) 内的值到 (pi, 0) 之间
        x1, x2 = Visual_Feeling_Mean - Visual_Feeling_Delta, Visual_Feeling_Mean + Visual_Feeling_Delta
        y1, y2 = np.pi, 0
        V_Factor = y1 + (V_En - x1) * (y2 - y1) / (x2 - x1)
        return V_Factor


# Core：写一个Learning Walk Class
class LearningWalkAgent(object):
    # 构造函数
    def __init__(self,world,CoordinateX,CoordinateY,Heading,VH_Instance,Route_Memory,Residue_Step):
        # 蚂蚁当前所处的世界
        self.world=world
        # 蚂蚁当前所在位置的横纵坐标
        self.CoordinateX=CoordinateX;self.CoordinateY=CoordinateY
        # 蚂蚁当前的头部朝向
        self.Heading=Heading
        # a dictionary with keys: ['imgs', 'h', 'pos']
        self.Route_Memory=dict(h=[],imgs=[],pos=[])

        # 蚂蚁的剩余步数
        self.Residue_Step=Residue_Step
        # 蚂蚁的视觉导航类Mushroom Body
        # set up parameters
        # PN神经元的数量
        num_pn = 81
        # KC神经元的数量
        num_kc = 40000
        # 学习率
        vh_learning_rate = 0.015
        # KC神经元的阈值
        vh_kc_thr = 0.04
        zm_n_max=16
        # create an instance
        self.VH_Instance_Homing = VisualHomingAgent(self.world,self.Route_Memory, None , zm_n_max, vh_learning_rate, vh_kc_thr, num_pn, num_kc)

        # 新建一个MB用于记忆ZM
        self.VH_ZM = VisualHomingAgent(self.world , self.Route_Memory , None , zm_n_max, 0.01 , 0.04 , 81 , 4000)

        # 视觉感受的均值与对波动的容忍程度
        self.Visual_Feeling_Mean = [400]
        self.Visual_Feeling_Delta = [0]
        self.Visula_EN_Recode = []

        # 构造一个记忆权重的Record
        self.Weight_Record=[]

        # 构造一个记录Offset的Record
        self.Offset_Record=[] ; self.Visual_Offset_Record=[]

    def Change_Position(self,NewX,NewY):
        self.CoordinateX=NewX;self.CoordinateY=NewY

    # 走一步小步的随机步
    # Von Mises 分布
    def Small_VonRandomStep(self,kappa=25,mu=0,StepUnit=1):
        # Self 代理自身
        # kappa 为集中参数
        # mu 分布均值
        # StepUnit: 单位步长
        Deta_H=vonmises.rvs(kappa, loc=mu)
        Deta_Angle=math.degrees(Deta_H)
        # 改变当前的头朝向
        self.Heading=self.Heading+Deta_H
        # 走一步
        self.CoordinateX=self.CoordinateX+StepUnit*np.cos(self.Heading)
        self.CoordinateY=self.CoordinateY+StepUnit*np.sin(self.Heading)
        self.Residue_Step=self.Residue_Step-1
        
    # 收集某一点处的视图构造ZM记忆输入到蘑菇体中
    def Get_ZM_VisualMemory(self):
        # 将更改项一同输入到蘑菇体中
        # 调用记忆构造函数，必须注意这些记忆都是朝向Nest的

        CoodinateX,CoodinateY,Heading,TestView_Img,TestView_A,TestView_P=Construction_ZM_Memory(self.CoordinateX,self.CoordinateY,self.world)

        # 在此处要求蘑菇体进行学习！把值改变为True

        self.VH_ZM.mb.reward=True
        en = self.VH_ZM.mb.run(TestView_A[:self.VH_ZM.zm_coeff_num])
        print(en)

        # 训练结束后记得改变蘑菇体为读取记忆状态
        self.VH_ZM.mb.reward=False

    # 计算Agent在所在点的视觉熟悉度与视觉系统返回的最小熟悉度方向
    # 可以用于原点选择下一次出巢穴的方向
    def Get_VisualFamiliarity(self,Scan_Interval=15):
        # 计算每个方向的熟悉度
        Familiarity_All=Get_AllHeading(self.VH_Instance_Homing,self.CoordinateX,self.CoordinateY,self.world,Scan_Interval)
        # 找出最小的En值与其对应的角
        min_Familiarity,min_angle=Get_BestHeading(Familiarity_All)
        # 将所有信息都返回
        return Familiarity_All,min_Familiarity,min_angle
    
    # 让Agent确定出巢穴的方向
    # 利用原则：选取巢穴时观看周围熟悉度最高的那个方向，也就是EN值的最小的方向
    # 也可以随机产生

    def Choose_LWDirection(self, Scan_Interval=15):
        """
        Randomly choose an initial Learning Walk direction with angle intervals of `Scan_Interval` degrees.
        Then set the agent's heading and shift its position slightly outward from the nest center.
        """
        import numpy as np

        # 从所有可选角度中随机选择一个
        possible_angles_deg = np.arange(0, 360, Scan_Interval)
        chosen_angle_deg = np.random.choice(possible_angles_deg)
        chosen_angle_rad = np.radians(chosen_angle_deg)

        # 设置头部朝向
        self.Heading = wrap_to_pi(chosen_angle_rad)

        # 将蚂蚁微移出巢穴中心
        self.CoordinateX = self.CoordinateX + 0.001 * np.cos(self.Heading)
        self.CoordinateY = self.CoordinateY + 0.001 * np.sin(self.Heading)


    # 以下几个函数都是需要灵活调整的

    """
    以下将在该类重新设计一些函数，用于建立一个新的模型（不局限于视觉场与嗅觉场）

    关于视觉方向的确定不依赖场论，依赖熟悉度之间相比得到的蚂蚁倾向

    在这个过程中Olf_Factor与V_Factor都需要重新书写一个函数
    """
    # 计算Agent在当前点的嗅觉熟悉度(利用Gauss分布) sigma越大衰减越慢
    def Olf_Factor(self,sigma=2):
        """计算给定位置的高斯浓度值。"""

        return np.exp(-(self.CoordinateX**2 + self.CoordinateY**2) / (2 * sigma**2))
       
    def ZM_Vh_HomeVector(self, Scan_Interval=60):
        """
        Use ZM familiarity values to simulate ant scanning behavior.

        Parameters:
        - Scan_Interval (int): Angle step for scanning in degrees (default: 60°)
        
        Returns:
        - min_familiarity (float): The minimum MBON_ZM value observed across directions
        - ArgV (float): The preferred heading direction (in radians) based on minimum familiarity
        """
        import numpy as np

        U = 0.1  # Step size for probing direction
        test_angles_deg = np.arange(0, 360, Scan_Interval)  # e.g., [0, 60, 120, ..., 300]
        
        familiarity_values = []

        for theta_deg in test_angles_deg:
            theta_rad = np.deg2rad(theta_deg)

            # Compute new test position at direction theta
            test_x = self.CoordinateX + U * np.cos(theta_rad)
            test_y = self.CoordinateY + U * np.sin(theta_rad)

            # Generate ZM memory at the new test location
            _, _, _, _, TestView_A, _ = Construction_ZM_Memory(test_x, test_y, self.world)
            MBON_ZM = self.VH_ZM.mb.run(TestView_A[:self.VH_ZM.zm_coeff_num])

            familiarity_values.append(MBON_ZM[0])  # Assuming familiarity is scalar

        # Find index of minimum familiarity (least familiar = candidate for homing)
        min_index = int(np.argmin(familiarity_values))
        min_familiarity = familiarity_values[min_index]

        # Convert index back to radian direction
        ArgV = np.deg2rad(test_angles_deg[min_index])

        return min_familiarity, ArgV

    # 计算嗅觉与PI对应的幅角
    def Olf_and_PI_HomeVector(self):
        """
        确定嗅觉与PI指示的Homing方向
        """
        # 直接计算ArgO，ArgPI
        Theta=Get_HeadingFun(self.CoordinateX,self.CoordinateY)
        # 将Theta移到[-pi,pi]
        Theta=wrap_to_pi(Theta)
        ArgO=Theta;ArgPI=Theta
        return ArgO,ArgPI
    
    # 计算嗅觉参数alpha与视觉参数gamma,对函数功能进行封装
    def calculate_alpha_and_gamma(self,V_En,O_En):
        # 先将Feeling转化为权重
        gamma=map_visual_en_to_gamma(V_En)
        alpha=map_o_en_to_alpha(O_En)
        return alpha,gamma

    # 综合以上信息得到Homing向量的组合
    def Blend_HomeVector(self,ArgO,ArgPI,ArgV,beta,gamma,alpha):
        """
        此函数用于得到一个蚂蚁预计的Homing方向
        注意:此函数是修改新问题的一个主函数,Alpha,beta,amma这三个参数应当有更合理的方向构造
        ArgO,ArgPI,ArgV:三个系统分别对应的Homing方向
        alpha,beta,gamma:嗅觉,PI与视觉分别对应的参数,alpha gamma需要通过计算,beta由自己确定
        """
        # 将 ArgO, ArgPI, ArgV 的幅角转化为向量的x, y分量
        x_O = alpha * np.cos(ArgO)  # 嗅觉向量的x分量
        y_O = alpha * np.sin(ArgO)  # 嗅觉向量的y分量

        x_PI = beta * np.cos(ArgPI)  # PI向量的x分量
        y_PI = beta * np.sin(ArgPI)  # PI向量的y分量

        x_V = gamma * np.cos(ArgV)  # 视觉向量的x分量
        y_V = gamma * np.sin(ArgV)  # 视觉向量的y分量

        # 合成向量的x, y分量
        x_H = x_O + x_PI + x_V
        y_H = y_O + y_PI + y_V

        # 计算合成向量的幅角
        ArgH = np.arctan2(y_H, x_H)
        # print("gamma=",round(gamma,4),"  alpha=",round(alpha,4),"  beta=",round(beta,4),"  DeviationArg=",round(wrap_to_pi(ArgH-ArgO),4))
        return ArgH  # 返回合成向量的幅角

    # 设计Learning Factor,将Residue Step直接封装在该函数中，作为反应学习状态的变量
    def L_Factor(self,V_En,Olf_En,Proper_PI_Distance,TotalStep,Step_minRate,Step_maxRate,alpha,beta,gamma):
        """
        本函数用于计算学习因子,也就是detaFi的部分
        1. En:视觉熟悉度
        2. Olf_En:嗅觉熟悉度
        3. TotalStep:一次Walk的预计步数
        4. Step_minRate,Step_maxRate:两个临界值,当剩余步数小于Step_minRate应当归巢,反之应当鼓励探索
        5. alpha,beta,gamma:嗅觉、PI、视觉的权重
        """
        # 根据预计步数与剩余步数计算活跃度函数Ac
        StepMax=TotalStep*Step_maxRate;StepMin=TotalStep*Step_minRate
        if self.Residue_Step>StepMax:
            Ac=1
        elif self.Residue_Step>StepMin:
            Ac=1
        elif self.Residue_Step>=0:
            Ac=0
        else:
            Ac=0
        
        # 求出离巢穴的距离r
        r=np.sqrt(self.CoordinateX**2+self.CoordinateY**2)

        # 对先求出Learning Factor中嗅觉指示的部分
        Lo=np.pi*Olf_En
        
        # 求出PI指示的部分
        Lpi = calculate_Lpi(r,Proper_PI_Distance)
        
        # # 求出视觉指示的部分
        Lv =  calculate_Lv(V_En,self.Visual_Feeling_Mean[-1],self.Visual_Feeling_Delta[-1])

        # 将所有导航工具参数归一化
        total=alpha+beta+gamma

        alpha/=total
        beta/=total
        gamma/=total

        # 保留若干位有效数字即可
        alpha=np.round(alpha,4);beta=np.round(beta,4);gamma=np.round(gamma,4)

        # 更新Record
        self.Weight_Record.append([alpha,beta,gamma])

        # 组合得到最终的Learning Factor
        L =min( Lo * alpha + Lpi * beta + Lv * gamma , np.pi)

        return L * Ac , Lv

    """
    以下在这些特殊函数组成蚂蚁的步型
    1. Walk0:没有特点的随机行走
    2. Walk1:不存储视图,但是会依据当前情况进行转向
    3. Pirouettes: 基于Walk1但是需要存储视图
    """
    # 走一步小步的随机步
    # Von Mises 分布
    def Walk0(self,kappa=25,mu=0,StepUnit=1):
        # Self 代理自身
        # kappa 为集中参数
        # mu 分布均值
        # StepUnit: 单位步长
        Deta_H=vonmises.rvs(kappa, loc=mu)
        # 改变当前的头朝向
        self.Heading=self.Heading+Deta_H
        # 走一步
        self.CoordinateX=self.CoordinateX+StepUnit*np.cos(self.Heading)
        self.CoordinateY=self.CoordinateY+StepUnit*np.sin(self.Heading)
        self.Residue_Step=self.Residue_Step-1

    def Walk1(self,Total_Step,Step_minRate,Step_maxRate,Scan_Interval,Proper_PI_Distance,
              beta,sigma,kappa,StepUnit):
        """
        不存储视图,但是会依据当前情况进行转向
        参数简介：
        - Total_Step:预计一次Walk的总步数
        - Step_minRate,Step_maxRate:两个临界值,当剩余步数小于Step_minRate应当归巢,反之应当鼓励探索
        - Scan_Interval:视觉导航中的扫描间隔
        - alpha,beta,gamma:嗅觉,PI与视觉分别对应的参数,其和应当为1
        - sigma:嗅觉参数,反应区域的气味浓郁程度
        - kappa:随机性参数
        - StepUnit:步长
        """
        # 求当前视觉的熟悉度与视觉归巢对应的ArgV

        V_En,ArgV=self.ZM_Vh_HomeVector()

        # V_MBON采用ZM记忆的值

        _ , _ , _ , _ ,TestView_A, _ = Construction_ZM_Memory(self.CoordinateX , self.CoordinateY , self.world)
        MBON_ZM = self.VH_ZM.mb.run(TestView_A[:self.VH_ZM.zm_coeff_num])

        # 求嗅觉与PI归巢对应的ArgO，ArgPI
        ArgO,ArgPI=self.Olf_and_PI_HomeVector()
        # 求嗅觉熟悉度对应的值
        Olf_En=self.Olf_Factor(sigma)

        # 求Alpha gamma
        alpha,gamma=self.calculate_alpha_and_gamma(MBON_ZM[0],Olf_En)

        # 求Beta,用于补偿嗅觉与视觉无法完成的工作
        beta=max(1-alpha-gamma,0)

        # 倘若执行Homing状态我们将增大PI的使用权重
        if self.Residue_Step/Total_Step < Step_minRate:
            beta=1


        # 组合蚂蚁当前状态的Homing Vector
        ArgHoming=self.Blend_HomeVector(ArgO,ArgPI,ArgV,beta,gamma,alpha)
        # 获取由当前状态得到的学习因子
        L_Factor,Lv=self.L_Factor(MBON_ZM[0],Olf_En,Proper_PI_Distance,Total_Step,Step_minRate,Step_maxRate,alpha,beta,gamma)

        self.Offset_Record.append(L_Factor)
        self.Visual_Offset_Record.append(Lv)

        # 随机误差
        Deta_H=vonmises.rvs(kappa,0)

        # 最终得到LearningVector
        ArgLearning=ArgHoming-L_Factor+Deta_H

        # 以此为方向进行一步行走
        self.Heading=ArgLearning
        self.CoordinateX=self.CoordinateX+StepUnit*np.cos(self.Heading)
        self.CoordinateY=self.CoordinateY+StepUnit*np.sin(self.Heading)
        self.Residue_Step=self.Residue_Step-1
        # 返回参数
        # 此时的返回格式只是保持调用格式的一致性
        return ArgLearning,V_En
    
    def Walk2(self,Total_Step,Step_minRate,Step_maxRate,Scan_Interval,Proper_PI_Distance,
              beta,sigma,kappa,StepUnit):
        """
        ！存储视图,并依照情况转向
        参数简介：
        - Total_Step:预计一次Walk的总步数
        - Step_minRate,Step_maxRate:两个临界值,当剩余步数小于Step_minRate应当归巢,反之应当鼓励探索
        - Scan_Interval:视觉导航中的扫描间隔
        - alpha,beta,gamma:嗅觉,PI与视觉分别对应的参数,其和应当为1
        - sigma:嗅觉参数,反应区域的气味浓郁程度
        - kappa:随机性参数
        - StepUnit:步长
        """
 
        # 求当前视觉的熟悉度与视觉归巢对应的ArgV
        V_En,ArgV=self.ZM_Vh_HomeVector()
        # 求嗅觉与PI归巢对应的ArgO，ArgPI
        ArgO,ArgPI=self.Olf_and_PI_HomeVector()
        # 求嗅觉熟悉度对应的值
        Olf_En=self.Olf_Factor(sigma)

        # V_MBON采用ZM记忆的值

        _ , _ , _ , _ ,TestView_A, _ = Construction_ZM_Memory(self.CoordinateX , self.CoordinateY , self.world)
        MBON_ZM = self.VH_ZM.mb.run(TestView_A[:self.VH_ZM.zm_coeff_num])

        # 记录ZM记忆
        self.Get_ZM_VisualMemory() 

        # 求Alpha gamma
        alpha,gamma=self.calculate_alpha_and_gamma(MBON_ZM[0],Olf_En)

        # 求Beta,用于补偿嗅觉与视觉无法完成的工作
        beta=max(1-alpha-gamma,0)

        # 倘若执行Homing状态我们将增大PI的使用权重
        if self.Residue_Step/Total_Step < Step_minRate:
            beta=1


        # 组合蚂蚁当前状态的Homing Vector
        ArgHoming=self.Blend_HomeVector(ArgO,ArgPI,ArgV,beta,gamma,alpha)
        # 获取由当前状态得到的学习因子
        L_Factor,Lv=self.L_Factor(MBON_ZM[0],Olf_En,Proper_PI_Distance,Total_Step,Step_minRate,Step_maxRate,alpha,beta,gamma)

        self.Offset_Record.append(L_Factor)
        self.Visual_Offset_Record.append(Lv)

        # 随机误差
        Deta_H=vonmises.rvs(kappa,0)

        # 最终得到LearningVector
        ArgLearning=ArgHoming-L_Factor+Deta_H

        # 以此为方向进行一步行走
        self.Heading=ArgLearning
        self.CoordinateX=self.CoordinateX+StepUnit*np.cos(self.Heading)
        self.CoordinateY=self.CoordinateY+StepUnit*np.sin(self.Heading)
        self.Residue_Step=self.Residue_Step-1
        # 返回参数
        # 此时的返回格式只是保持调用格式的一致性
        return ArgLearning,V_En
    
    """
    基于以上的三种不同的步法,重新写AllWalk函数
    """
    def AllWalk1(self, Count, StepUnit, Total_Step,beta,Proper_PI_Distance,kappa=25, eps=0.02,
                Step_minRate=0.5,Step_maxRate=0.5,Scan_Interval=15, sigma=2,random_rate=0.8):
        """
        模拟蚂蚁的单次行走行为。

        参数:
        - self: Agent对象
        - Count: 用于控制pirouettes频率的计数变量
        - StepUnit: 蚂蚁的步长
        - TotalStep: 总行走步数
        - kappa: 控制路径随机性的参数，默认为25。值越大路径越直，反之越弯曲
        - C: 嗅觉与视觉的权重系数
        - eps: 触发回家的误差距离
        - VMinThreshold_Value: 视觉熟悉度的最小阈值
        - VMaxThreshhold_value: 视觉熟悉度的最大阈值
        - sigma: 控制步幅变化的参数

        返回:
        - x_coords, y_coords: 每一步的x和y坐标
        - learning_x_coords, learning_y_coords: pirouettes的x和y坐标
        - Homing_coords: Homing向量
        - Learning_coords: Learning向量
        - Visual_coords: 视觉熟悉度
        - Heading_coords: 每一步的朝向角度
        """
        
        # 初始化存储变量
        x_coords, y_coords, Heading_coords = [], [], []
        learning_x_coords, learning_y_coords = [], []
        Learning_coords, Homing_coords, Visual_coords = [], [], []
        
        # 每一次大Walk都要初始化自己的Record
        self.Weight_Record=[]
        self.Offset_Record=[]
        self.Visual_Offset_Record=[]


        # 初始化剩余步数和初始位置
        self.Residue_Step = Total_Step
        self.Change_Position(0, 0)

        # 初始化一个计数器与特殊步的记录
        t=0;T=[];T_Scan=[]

        # 选择初始方向
        self.Choose_LWDirection(15)
        
        # 开始行走模拟
        while True:

            # 检查蚂蚁是否迷路，无法完成Homing
            # 若蚂蚁的消耗了两倍以上的剩余步数则认为蚂蚁无法完成Homing
            if self.Residue_Step < -1 * Total_Step:
                print('Ant fail to Homing')
                break

            # 记录当前位置和朝向
            x_coords.append(self.CoordinateX)
            y_coords.append(self.CoordinateY)
            Heading_coords.append(self.Heading)

            # 对当前朝向的视觉V_En进行存储

            # V_MBON采用ZM记忆的值

            _ , _ , _ , _ ,TestView_A, _ = Construction_ZM_Memory(self.CoordinateX , self.CoordinateY , self.world)
            MBON_ZM = self.VH_ZM.mb.run(TestView_A[:self.VH_ZM.zm_coeff_num])
            self.Visula_EN_Recode.append(MBON_ZM[0])


            # 实时计算视觉感受的均值与波动的容忍程度
            window_size = 20  # 定义窗口大小
            window_data = self.Visula_EN_Recode[-window_size:]  # 取最近的窗口数据
            mean_value = np.mean(window_data)
            delta_value = np.std(window_data)
            
            # 记录均值和波动容忍程度
            self.Visual_Feeling_Mean.append(mean_value)
            self.Visual_Feeling_Delta.append(delta_value)

            # 判断是否接近巢穴，是否结束探索
            if (self.CoordinateX**2 + self.CoordinateY**2 < eps**2 and
                self.Residue_Step < 0.2 * Total_Step):
                break
            

            # 如果满足计数条件，则进行大步走
            if self.Residue_Step % Count == 0:
                learning_x_coords.append(self.CoordinateX)
                learning_y_coords.append(self.CoordinateY)
                H = Get_HeadingFun(self.CoordinateX, self.CoordinateY)

                # 执行大步走（Walk2）
                L = self.Walk2(Total_Step,Step_minRate,Step_maxRate,Scan_Interval,Proper_PI_Distance,
                                beta,sigma,kappa,StepUnit)
                
                t=t+1;T.append(t);T_Scan.append(t)
                
                Learning_coords.append(L[0])
                Homing_coords.append(H)
                Visual_coords.append(L[1])
            else:
                # 否则以4:1的概率选择Walk0或Walk1
                if np.random.rand() < random_rate:
                    # 执行小步随机走（Walk0）
                    self.Walk0(kappa, 0, StepUnit)
                    t=t+1

                else:
                    # 执行小步有方向性走（Walk1）
                    self.Walk1(Total_Step,Step_minRate,Step_maxRate,Scan_Interval,Proper_PI_Distance,
                                beta,sigma,kappa,StepUnit)
                    t=t+1;T.append(t)

        
        # 输出剩余步数
        Actual_Step = Total_Step - self.Residue_Step
        print(f'实际走了 {Actual_Step} 步')
        
        # 数据扁平化处理
        x_coords = flatten_coords(x_coords)
        y_coords = flatten_coords(y_coords)
        learning_x_coords = flatten_coords(learning_x_coords)
        learning_y_coords = flatten_coords(learning_y_coords)
        Homing_coords = flatten_coords(Homing_coords)
        Learning_coords = flatten_coords(Learning_coords)
        Visual_coords = flatten_coords(Visual_coords)
        Heading_coords = flatten_coords(Heading_coords)
        
        # 返回行走数据
        return (x_coords, y_coords, learning_x_coords, learning_y_coords,
                Homing_coords, Learning_coords, Visual_coords, Heading_coords, Actual_Step,self.Weight_Record,T,T_Scan)
    