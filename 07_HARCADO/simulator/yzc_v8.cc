// 16 个基站在 5km*5km 的区域内均匀分布,即每个基站的坐标为 (i*1000, j*1000),其中 i,j=1,2,3,4
// 10 个 UE 的初始位置为整个区域的中心,即 (2500, 2500),速度大小为 3.0 m/s,方向为随机方向
// 每个 UE 每秒更新一次位置,速度和方向,更新后的位置为当前位置加上速度乘以更新间隔,即 x = x + v_x * t_step, y = y + v_y * t_step
// 如果 UE 超出区域范围,即 x < 0 或 x > 5000 或 y < 0 或 y > 5000,则将 UE 的位置设置为区域的边界位置,速度大小不变,方向改变
// 第一步更新后,根据 UE 所在位置,选择距离 UE 最近的基站作为服务基站
// 每步更新后,计算 UE 与所有基站之间的距离,然后计算该举例对应的 RSRP 值,基于 RSRP 值判断 TTT 和 HOM 这两个事件是否发生

#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/animation-interface.h"  // 用于生成 xml 文件
#include <cmath>  // 用于三角函数计算
#include <fstream>  // 用于文件操作
#include <map>
#include <string>
#include <ctime>  // 用于获取当前时间
#include "ns3/random-variable-stream.h"

using namespace ns3;

#define AREA_SIZE 5000.0  // 模拟区域大小为 5000 m x 5000 m
#define NUM_BASE_STATIONS 25  // 基站数量
#define BS_GRID_SIZE 5  // 基站网格大小, 5 x 5 网格
#define BS_SPACING 833  // 基站间距, 833.33 m
#define NUM_UE 10  // 用户设备数量
#define T_STEP 1.0  // 更新间隔为 1.0 s
#define TTT_THRESHOLD 3  // Time to Trigger, TTT 阈值, s
#define HOM_THRESHOLD 5  // Handover Margin, HOM 阈值, dB
#define BS_TX_POWER 46  // 基站发射功率, dBm, 46 dBm = 40 W
#define BS_FREQUENCY 3.5  // 基站工作频率, GHz, 3.5 GHz
const double NOISE_POWER_DBM = -100.0;  // 定义噪声功率（单位 dBm）
long long int step_counter = 0;  // 步数计数器
#define STEP_GAP_FOR_DIRECTION_CHANGE 5  // 改变用户设备的方向和速度的步数间隔

static double g_ueSpeed = 6.0;  // UE speed in m/s; override with --ueSpeed.
static uint32_t g_rngSeed = 1;  // Stable seed for reproducible paper runs.
static uint32_t g_rngRun = 1;
static std::string g_outputPrefix = "scratch/yzc_v8";

// Declare a map to hold ofstream objects for each UE
std::map<uint32_t, std::ofstream> ueFiles;

void OpenUeFiles(uint32_t numUes) {
    for (uint32_t i = 0; i < numUes; ++i) {
        std::ofstream outFile;
        std::string fileName = g_outputPrefix + "_ue_" + std::to_string(i) + ".csv";
        outFile.open(fileName, std::ios::out | std::ios::app);  // Open the file in append mode
        if (!outFile.is_open()) {
            std::cerr << "Failed to open file for UE " << i << std::endl;
            continue;
        }

        // Write the header to the CSV file
        if (outFile.tellp() == 0) {
            outFile << "Time,UEId,PosX,PosY,VelX,VelY,Direction,SourceBsId,SourceDis,SourceSinr,SourceRsrp,TargetBsId,TargetDis,TargetSinr,TargetRsrp,TTT,HOM" << std::endl;
        }
        ueFiles[i] = std::move(outFile);
    }
}

void CloseUeFiles() {
    for (auto& pair : ueFiles) {
        if (pair.second.is_open()) {
            pair.second.close();
        }
    }
}

// 定义基站类
class BaseStation {
    public:
        BaseStation(uint32_t id, double x, double y) : m_id(id), m_x(x), m_y(y) {}

        uint32_t GetId() const {
            return m_id;
        }

        double GetX() const {
            return m_x;
        }

        double GetY() const {
            return m_y;
        }

    private:
        uint32_t m_id;
        double m_x;
        double m_y;
};

// 定义用户设备类
class UE {
    public:
        UE(uint32_t id, double x, double y, uint32_t source = 0, uint32_t target = 0) : m_id(id), m_x(x), m_y(y), m_source_bs_id(source), m_target_bs_id(target) {}

        uint32_t GetId() const {
            return m_id;
        }

        double GetX() const {
            return m_x;
        }

        double GetY() const {
            return m_y;
        }

        void SetSourceBsId(uint32_t sourceBsId) {
            m_source_bs_id = sourceBsId;
        }

        uint32_t GetSourceBsId() const {
            return m_source_bs_id;
        }

        void SetTargetBsId(uint32_t targetBsId) {
            m_target_bs_id = targetBsId;
        }

        uint32_t GetTargetBsId() const {
            return m_target_bs_id;
        }

        void TttCounterDecrement() {
            ttt_counter--;
        }

        uint32_t GetTttCounter() const {
            return ttt_counter;
        }

        void SetTttCounter(uint32_t tttCounter) {
            ttt_counter = tttCounter;
        }

    private:
        uint32_t m_id;
        double m_x;
        double m_y;
        uint32_t m_source_bs_id;  // 服务基站 ID
        uint32_t m_target_bs_id;  // 目标基站 ID
        uint32_t ttt_counter;  // TTT 计数器
        uint32_t source_bs_rsrp_array[TTT_THRESHOLD * 2] = {0};  // 服务基站 RSRP 值列表
        uint32_t target_bs_rsrp_array[TTT_THRESHOLD * 2] = {0};  // 目标基站 RSRP 值列表
};

void UpdateNodePositions(NodeContainer ueNodes, std::vector<BaseStation> baseStations, std::vector<UE> ues) {
    // 输出仿真时间
    NS_LOG_UNCOND("Simulation Time: " << Simulator::Now().GetSeconds() << " s");


    // 从第二步开始,每步计算用户设备与基站之间的距离,然后将除服务基站外的最近基站作为目标基站
    // 分别计算到服务基站和目标基站的路径损耗,然后计算 RSRP 值
    // 当目标基站的 RSRP 值比服务基站的 RSRP 值高于 HOM 阈值时,并超过 TTT 时间时,则发生切换
    // 即用户设备从服务基站切换到目标基站,目标基站成为新的服务基站

    // 所有用户移动一步,根据用户设备的位置,选择除当前服务基站外距离用户设备最近的基站作为目标基站
    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        Ptr<Node> node = ueNodes.Get(i);  // NodeContainer::Get() 方法返回一个指向节点的智能指针(Ptr<Node> is a smart pointer to a Node object)
        if (node == nullptr) {
            NS_LOG_UNCOND("Node is nullptr at index " << i);
            continue;
        }

        Ptr<MobilityModel> mobilityModel = node->GetObject<MobilityModel>();  // Node::GetObject<MobilityModel>() 方法返回节点的移动模型
        if (mobilityModel == nullptr) {
            NS_LOG_UNCOND("MobilityModel is nullptr for Node " << node->GetId());
            continue;
        }

        Ptr<ConstantVelocityMobilityModel> cvmm = mobilityModel->GetObject<ConstantVelocityMobilityModel>();  // MobilityModel::GetObject<ConstantVelocityMobilityModel>() 方法返回 ConstantVelocityMobilityModel 对象
        if (cvmm == nullptr) {
            NS_LOG_UNCOND("ConstantVelocityMobilityModel is nullptr for Node " << node->GetId());
            continue;
        }

        Vector position = mobilityModel->GetPosition();  // 获取位置
        Vector velocity = cvmm->GetVelocity();  // 获取速度
        NS_LOG_UNCOND("User Equipment " << node->GetId() << " Current Position: (" << position << ") Current Velocity: (" << velocity << ")");
        position.x += velocity.x * T_STEP;  // 更新位置
        position.y += velocity.y * T_STEP;  // 更新位置

        // 为每个用户节点的每一步分配随机方向和速度
        Ptr<UniformRandomVariable> randomDirection = CreateObject<UniformRandomVariable>();  // 创建一个 UniformRandomVariable 对象
        randomDirection->SetAttribute("Min", DoubleValue(0.0));  // 设置随机方向的最小值
        randomDirection->SetAttribute("Max", DoubleValue(2 * M_PI));  // 设置随机方向的最大值

        // 如果用户设备超出区域范围,则将用户设备的位置设置为区域的边界位置,速度大小不变,方向改变
        if (position.x < 0 || position.x > AREA_SIZE || position.y < 0 || position.y > AREA_SIZE) {
            position.x = std::max(0.0, std::min(position.x, AREA_SIZE));
            position.y = std::max(0.0, std::min(position.y, AREA_SIZE));
            double direction = randomDirection->GetValue();  // 生成一个随机方向
            velocity.x = g_ueSpeed * std::cos(direction);  // 更新速度
            velocity.y = g_ueSpeed * std::sin(direction);  // 更新速度
        }
        mobilityModel->SetPosition(position);  // 设置位置
        cvmm->SetVelocity(velocity);  // 设置速度
        NS_LOG_UNCOND("                  moved to (" << mobilityModel->GetPosition() << ")");

        // 判断目标基站是否发生变化,如果发生变化,则更新目标基站,并重置 TTT 计数器
        Vector uePosition = mobilityModel->GetPosition();  // 获取用户设备的位置
        double minDistance = std::numeric_limits<double>::max();  // 初始化最小距离为 double 类型的最大值
        uint32_t sourceBsId = ues[i].GetSourceBsId();  // 获取用户设备的服务基站 ID
        uint32_t tempTargetBsId = 0;  // 初始化临时目标基站 ID 为 0
        for (uint32_t j = 0; j < NUM_BASE_STATIONS; j++) {
            BaseStation baseStation = baseStations[j];  // 获取基站
            if (baseStation.GetId() == sourceBsId) {
                continue;
            }
            double distance = std::sqrt(std::pow(uePosition.x - baseStation.GetX(), 2) + std::pow(uePosition.y - baseStation.GetY(), 2));  // 计算用户设备与基站之间的距禨
            if (distance < minDistance) {
                minDistance = distance;
                tempTargetBsId = baseStation.GetId();
            }
        }

        // 根据自由空间路径损耗公式分别计算到服务基站和目标基站的路径损耗,然后计算 RSRP 值
        double sourceDistance = std::sqrt(std::pow(uePosition.x - baseStations[sourceBsId].GetX(), 2) + std::pow(uePosition.y - baseStations[sourceBsId].GetY(), 2));  // 计算到服务基站的距离
        // double pathLossSource = 20 * std::log10(3e8 / (4 * M_PI)) - 20 * std::log10(BS_FREQUENCY) + 20 * std::log10(sourceDistance);  // 计算到服务基站的路径损耗,单位为 dB
        double pathLossSource = 36.7 * std::log10(sourceDistance) + 26 * std::log10(BS_FREQUENCY) + 22.7;  // 计算到服务基站的路径损耗,单位为 dB
        if (pathLossSource < 0) {
            pathLossSource = 0;
        }
        // double pathLossTarget = 20 * std::log10(3e8 / (4 * M_PI)) - 20 * std::log10(BS_FREQUENCY) + 20 * std::log10(minDistance);  // 计算到目标基站的路径损耗,单位为 dB
        double pathLossTarget = 36.7 * std::log10(minDistance) + 26 * std::log10(BS_FREQUENCY) + 22.7;  // 计算到目标基站的路径损耗,单位为 dB
        if (pathLossTarget < 0) {
            pathLossTarget = 0;
        }
        NS_LOG_UNCOND("User Equipment " << node->GetId() << " Path Loss Source: " << pathLossSource << " Path Loss Target: " << pathLossTarget);
        
        Ptr<NormalRandomVariable> normalRV = CreateObject<NormalRandomVariable>();
        normalRV->SetAttribute("Mean", DoubleValue(0.0));
        normalRV->SetAttribute("Variance", DoubleValue(2.0));

        double chi_sigma = normalRV->GetValue();  // 随机扰动项
        double rsrpSource = BS_TX_POWER - pathLossSource - chi_sigma;  // 计算服务基站的 RSRP 值,单位为 dBm，加入扰动项
        double rsrpTarget = BS_TX_POWER - pathLossTarget - chi_sigma;  // 计算目标基站的 RSRP 值,单位为 dBm
        NS_LOG_UNCOND("User Equipment " << node->GetId() << " RSRP Source: " << rsrpSource << " RSRP Target: " << rsrpTarget);

        if (tempTargetBsId != ues[i].GetTargetBsId()) {
            NS_LOG_UNCOND("User Equipment " << node->GetId() << " is changing target from Base Station " << ues[i].GetTargetBsId() << " to Base Station " << tempTargetBsId);
            ues[i].SetTargetBsId(tempTargetBsId);  // 设置用户设备的目标基站 ID 并重置 TTT 计数器
            ues[i].SetTttCounter(TTT_THRESHOLD);  // 重置 TTT 计数器
        }
        else if (ues[i].GetTargetBsId() != 0 && rsrpTarget - rsrpSource >= HOM_THRESHOLD) {
            NS_LOG_UNCOND("rsrpTarget " << rsrpTarget << " - rsrpSource " << rsrpSource << " >= HOM_THRESHOLD " << HOM_THRESHOLD);
            ues[i].TttCounterDecrement();  // TTT 计数器递减
            NS_LOG_UNCOND("User Equipment " << node->GetId() << " TTT Counter: " << ues[i].GetTttCounter());
            // std::this_thread::sleep_for(std::chrono::seconds(1));  // 暂停 1 秒
            if (ues[i].GetTttCounter() == 0) {
                // 计算到服务基站和目标基站的路径损耗,然后计算 RSRP 值
                // 当目标基站的 RSRP 值比服务基站的 RSRP 值高于 HOM 阈值时,并超过 TTT 时间时,则发生切换
                // 即用户设备从服务基站切换到目标基站,目标基站成为新的服务基站
                NS_LOG_UNCOND("User Equipment " << node->GetId() << " handover from Base Station " << ues[i].GetSourceBsId() << " to Base Station " << ues[i].GetTargetBsId());
                ues[i].SetSourceBsId(ues[i].GetTargetBsId());  // 设置用户设备的服务基站 ID
                ues[i].SetTargetBsId(0);  // 设置用户设备的目标基站 ID
                // ues[i].SetTttCounter(TTT_THRESHOLD);  // 重置 TTT 计数器
            }
        }
        else {
            ues[i].SetTttCounter(TTT_THRESHOLD);  // 重置 TTT 计数器
        }
    }

    // 根据用户设备的位置,选择除当前服务基站外距离用户设备最近的基站作为目标基站
    for (uint32_t i = 0; i < NUM_UE; i++) {
        Ptr<Node> ueNode = ueNodes.Get(i);  // NodeContainer::Get() 方法返回一个指向节点的智能指针(Ptr<Node> is a smart pointer to a Node object)
        Ptr<MobilityModel> ueMobilityModel = ueNode->GetObject<MobilityModel>();  // Node::GetObject<MobilityModel>() 方法返回节点的移动模型
        Vector uePosition = ueMobilityModel->GetPosition();  // 获取用户设备的位置
        double minDistance = std::numeric_limits<double>::max();  // 初始化最小距离为 double 类型的最大值
        uint32_t sourceBsId = ues[i].GetSourceBsId();  // 获取用户设备的服务基站 ID
        uint32_t targetBsId = 0;  // 初始化目标基站 ID 为 0
        for (uint32_t j = 0; j < NUM_BASE_STATIONS; j++) {
            BaseStation baseStation = baseStations[j];  // 获取基站
            if (baseStation.GetId() == sourceBsId) {
                continue;
            }
            double distance = std::sqrt(std::pow(uePosition.x - baseStation.GetX(), 2) + std::pow(uePosition.y - baseStation.GetY(), 2));  // 计算用户设备与基站之间的距离
            if (distance < minDistance) {
                minDistance = distance;
                targetBsId = baseStation.GetId();
            }
        }
        ues[i].SetTargetBsId(targetBsId);  // 设置用户设备的目标基站 ID
        NS_LOG_UNCOND("User Equipment " << ueNode->GetId() << " is targeted to Base Station " << targetBsId);

        // 将数据写入 csv 文件
        Ptr<ConstantVelocityMobilityModel> cvmm = ueMobilityModel->GetObject<ConstantVelocityMobilityModel>();  // MobilityModel::GetObject<ConstantVelocityMobilityModel>() 方法返回 ConstantVelocityMobilityModel 对象
        Vector ueVelocity = cvmm->GetVelocity();  // 获取用户设备的速度
        
        double ueDistanceToSourceBs = std::sqrt(std::pow(uePosition.x - baseStations[sourceBsId].GetX(), 2) + std::pow(uePosition.y - baseStations[sourceBsId].GetY(), 2));  // 计算用户设备到服务基站的距离
        // double pathLossSource = 20 * std::log10(3e8 / (4 * M_PI)) - 20 * std::log10(BS_FREQUENCY) + 20 * std::log10(ueDistanceToSourceBs);  // 计算用户设备到服务基站的路径损耗
        double pathLossSource = 36.7 * std::log10(ueDistanceToSourceBs) + 26 * std::log10(BS_FREQUENCY) + 22.7;  // 计算用户设备到服务基站的路径损耗
        
        // 在使用 normalRV 前声明并设置属性
        Ptr<NormalRandomVariable> normalRV = CreateObject<NormalRandomVariable>();
        normalRV->SetAttribute("Mean", DoubleValue(0.0));
        normalRV->SetAttribute("Variance", DoubleValue(2.0));
        double chi_sigma_source = normalRV->GetValue();  // 用于服务基站的随机扰动
        double ueRsrpToSourceBs = BS_TX_POWER - pathLossSource - chi_sigma_source;  // 计算用户设备到服务基站的 RSRP 值
        
        double ueDistanceToTargetBs = std::sqrt(std::pow(uePosition.x - baseStations[targetBsId].GetX(), 2) + std::pow(uePosition.y - baseStations[targetBsId].GetY(), 2));  // 计算用户设备到目标基站的距离
        // double pathLossTarget = 20 * std::log10(3e8 / (4 * M_PI)) - 20 * std::log10(BS_FREQUENCY) + 20 * std::log10(ueDistanceToTargetBs);  // 计算用户设备到目标基站的路径损耗
        double pathLossTarget = 36.7 * std::log10(ueDistanceToTargetBs) + 26 * std::log10(BS_FREQUENCY) + 22.7;  // 计算用户设备到目标基站的路径损耗
        double chi_sigma_target = normalRV->GetValue();  // 用于目标基站的随机扰动
        double ueRsrpToTargetBs = BS_TX_POWER - pathLossTarget - chi_sigma_target;   // 计算用户设备到目标基站的 RSRP 值
        
        // 转换 dBm 到 mW 的 lambda 表达式
        auto dBmToMw = [](double dBm) -> double {
            return std::pow(10, dBm / 10.0);
        };

        double noise_mW = dBmToMw(NOISE_POWER_DBM);

        // 计算服务基站处的干扰总功率（排除当前服务基站）
        double totalInterferenceSource_mW = 0.0;
        for (uint32_t j = 0; j < NUM_BASE_STATIONS; j++) {
            if (j == ues[i].GetSourceBsId()) continue;
            double distanceInterf = std::sqrt(std::pow(uePosition.x - baseStations[j].GetX(), 2) +
                                            std::pow(uePosition.y - baseStations[j].GetY(), 2));
            double pathLossInterf = 36.7 * std::log10(distanceInterf) + 26 * std::log10(BS_FREQUENCY) + 22.7;
            if (pathLossInterf < 0) { pathLossInterf = 0; }
            double chi_sigma_interf = normalRV->GetValue();
            double interf_dBm = BS_TX_POWER - pathLossInterf - chi_sigma_interf;
            totalInterferenceSource_mW += dBmToMw(interf_dBm);
        }
        double signalSource_mW = dBmToMw(ueRsrpToSourceBs);
        double sinrSource_linear = signalSource_mW / (totalInterferenceSource_mW + noise_mW);
        double ueSinrToSourceBs = 10 * std::log10(sinrSource_linear);

        // 计算目标基站处的干扰总功率（排除当前目标基站）
        double totalInterferenceTarget_mW = 0.0;
        for (uint32_t j = 0; j < NUM_BASE_STATIONS; j++) {
            if (j == ues[i].GetTargetBsId()) continue;
            double distanceInterf = std::sqrt(std::pow(uePosition.x - baseStations[j].GetX(), 2) +
                                            std::pow(uePosition.y - baseStations[j].GetY(), 2));
            double pathLossInterf = 36.7 * std::log10(distanceInterf) + 26 * std::log10(BS_FREQUENCY) + 22.7;
            if (pathLossInterf < 0) { pathLossInterf = 0; }
            double chi_sigma_interf = normalRV->GetValue();
            double interf_dBm = BS_TX_POWER - pathLossInterf - chi_sigma_interf;
            totalInterferenceTarget_mW += dBmToMw(interf_dBm);
        }
        double signalTarget_mW = dBmToMw(ueRsrpToTargetBs);
        double sinrTarget_linear = signalTarget_mW / (totalInterferenceTarget_mW + noise_mW);
        double ueSinrToTargetBs = 10 * std::log10(sinrTarget_linear);

        NS_LOG_UNCOND("User Equipment " << ueNode->GetId() 
            << " SINR Source: " << ueSinrToSourceBs 
            << " SINR Target: " << ueSinrToTargetBs); 


        ueFiles[i] << Simulator::Now().GetSeconds() << 
            "," << ueNode->GetId() << 
            "," << uePosition.x << 
            "," << uePosition.y << 
            "," << ueVelocity.x << 
            "," << ueVelocity.y << 
            "," << std::atan2(ueVelocity.y, ueVelocity.x) << 
            "," << ues[i].GetSourceBsId() <<
            "," << ueDistanceToSourceBs <<
            "," << ueSinrToSourceBs <<
            "," << ueRsrpToSourceBs << 
            "," << ues[i].GetTargetBsId() << 
            "," << ueDistanceToTargetBs <<
            "," << ueSinrToTargetBs <<
            "," << ueRsrpToTargetBs << 
            "," << ues[i].GetTttCounter() << 
            "," << HOM_THRESHOLD << std::endl; 
        NS_LOG_UNCOND("Data written to file for UE " << ueNode->GetId());
    }

    // 20250224 更改用户设备的方向和速度
    for (uint32_t i = 0; i < NUM_UE; i++)
    {
        if (step_counter % STEP_GAP_FOR_DIRECTION_CHANGE != 0)
        {
            break;
        }

        Ptr<Node> ueNode = ueNodes.Get(i);                                                                     // NodeContainer::Get() 方法返回一个指向节点的智能指针(Ptr<Node> is a smart pointer to a Node object)
        Ptr<MobilityModel> ueMobilityModel = ueNode->GetObject<MobilityModel>();                               // Node::GetObject<MobilityModel>() 方法返回节点的移动模型
        Ptr<ConstantVelocityMobilityModel> cvmm = ueMobilityModel->GetObject<ConstantVelocityMobilityModel>(); // MobilityModel::GetObject<ConstantVelocityMobilityModel>() 方法返回 ConstantVelocityMobilityModel 对象
        // Vector uePosition = ueMobilityModel->GetPosition();  // 获取用户设备的位置
        Vector ueVelocity = cvmm->GetVelocity(); // 获取用户设备的速度

        // 为每个用户节点的每一步分配随机方向和速度
        Ptr<UniformRandomVariable> randomDirection = CreateObject<UniformRandomVariable>(); // 创建一个 UniformRandomVariable 对象
        randomDirection->SetAttribute("Min", DoubleValue(0.0));                             // 设置随机方向的最小值
        randomDirection->SetAttribute("Max", DoubleValue(2 * M_PI));                        // 设置随机方向的最大值
        double direction = randomDirection->GetValue();                                     // 生成一个随机方向

        // 更新速度
        ueVelocity.x = g_ueSpeed * std::cos(direction); // 更新速度
        ueVelocity.y = g_ueSpeed * std::sin(direction); // 更新速度

        cvmm->SetVelocity(ueVelocity); // 设置速度
    }

    step_counter++;  // 步数计数器递增

    // 重新调度下一步更新
    Simulator::Schedule(Seconds(T_STEP), &UpdateNodePositions, ueNodes, baseStations, ues);
}

void VisualizeCoverageAndTrajectories(AnimationInterface &anim, NodeContainer &bsNodes, NodeContainer &ueNodes) {
    NS_LOG_UNCOND("Visualize Coverage and Trajectories!");

    // 设置基站覆盖范围为蓝色圆圈
    for (uint32_t i = 0; i < bsNodes.GetN(); i++) {
        Ptr<Node> bsNode = bsNodes.Get(i);
        anim.UpdateNodeDescription(bsNode, "BaseStation_" + std::to_string(bsNode->GetId()));
        anim.UpdateNodeColor(bsNode, 0, 0, 255); // 蓝色
        anim.UpdateNodeSize(bsNode, 500.0, 500.0); // 覆盖范围为500米
    }

    // 设置用户设备轨迹为红色
    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        Ptr<Node> ueNode = ueNodes.Get(i);
        anim.UpdateNodeDescription(ueNode, "User_" + std::to_string(ueNode->GetId()));
        anim.UpdateNodeColor(ueNode, 255, 0, 0); // 红色
    }
}

int main(int argc, char *argv[]) {
    CommandLine cmd;
    cmd.AddValue("ueSpeed", "UE speed in m/s", g_ueSpeed);
    cmd.AddValue("rngSeed", "ns-3 RNG seed", g_rngSeed);
    cmd.AddValue("rngRun", "ns-3 RNG run", g_rngRun);
    cmd.AddValue("outputPrefix", "CSV output prefix, without _ue_<id>.csv", g_outputPrefix);
    cmd.Parse(argc, argv);

    // Stable RNG contract for reproducible paper runs.
    RngSeedManager::SetSeed(g_rngSeed);
    RngSeedManager::SetRun(g_rngRun);

    NS_LOG_UNCOND("Initialize Base Stations!");
    // 初始化 16 个基站
    NodeContainer bsNodes;  // 创建一个 NodeContainer 对象
    bsNodes.Create(NUM_BASE_STATIONS);  // 创建 16 个节点
    NS_LOG_UNCOND(bsNodes.GetN() << " nodes created.");

    MobilityHelper bsMobility;  // 创建一个 MobilityHelper 对象
    bsMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");  // 设置移动模型为 ConstantPositionMobilityModel
    bsMobility.Install(bsNodes);  // 安装移动模型到节点上

    std::vector<BaseStation> baseStations;  // 创建一个 BaseStation 类型的 vector

    for (uint32_t i = 0; i < BS_GRID_SIZE; i++) {
        for (uint32_t j = 0; j < BS_GRID_SIZE; j++) {
            uint32_t index = i * BS_GRID_SIZE + j;
            Ptr<Node> node = bsNodes.Get(index);  // NodeContainer::Get() 方法返回一个指向节点的智能指针(Ptr<Node> is a smart pointer to a Node object)
            Ptr<MobilityModel> mobilityModel = node->GetObject<MobilityModel>();  // Node::GetObject<MobilityModel>() 方法返回节点的移动模型
            // 设置基站的位置
            double x = (i + 1) * BS_SPACING;
            double y = (j + 1) * BS_SPACING;
            mobilityModel->SetPosition(Vector(x, y, 0));
            NS_LOG_UNCOND("Base Station " << node->GetId() << " created at (" << mobilityModel->GetPosition() << ")");

            baseStations.push_back(BaseStation(node->GetId(), x, y));  // 将基站添加到 vector 中
        }
    }

    NS_LOG_UNCOND("Initialize User Equipments!");
    // 初始化 10 个用户设备,初始位置为整个区域的中心
    NodeContainer ueNodes;  // 创建一个 NodeContainer 对象
    ueNodes.Create(NUM_UE);  // 创建 10 个节点
    NS_LOG_UNCOND(ueNodes.GetN() << " nodes created.");

    MobilityHelper ueMobility;  // 创建一个 MobilityHelper 对象
    ueMobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");  // 设置移动模型为 ConstantVelocityMobilityModel
    ueMobility.Install(ueNodes);  // 安装移动模型到节点上

    // 为每个用户节点的每一步分配随机方向和速度
    Ptr<UniformRandomVariable> randomDirection = CreateObject<UniformRandomVariable>();  // 创建一个 UniformRandomVariable 对象
    randomDirection->SetAttribute("Min", DoubleValue(0.0));  // 设置随机方向的最小值
    randomDirection->SetAttribute("Max", DoubleValue(2 * M_PI));  // 设置随机方向的最大值

    std::vector<UE> ues;  // 创建一个 UE 类型的 vector

    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        Ptr<Node> node = ueNodes.Get(i);  // NodeContainer::Get() 方法返回一个指向节点的智能指针(Ptr<Node> is a smart pointer to a Node object)
        Ptr<MobilityModel> mobilityModel = node->GetObject<MobilityModel>();  // Node::GetObject<MobilityModel>() 方法返回节点的移动模型
        Ptr<ConstantVelocityMobilityModel> cvmm = mobilityModel->GetObject<ConstantVelocityMobilityModel>();  // MobilityModel::GetObject<ConstantVelocityMobilityModel>() 方法返回 ConstantVelocityMobilityModel 对象
        double x = AREA_SIZE / 2;
        double y = AREA_SIZE / 2;
        mobilityModel->SetPosition(Vector(x, y, 0));  // 设置用户设备的初始位置
        Vector velocity = cvmm->GetVelocity();  // 获取速度
        double direction = randomDirection->GetValue();  // 生成一个随机方向
        velocity.x = g_ueSpeed * std::cos(direction);  // 更新速度
        velocity.y = g_ueSpeed * std::sin(direction);  // 更新速度
        cvmm->SetVelocity(velocity);  // 设置速度
        NS_LOG_UNCOND("User Equipment " << node->GetId() << " random direction: " << direction << " random speed: " << node->GetObject<MobilityModel>()->GetObject<ConstantVelocityMobilityModel>()->GetVelocity());
        NS_LOG_UNCOND("                  created at (" << mobilityModel->GetPosition() << ")");  // NodeContainer::GetId() 返回的节点 ID 为全局唯一的,即与基站的 ID 不会重复
        NS_LOG_UNCOND("                  velocity: " << node->GetObject<MobilityModel>()->GetObject<ConstantVelocityMobilityModel>()->GetVelocity());

        ues.push_back(UE(i, AREA_SIZE / 2, AREA_SIZE / 2));  // 将用户设备添加到 vector 中
    }

    // 所有用户移动一步
    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        Ptr<Node> node = ueNodes.Get(i);  // NodeContainer::Get() 方法返回一个指向节点的智能指针(Ptr<Node> is a smart pointer to a Node object)
        Ptr<MobilityModel> mobilityModel = node->GetObject<MobilityModel>();  // Node::GetObject<MobilityModel>() 方法返回节点的移动模型
        Ptr<ConstantVelocityMobilityModel> cvmm = mobilityModel->GetObject<ConstantVelocityMobilityModel>();  // MobilityModel::GetObject<ConstantVelocityMobilityModel>() 方法返回 ConstantVelocityMobilityModel 对象
        Vector position = mobilityModel->GetPosition();  // 获取位置
        Vector velocity = cvmm->GetVelocity();  // 获取速度
        NS_LOG_UNCOND("User Equipment " << node->GetId() << " Current Position: (" << position << ") Current Velocity: (" << velocity << ")");
        position.x += velocity.x * T_STEP;  // 更新位置
        position.y += velocity.y * T_STEP;  // 更新位置

        // 为每个用户节点的每一步分配随机方向和速度
        Ptr<UniformRandomVariable> randomDirection = CreateObject<UniformRandomVariable>();  // 创建一个 UniformRandomVariable 对象
        randomDirection->SetAttribute("Min", DoubleValue(0.0));  // 设置随机方向的最小值
        randomDirection->SetAttribute("Max", DoubleValue(2 * M_PI));  // 设置随机方向的最大值

        // 如果用户设备超出区域范围,则将用户设备的位置设置为区域的边界位置,速度大小不变,方向改变
        if (position.x < 0 || position.x > AREA_SIZE || position.y < 0 || position.y > AREA_SIZE) {
            position.x = std::max(0.0, std::min(position.x, AREA_SIZE));
            position.y = std::max(0.0, std::min(position.y, AREA_SIZE));
            double direction = randomDirection->GetValue();  // 生成一个随机方向
            velocity.x = g_ueSpeed * std::cos(direction);  // 更新速度
            velocity.y = g_ueSpeed * std::sin(direction);  // 更新速度
        }
        mobilityModel->SetPosition(position);  // 设置位置
        cvmm->SetVelocity(velocity);  // 设置速度
        NS_LOG_UNCOND("                  moved to (" << mobilityModel->GetPosition() << ")");
    }

    // 根据用户设备的位置,选择距离用户设备最近的基站作为服务基站
    for (uint32_t i = 0; i < NUM_UE; i++) {
        Ptr<Node> ueNode = ueNodes.Get(i);  // NodeContainer::Get() 方法返回一个指向节点的智能指针(Ptr<Node> is a smart pointer to a Node object)
        Ptr<MobilityModel> ueMobilityModel = ueNode->GetObject<MobilityModel>();  // Node::GetObject<MobilityModel>() 方法返回节点的移动模型
        Vector uePosition = ueMobilityModel->GetPosition();  // 获取用户设备的位置
        double minDistance = std::numeric_limits<double>::max();  // 初始化最小距离为 double 类型的最大值
        uint32_t sourceBsId = 0;  // 初始化服务基站 ID 为 0
        for (uint32_t j = 0; j < NUM_BASE_STATIONS; j++) {
            BaseStation baseStation = baseStations[j];  // 获取基站
            double distance = std::sqrt(std::pow(uePosition.x - baseStation.GetX(), 2) + std::pow(uePosition.y - baseStation.GetY(), 2));  // 计算用户设备与基站之间的距禨
            if (distance < minDistance) {
                minDistance = distance;
                sourceBsId = baseStation.GetId();
            }
        }
        ues[i].SetSourceBsId(sourceBsId);  // 设置用户设备的服务基站 ID
        NS_LOG_UNCOND("User Equipment " << ueNode->GetId() << " is initially served by Base Station " << sourceBsId);
    }

    // 设置生成 xml 文件的文件名
    AnimationInterface anim("scratch/yzc_v8.xml");

    // 打开 csv 文件
    OpenUeFiles(NUM_UE);

    // 设置仿真结束时间
    Simulator::Stop(Seconds(17280));
    
    Simulator::Schedule(Seconds(T_STEP), &UpdateNodePositions, ueNodes, baseStations, ues);

    // 调用函数设置基站覆盖范围和用户轨迹颜色
    // VisualizeCoverageAndTrajectories(anim, bsNodes, ueNodes);

    // 将BS节点添加到动画中
    for (uint32_t i = 0; i < bsNodes.GetN(); i++) {
        Ptr<Node> bsNode = bsNodes.Get(i);  // NodeContainer::Get() 方法返回一个指向节点的智能指针(Ptr<Node> is a smart pointer to a Node object)
        if (bsNode == nullptr) continue;

        uint32_t row = i / BS_GRID_SIZE;
        uint32_t col = i % BS_GRID_SIZE;
        double x = (row + 1) * BS_SPACING;
        double y = (col + 1) * BS_SPACING;
        anim.SetConstantPosition(bsNode, x, y);
    }

    // 将UE节点添加到动画中
    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        Ptr<Node> ueNode = ueNodes.Get(i);  // NodeContainer::Get() 方法返回一个指向节点的智能指针(Ptr<Node> is a smart pointer to a Node object)
        if (ueNode == nullptr) {
            NS_LOG_UNCOND("ueNode is nullptr at index " << i);
            continue;
        }

        anim.SetConstantPosition(ueNode, 2500, 2500);  // 设置用户设备的位置
    }

    // 启动仿真
    NS_LOG_UNCOND("Start Simulation!");
    Simulator::Run();
    Simulator::Destroy();

    // 关闭 csv 文件
    CloseUeFiles();
}
