# -*- coding: utf-8 -*-
"""
故障诊断系统配置文件
"""

# ===================== 系统配置 =====================
SYSTEM_CONFIG = {
    "ali_api_key": None,  # 从环境变量获取
    "llm_model": "qwen-turbo",  # 可选qwen-turbo/qwen-plus/qwen-max
    "chunk_size": 500,  # 文档分块大小
    "chunk_overlap": 50,  # 分块重叠大小
    "top_k": 3,  # 检索返回文档数量
}

# ===================== 历史故障数据 =====================
HISTORICAL_DATA = [
    {
        "symptom": "数据库响应延迟突增至5s，CPU使用率90%",
        "category": "数据库死锁",
        "solution": "1. 检查锁等待链\n2. 终止阻塞事务\n3. 优化索引",
        "source": "production_incident_001",
        "timestamp": "2024-01-15T10:30:00"
    },
    {
        "symptom": "内存占用每小时增长2%，Full GC频繁",
        "category": "内存泄漏",
        "solution": "1. 生成堆转储\n2. 分析内存对象\n3. 修复引用链",
        "source": "production_incident_002",
        "timestamp": "2024-01-20T14:15:00"
    },
    {
        "symptom": "服务接口成功率从99.9%骤降至60%，错误码ConfigNotFount",
        "category": "配置错误",
        "solution": "1. 检查最近部署的配置\n2. 回滚最新配置\n3. 验证配置加载流程",
        "source": "production_incident_003",
        "timestamp": "2024-01-25T09:45:00"
    },
    {
        "symptom": "接口流量超过阈值95%，TCP重传率增加300%",
        "category": "网络阻塞",
        "solution": "1. 扩容带宽\n2. 优化流量调度\n3. 分析异常流量源",
        "source": "production_incident_004",
        "timestamp": "2024-02-01T16:20:00"
    },
    {
        "symptom": "接口成功率下降，下游依赖的接口成功率下降，下游依赖接口服务的DB慢查询增加",
        "category": "接口依赖问题",
        "solution": "1. 检查下游依赖接口\n2. 优化DB查询\n3. 增加缓存",
        "source": "production_incident_005",
        "timestamp": "2024-02-05T11:30:00"
    },
    {
        "symptom": "接口QPS下降，接口成功率不变，上游QPS也存在跌落",
        "category": "上游QPS跌落异常",
        "solution": "1. 通知上游服务的SRE，确认是否存在流量控制",
        "source": "production_incident_006",
        "timestamp": "2024-02-10T13:45:00"
    }
]

# ===================== 测试用例 =====================
TEST_CASES = [
    {
        "desc": "订单服务响应超时率超过60%",
        "metrics": "DB连接池活跃连接100%，SQL执行平均耗时2.5s",
        "logs": "发现大量'Lock wait timeout'错误日志",
        "expected_category": "数据库死锁"
    },
    {
        "desc": "用户服务API失败率突增",
        "metrics": "最近部署后失败率从0.1%升至40%",
        "logs": "出现大量'ConfigNotFount'错误",
        "expected_category": "配置错误"
    },
    {
        "desc": "服务内存使用持续增长",
        "metrics": "JVM内存占用每小时增长3%，GC频率增加",
        "logs": "观察到频繁的Full GC日志",
        "expected_category": "内存泄漏"
    },
    {
        "desc": "当前告警接口A存在QPS下降，接口A的成功率无异常，接口A上游接口B存在QPS跌落，接口B上游接口C存在QPS跌落",
        "metrics": "上游接口QPS持续下降，无明显的成功率异常",
        "logs": "上游服务可能存在流量控制或限流",
        "expected_category": "上游QPS跌落异常"
    },
    {
        "desc": "接口A成功率下降，下游依赖的接口B成功率下降，接口B依赖的下游接口C成功率跌落，接口C服务的DB慢查询增加",
        "metrics": "依赖链上多个接口成功率下降，底层DB查询性能异常",
        "logs": "DB慢查询日志增加，依赖链传递故障",
        "expected_category": "接口依赖问题"
    }
]

# ===================== 配置验证函数 =====================
def validate_config():
    """验证配置的完整性"""
    errors = []
    
    # 检查系统配置
    required_fields = ["llm_model", "chunk_size", "chunk_overlap", "top_k"]
    for field in required_fields:
        if field not in SYSTEM_CONFIG:
            errors.append(f"缺少系统配置字段: {field}")
    
    # 检查历史数据
    if not HISTORICAL_DATA:
        errors.append("历史故障数据不能为空")
    else:
        for i, record in enumerate(HISTORICAL_DATA):
            required_fields = ["symptom", "category", "solution"]
            for field in required_fields:
                if field not in record:
                    errors.append(f"历史数据记录 {i} 缺少字段: {field}")
    
    # 检查测试用例
    if not TEST_CASES:
        errors.append("测试用例不能为空")
    else:
        for i, case in enumerate(TEST_CASES):
            required_fields = ["desc", "metrics", "logs"]
            for field in required_fields:
                if field not in case:
                    errors.append(f"测试用例 {i} 缺少字段: {field}")
    
    if errors:
        raise ValueError(f"配置验证失败:\n" + "\n".join(errors))
    
    print("配置验证通过")
    return True 