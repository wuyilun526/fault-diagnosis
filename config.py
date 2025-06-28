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
    # 数据库相关故障
    {
        "symptom": "数据库响应延迟突增至5s，CPU使用率90%",
        "category": "数据库死锁",
        "solution": "1. 检查锁等待链\n2. 终止阻塞事务\n3. 优化索引",
        "source": "production_incident_001",
        "timestamp": "2024-01-15T10:30:00"
    },
    {
        "symptom": "数据库连接池耗尽，大量连接超时错误",
        "category": "数据库连接池耗尽",
        "solution": "1. 增加连接池大小\n2. 检查连接泄漏\n3. 优化SQL查询",
        "source": "production_incident_002",
        "timestamp": "2024-01-20T14:15:00"
    },
    {
        "symptom": "数据库磁盘空间不足，写入操作失败",
        "category": "数据库磁盘空间不足",
        "solution": "1. 清理日志文件\n2. 归档历史数据\n3. 扩容磁盘空间",
        "source": "production_incident_003",
        "timestamp": "2024-01-25T09:45:00"
    },
    
    # 内存相关故障
    {
        "symptom": "内存占用每小时增长2%，Full GC频繁",
        "category": "内存泄漏",
        "solution": "1. 生成堆转储\n2. 分析内存对象\n3. 修复引用链",
        "source": "production_incident_004",
        "timestamp": "2024-02-01T16:20:00"
    },
    {
        "symptom": "JVM堆内存使用率超过95%，频繁GC",
        "category": "内存不足",
        "solution": "1. 增加堆内存大小\n2. 优化内存配置\n3. 检查内存泄漏",
        "source": "production_incident_005",
        "timestamp": "2024-02-05T11:30:00"
    },
    
    # 配置相关故障
    {
        "symptom": "服务接口成功率从99.9%骤降至60%，错误码ConfigNotFount",
        "category": "配置错误",
        "solution": "1. 检查最近部署的配置\n2. 回滚最新配置\n3. 验证配置加载流程",
        "source": "production_incident_006",
        "timestamp": "2024-02-10T13:45:00"
    },
    {
        "symptom": "服务启动失败，配置文件格式错误",
        "category": "配置文件格式错误",
        "solution": "1. 检查配置文件语法\n2. 验证配置项格式\n3. 重新部署配置",
        "source": "production_incident_007",
        "timestamp": "2024-02-15T08:20:00"
    },
    
    # 网络相关故障
    {
        "symptom": "接口流量超过阈值95%，TCP重传率增加300%",
        "category": "网络阻塞",
        "solution": "1. 扩容带宽\n2. 优化流量调度\n3. 分析异常流量源",
        "source": "production_incident_008",
        "timestamp": "2024-02-20T12:30:00"
    },
    {
        "symptom": "服务间调用超时，网络延迟异常",
        "category": "网络延迟异常",
        "solution": "1. 检查网络链路\n2. 优化网络配置\n3. 增加超时时间",
        "source": "production_incident_009",
        "timestamp": "2024-02-25T15:45:00"
    },
    {
        "symptom": "DNS解析失败，服务无法访问外部资源",
        "category": "DNS解析故障",
        "solution": "1. 检查DNS配置\n2. 切换DNS服务器\n3. 添加本地hosts",
        "source": "production_incident_010",
        "timestamp": "2024-03-01T10:15:00"
    },
    
    # 依赖链故障
    {
        "symptom": "接口成功率下降，下游依赖的接口成功率下降，下游依赖接口服务的DB慢查询增加",
        "category": "接口依赖问题",
        "solution": "1. 检查下游依赖接口\n2. 优化DB查询\n3. 增加缓存",
        "source": "production_incident_011",
        "timestamp": "2024-03-05T11:30:00"
    },
    {
        "symptom": "接口QPS下降，接口成功率不变，上游QPS也存在跌落",
        "category": "上游QPS跌落异常",
        "solution": "1. 通知上游服务的SRE，确认是否存在流量控制",
        "source": "production_incident_012",
        "timestamp": "2024-03-10T13:45:00"
    },
    
    # 应用层故障
    {
        "symptom": "服务线程池耗尽，大量请求排队等待",
        "category": "线程池耗尽",
        "solution": "1. 增加线程池大小\n2. 优化任务处理逻辑\n3. 添加熔断机制",
        "source": "production_incident_013",
        "timestamp": "2024-03-15T09:20:00"
    },
    {
        "symptom": "服务CPU使用率100%，响应时间急剧增加",
        "category": "CPU过载",
        "solution": "1. 扩容CPU资源\n2. 优化算法逻辑\n3. 添加负载均衡",
        "source": "production_incident_014",
        "timestamp": "2024-03-20T14:30:00"
    },
    {
        "symptom": "服务频繁重启，日志显示OOM错误",
        "category": "内存溢出",
        "solution": "1. 增加内存限制\n2. 优化内存使用\n3. 添加监控告警",
        "source": "production_incident_015",
        "timestamp": "2024-03-25T16:45:00"
    },
    
    # 存储相关故障
    {
        "symptom": "磁盘IO使用率100%，读写操作超时",
        "category": "磁盘IO瓶颈",
        "solution": "1. 扩容磁盘IOPS\n2. 优化IO操作\n3. 使用SSD存储",
        "source": "production_incident_016",
        "timestamp": "2024-04-01T11:20:00"
    },
    {
        "symptom": "文件系统只读，无法写入新数据",
        "category": "文件系统故障",
        "solution": "1. 检查磁盘健康状态\n2. 修复文件系统\n3. 恢复数据备份",
        "source": "production_incident_017",
        "timestamp": "2024-04-05T13:15:00"
    },
    
    # 缓存相关故障
    {
        "symptom": "Redis连接超时，缓存命中率下降",
        "category": "缓存连接故障",
        "solution": "1. 检查Redis服务状态\n2. 优化连接池配置\n3. 添加缓存降级",
        "source": "production_incident_018",
        "timestamp": "2024-04-10T10:30:00"
    },
    {
        "symptom": "缓存数据不一致，业务逻辑异常",
        "category": "缓存数据不一致",
        "solution": "1. 清理缓存数据\n2. 检查缓存更新逻辑\n3. 添加数据校验",
        "source": "production_incident_019",
        "timestamp": "2024-04-15T15:20:00"
    },
    
    # 安全相关故障
    {
        "symptom": "大量异常登录尝试，账户被锁定",
        "category": "安全攻击",
        "solution": "1. 检查安全日志\n2. 加强访问控制\n3. 更新安全策略",
        "source": "production_incident_020",
        "timestamp": "2024-04-20T12:45:00"
    },
    {
        "symptom": "SSL证书过期，HTTPS访问失败",
        "category": "证书过期",
        "solution": "1. 更新SSL证书\n2. 检查证书配置\n3. 监控证书有效期",
        "source": "production_incident_021",
        "timestamp": "2024-04-25T09:30:00"
    },
    
    # 部署相关故障
    {
        "symptom": "新版本部署后服务启动失败，回滚后正常",
        "category": "部署失败",
        "solution": "1. 检查部署配置\n2. 验证依赖版本\n3. 完善部署流程",
        "source": "production_incident_022",
        "timestamp": "2024-05-01T14:20:00"
    },
    {
        "symptom": "灰度发布后部分用户访问异常",
        "category": "灰度发布问题",
        "solution": "1. 检查路由配置\n2. 验证版本兼容性\n3. 调整灰度策略",
        "source": "production_incident_023",
        "timestamp": "2024-05-05T11:15:00"
    },
    
    # 监控告警故障
    {
        "symptom": "监控系统告警延迟，故障发现不及时",
        "category": "监控告警延迟",
        "solution": "1. 检查监控系统状态\n2. 优化告警规则\n3. 增加告警渠道",
        "source": "production_incident_024",
        "timestamp": "2024-05-10T16:30:00"
    },
    {
        "symptom": "日志收集异常，无法查看最新日志",
        "category": "日志收集故障",
        "solution": "1. 检查日志收集服务\n2. 验证存储空间\n3. 重启收集进程",
        "source": "production_incident_025",
        "timestamp": "2024-05-15T13:45:00"
    },
    {
        "symptom": "供应链接口成功率下跌, QPS下跌, 依赖下游接口无异常",
        "category": "该接口前端发布异常",
        "solution": "1. 和前端确认是否发布\n2. 回滚发布",
        "source": "production_incident_026",
        "timestamp": "2024-05-15T13:45:00"
    },
    {
        "symptom": "支付接口成功率下跌, QPS下跌, 依赖下游接口无异常, 依赖的DB慢查询增加",
        "category": "该接口依赖的DB异常",
        "solution": "1. 确认DB异常\n2. 优化DB查询",
        "source": "production_incident_027",
        "timestamp": "2024-05-15T13:45:00"
    }
]

# ===================== 测试用例 =====================
TEST_CASES = [
    # 数据库相关测试
    {
        "desc": "订单服务响应超时率超过60%",
        "metrics": "DB连接池活跃连接100%，SQL执行平均耗时2.5s",
        "logs": "发现大量'Lock wait timeout'错误日志",
        "expected_category": "数据库死锁"
    },
    {
        "desc": "用户服务无法连接数据库",
        "metrics": "数据库连接池使用率100%，连接获取超时",
        "logs": "Connection pool exhausted错误",
        "expected_category": "数据库连接池耗尽"
    },
    {
        "desc": "数据库写入操作失败",
        "metrics": "磁盘使用率98%，数据库写入错误",
        "logs": "No space left on device错误",
        "expected_category": "数据库磁盘空间不足"
    },
    
    # 内存相关测试
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
        "desc": "服务频繁OOM重启",
        "metrics": "堆内存使用率100%，频繁GC",
        "logs": "java.lang.OutOfMemoryError错误",
        "expected_category": "内存溢出"
    },
    
    # 网络相关测试
    {
        "desc": "支付接口响应缓慢",
        "metrics": "网络延迟从10ms增加到500ms，TCP重传率增加",
        "logs": "Connection timeout错误",
        "expected_category": "网络延迟异常"
    },
    {
        "desc": "服务无法访问外部API",
        "metrics": "DNS查询失败率100%",
        "logs": "Name resolution failed错误",
        "expected_category": "DNS解析故障"
    },
    
    # 依赖链测试
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
    },
    
    # 应用层测试
    {
        "desc": "订单处理服务响应超时",
        "metrics": "线程池使用率100%，请求队列积压",
        "logs": "Thread pool exhausted错误",
        "expected_category": "线程池耗尽"
    },
    {
        "desc": "搜索服务CPU使用率异常",
        "metrics": "CPU使用率100%，响应时间超过10s",
        "logs": "CPU密集型操作日志",
        "expected_category": "CPU过载"
    },
    
    # 存储相关测试
    {
        "desc": "文件上传服务异常",
        "metrics": "磁盘IO使用率100%，文件写入超时",
        "logs": "Disk I/O error错误",
        "expected_category": "磁盘IO瓶颈"
    },
    {
        "desc": "日志写入失败",
        "metrics": "文件系统只读，无法写入日志",
        "logs": "Read-only file system错误",
        "expected_category": "文件系统故障"
    },
    
    # 缓存相关测试
    {
        "desc": "用户会话丢失",
        "metrics": "Redis连接超时，缓存命中率下降",
        "logs": "Redis connection timeout错误",
        "expected_category": "缓存连接故障"
    },
    {
        "desc": "商品价格显示错误",
        "metrics": "缓存数据与数据库不一致",
        "logs": "Cache data mismatch错误",
        "expected_category": "缓存数据不一致"
    },
    
    # 安全相关测试
    {
        "desc": "用户账户被异常锁定",
        "metrics": "异常登录尝试次数激增",
        "logs": "Multiple failed login attempts",
        "expected_category": "安全攻击"
    },
    {
        "desc": "HTTPS访问失败",
        "metrics": "SSL证书验证失败",
        "logs": "SSL certificate expired错误",
        "expected_category": "证书过期"
    },
    
    # 部署相关测试
    {
        "desc": "新版本部署后服务无法启动",
        "metrics": "服务启动失败，健康检查不通过",
        "logs": "Service startup failed错误",
        "expected_category": "部署失败"
    },
    {
        "desc": "灰度发布后部分用户访问异常",
        "metrics": "部分用户请求路由到错误版本",
        "logs": "Version routing error错误",
        "expected_category": "灰度发布问题"
    },
    
    # 监控告警测试
    {
        "desc": "故障发现延迟",
        "metrics": "监控告警延迟超过30分钟",
        "logs": "Alert system lag错误",
        "expected_category": "监控告警延迟"
    },
    {
        "desc": "无法查看最新日志",
        "metrics": "日志收集延迟，最新日志缺失",
        "logs": "Log collection failed错误",
        "expected_category": "日志收集故障"
    },
    {
        "desc": "供应链接口A成功率下跌",
        "metrics": "接口A的QPS下跌, 下游依赖接口没有指标异常",
        "logs": "",
        "expected_category": ""
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