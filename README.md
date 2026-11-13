# capacity-changepoint

产能分析按设备型号核对产出速率、运行小时和温度范围，保留维护日期。

## 使用

运行需要 Python 3.11 或更新版本，没有第三方运行依赖。数据在本地处理，不请求外部接口。

```bash
python3 -m analysis_app summary
python3 -m analysis_app summary --input data/observations.csv --output output/summary.json

python3 -m analysis_app segments                 # 分段趋势 + 拐点稳定性 + 滚动验证
python3 -m analysis_app segments --asset A00 --no-bootstrap   # 单台设备、跳过重采样
python3 -m analysis_app.demo                     # 一条足够短的本地演示摘要
```

标准输出是完整 JSON，输出文件使用临时文件原子替换；失败退出码为 2，不覆盖既有结果。空数据集保留零计数与空分组，缺失数值不进入数值分母。数据指纹由按 observation_id 排序后的解析记录计算。

`segments` 的所有参数都有默认值，可用 `--help` 查看；`--no-bootstrap` 会明确把日期稳定性标为“未评估”而不是“证据不足”。

## 分段趋势分析

`segments` 回答的问题是：某台设备的产能下降，是温度造成的、维护造成的，还是趋势本身发生了转折。

- **按设备拟合，不拼接观测。** 每台设备单独构造设计矩阵；跨设备只共享“同 model 的温度系数先验强度”，从不共享原始观测，也不会把多台设备首尾相接；某个 model 只有一台设备时不施加组先验。
- **模型。** 分段折线，斜率为 b1 + 各段折线系数之和；温度项 `gamma * (temperature - 窗口内均值)`，`gamma` 在每个训练窗口内估计；`service_day` 只作为水平跳变项。
- **候选拐点约束。** 必须两侧都有足够有效运行小时和运行日；与维护日重合或在维护日之后的紧邻窗口内不计候选，因为维护恢复和趋势转折无法区分；两侧紧邻观测空档过大也不计候选。
- **停机日。** `hours <= 0` 不进入似然，也不把 throughput 当作 0。
- **离群。** 先在空模型上按“与左右邻居的偏离”标出单次尖峰，被标出的观测不参与段数选择；稳健拟合用固定尺度的 Huber 重加权。
- **段数选择。** 显式复杂度惩罚 `n*log(Huber损失/n) + correction*(p + 3K)*log(n)`：每个拐点除 1 个参数外再付出变点位置的成本；小样本修正有上限，避免短窗口选不出任何结构。零拐点仅以极小分差胜出时标记为“结构不确定”，不宣布无转折。
- **稳定性重采样。** 对选定结构做残差重抽样，每次重抽样重新选择段数；报告拐点日期的众数、稳定范围、命中率与“证据不足”。斜率区间只是回归系数区间，**不是**日期置信区间。
- **证据状态。** 每台设备输出 `持续退化` / `退化（日期稳定性未评估）` / `证据不足` / `无趋势转折` / `样本不足` / `无持续退化` 之一，并附 onset、确认检查、候选淘汰原因、每段斜率与维护跳变估计。

## 历史滚动验证

`rolling_validation` 对每台设备按可见日推进截点，只用 `visible_at <= cutoff` 且 `day <= cutoff` 的记录重新拟合并提出维护提示；维护事件也只取 cutoff 之前已发生的。提示提出后，用 cutoff 之后实际到达的数据检验“持续退化是否成立”，输出：

- 提示次数、可判定次数、确认次数、误报次数与提示精确率；
- 相对全量数据参考起点的检测延迟（中位数）；
- 无法判定的提示单独计数，不计入精确率分母；
- `future_leaks` 应为 0：未来维护日期不会进入过去模型。

## 验证与容器

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q analysis_app
docker build -t capacity-changepoint .
docker run --rm capacity-changepoint
docker run --rm capacity-changepoint python -m analysis_app segments --no-bootstrap
```

容器内也可执行 `python -m unittest discover -s tests -v`。镜像构建阶段取得 Python 镜像，产品启动、测试和演示均无需下载。小规模样例的完整单元测试和 `segments` 可以连续执行，不需要等待定时任务。

测试覆盖不规则采样、单次离群、维护恢复、纯台阶、完全无转折、停机日、跨设备隔离、未来数据泄漏与选项边界；`tests/series.py` 里是这些场景的合成序列。

## 代码与数据

- `analysis_app/io.py`：有类型的 CSV 读取、完整性校验及内容指纹。
- `analysis_app/schema.py`：字段类型、主体与聚合维度。
- `analysis_app/domain.py`：当前业务核算函数。
- `analysis_app/summary.py`：确定性的分组与数值口径。
- `analysis_app/linalg.py`：小规模线性代数（求解、求逆、共线检测）。
- `analysis_app/segments.py`：分段趋势、段数选择、稳定性重采样与滚动验证。
- `analysis_app/cli.py`：本地命令及原子输出。
- `analysis_app/demo.py`：短演示摘要。
- `docs/data.md`：数据契约。

演示数据全部为本地合成，不对应现实机构、设备或人员；不能用来推断真实业务效果。
