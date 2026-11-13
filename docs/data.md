# 数据契约

产能分析按设备型号核对产出速率、运行小时和温度范围，保留维护日期。

主测量 `throughput` 的单位为件每小时，主体字段是 `asset`，核算维度是 `model`。`observation_id` 是文件内唯一观测编号，主体允许重复观测。数值只接受有限值；空单元格解析为 None，不能直接当成零。额外列不进入当前核算。

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `observation_id` | str | 必填 |
| `asset` | str | 必填 |
| `model` | str | 必填 |
| `day` | int | 必填 |
| `throughput` | float | 必填 |
| `hours` | float | 必填 |
| `temperature` | float | 必填 |
| `service_day` | int | 必填 |
| `visible_at` | int | 必填 |

时间字段 day、week、tick、hour、minute、cycle、start_day、reported_day、known_at、visible_at、opened、observed 均以各字段名所述的相对时间单位表示整数；不同字段并不必然代表同一种时间。缺少的登记码与浓度等信息保持为空。示例只表达产品数据结构，不能代替实际调查资料。

## 业务字段语义

throughput 是运行时每小时产能，hours 为当天实际运行小时。停机日 hours=0，但并不将 throughput 记为零。temperature 为摄氏度，service_day=-1 表示无维护记录；visible_at 为观测可用日。


## 分段趋势分析的口径

`summary` 与 `segments` 共用同一份解析记录，但 `segments` 对时间字段有额外约定：

- `day` 是分析用的相对天；每台设备单独拟合，不与其他设备拼接。
- `hours > 0` 才进入趋势似然。`hours = 0` 表示停机，`throughput` 在该日没有测量意义，既不当作零产能，也不参与加权。
- `temperature` 的系数在每个训练窗口内由该窗口数据估计；同 `model` 的其他设备只用于形成经验贝叶斯先验，不共享观测。
- `service_day = -1` 表示没有维护记录。其余值视为已知维护日，只作为水平跳变；维护日及其紧邻窗口不作为趋势拐点候选。
- `visible_at` 是观测可用日。历史滚动验证中，过去模型只使用 `visible_at <= cutoff` 且 `day <= cutoff` 的记录；未来维护日期不会进入过去模型。
- 斜率区间是回归系数区间；拐点日期的不确定范围来自重采样分布，二者不可互换。
