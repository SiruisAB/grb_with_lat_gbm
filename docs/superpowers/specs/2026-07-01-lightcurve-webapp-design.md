# Streamlit 光变曲线第二页面设计

## 背景

当前 `streamlit run /home/mxr/lee/gbmtest/grb_project/web_app.py` 只提供单次分析入口，虽然后端 `project.py` 已经在分析完成后调用 `lightcurves.py` 生成光变图，但前端没有一个独立、可配置、可预览的光变页面。用户希望在同一个 Streamlit 应用下增加第二个页面，专门用于绘制光变曲线，并且要求前后端一致：前端能配置的参数，后端必须能够读取、记录并回看。

## 目标

1. 在 `web_app.py` 中增加第二个页面，用于光变曲线配置与生成。
2. 前端页面直接复用 `lightcurves.py` 的绘图入口，避免重复实现绘图逻辑。
3. 保证前后端使用同一套参数来源：
   - 页面默认值来自目录表、LAT 表、项目配置。
   - 页面提交时的参数会传入后端。
   - 后端会把实际使用的光变参数写入结果目录，便于查看与追踪。
4. 保持当前单次分析页不受影响。

## 非目标

1. 不重写 `lightcurves.py` 的绘图算法。
2. 不拆分成独立多页应用文件结构。
3. 不增加新的任务队列或异步执行机制。
4. 不引入与光变绘图无关的新分析功能。

## 总体方案

采用“页内路由”方式，在同一个 `web_app.py` 中通过侧边栏或顶部切换实现两个页面：

- 页面 1：单次分析
- 页面 2：光变曲线

两个页面共享：

- 目录表加载逻辑
- LAT 目录加载逻辑
- 默认 GRB 目标选择逻辑
- `GRBProjectConfig` / `GRBRunOverrides` 的参数体系

页面 2 直接调用 `plot_gbm_lat_lightcurve_figure(...)`，并将输出文件保存到与后端分析结果一致的目录结构下。

## 关键设计原则

### 1. 前后端参数同源

前端页面所有可编辑字段必须映射到后端可见字段，不能只存在于 UI 状态中。设计上分两层：

- **UI 层**：Streamlit 采集输入。
- **后端层**：`GRBProjectConfig` / `GRBRunOverrides` 接收这些输入，并在运行时写入结果元数据。

### 2. 后端可回看

为保证“前端可以设置的在后端中可以查看得到”，需要把页面输入完整记录到结果目录中的 metadata 文件里，建议采用以下方式：

- 扩展 `run_metadata.json`，写入 `lightcurve_request` 或同名结构。
- 记录实际绘图使用的参数、自动推断值、默认回退值。
- 记录最终输出图片路径。

### 3. 绘图逻辑单一来源

页面 2 不独立实现光变图逻辑，而是直接复用 `lightcurves.py` 中的 `plot_gbm_lat_lightcurve_figure(...)`。

## 页面 2 功能设计

### 输入区

页面 2 提供以下输入项：

- `target`：GRB 目标选择，默认沿用目录表最近样本或当前默认目标。
- `grb_name`：GRB 名称，可编辑。
- `trigger_met`：触发时刻，可从 LAT 表自动读取并允许覆盖。
- `analysis_mode`：保持和后端统一的模式选择。
- `data_dir`：GBM 数据根目录。
- `result_root`：结果输出根目录。
- `gbm_start` / `gbm_stop`：光变横轴显示范围。
- `active_interval`：源时窗。
- `background_intervals`：本底时窗。
- `fixed_num_time_bins`：固定时间分 bin 数。
- `include_lat` / `lat_prob_threshold`：是否显示 LAT 以及阈值。
- `bands_kev` 或 `nai_bands_kev`：NaI 能段。
- `bgo_band_kev`：BGO 能段。
- `special_yaml` / `special_burst_name`：特殊暴参数。
- `plot_joint_lightcurve`：显式开关。

### 输出区

页面 2 显示：

- 生成的光变图。
- 实际采用的参数摘要。
- 结果保存路径。
- 运行失败时的错误信息。

## 前后端一致性设计

### 配置来源统一

页面 2 默认值必须按以下优先级生成：

1. 用户在页面中的手动输入。
2. 目录表 / LAT 表中的目标默认值。
3. `GRBProjectConfig` 中的项目默认值。
4. `lightcurves.py` 中的函数默认值。

### 参数传递统一

页面 2 的提交按钮触发时，应构造一个与后端一致的参数对象，并传给后端执行路径。建议新增或扩展如下字段：

- `GRBProjectConfig.plot_joint_lightcurve`
- `GRBProjectConfig.gbm_start`
- `GRBProjectConfig.gbm_stop`
- `GRBProjectConfig.gbm_display_pad_before_s`
- `GRBProjectConfig.gbm_display_pad_after_s`
- `GRBRunOverrides.gbm_start`
- `GRBRunOverrides.gbm_stop`
- `GRBRunOverrides.gbm_display_pad_before_s`
- `GRBRunOverrides.gbm_display_pad_after_s`

如果页面新增了目前 `config.py` 里尚未覆盖的光变相关字段，也应同步加入配置与覆盖参数，而不是只留在前端本地状态中。

### 后端记录统一

`project.py` 在运行结束后应将光变相关参数写入结果元数据中，至少包含：

- 目标信息：`bnname`、`grb_name`
- 绘图参数：`gbm_start`、`gbm_stop`、`active_interval`、`background_intervals`
- LAT 参数：`lat_prob_threshold`、`include_lat`
- 能段参数：`nai_bands_kev`、`bgo_band_kev`
- 特殊分段参数：`special_yaml`、`special_burst_name`
- 输出路径：`lightcurve_path`

这样即使用户只看后端结果目录，也能复原页面配置。

## 后端执行流程设计

页面 2 的执行建议复用现有分析流程中的同类配置整理方式，但单独调用光变绘图，不强制进入完整拟合流程。

推荐执行步骤：

1. 读取目录表与 LAT 表。
2. 根据目标生成默认值。
3. 组装运行配置与覆盖参数。
4. 构造光变图输入：
   - `bnname`
   - `grb_name`
   - `trigger_met`
   - `data_dir`
   - `lat_prob_bn_dir`
   - `gbm_start` / `gbm_stop`
   - `active_interval`
   - `background_intervals`
   - `special_time_segments`
   - `fixed_num_time_bins`
5. 调用 `plot_gbm_lat_lightcurve_figure(...)`。
6. 将输出图片路径和参数摘要写入 `run_metadata.json` 或等价文件。
7. 页面内展示图片与摘要。

## 结果目录约定

页面 2 应将光变图保存到与后端分析一致的结果根目录中，例如：

- `{result_root}/{grb_name}/{bnname}/{grb_name}_lightcurve.png`

如果光变图是通过已有分析结果重绘，则也应保持同样的保存规则，避免同一暴在前端和后端出现两个不同版本的图片。

## 运行状态与错误处理

- 如果目录表、LAT 表或数据路径加载失败，页面应给出明确错误提示。
- 如果 LAT prob 文件不存在，则允许退化为仅 GBM 光变图，并在页面与日志中说明原因。
- 如果探测器或 EBOUNDS 读取失败，应显示具体异常，不要吞错。
- 若输入参数与后端逻辑冲突，后端以实际执行参数为准，并把最终值记录到元数据。

## 测试与验证

至少验证以下场景：

1. 页面 1 仍可正常执行单次分析。
2. 页面 2 可正常根据默认值生成光变图。
3. 页面 2 手工修改参数后，生成图像与保存路径正确。
4. 结果目录中的元数据能完整看到页面上设置的参数。
5. 当 LAT 文件缺失时，页面能正常退化到 GBM 光变图。
6. 前端默认值与后端读取结果一致。

## 实现边界

本次实现应限制在现有 `web_app.py`、`project.py`、`config.py`、必要时少量调整 `lightcurves.py` 的范围内。若发现需要大量拆分模块，先保持最小可行改动，确保第二页面可用并且参数可追踪。

## 结论

该方案以“页内路由 + 统一参数对象 + 后端元数据回写”为核心，能够满足：

- 同一个 Streamlit 入口下有第二个页面。
- 前端可调参数在后端可见、可追踪。
- 光变图逻辑与现有后端一致。
- 改动尽量集中，便于维护。
